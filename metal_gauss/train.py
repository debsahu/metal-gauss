"""3DGS trainer: metal-gauss rasterizer + MCMC densification.

    python -m metal_gauss.train --colmap sparse/0 --images images/ --steps 10000

Design notes, measurement-first: loss (0.8 L1 + 0.2 (1-SSIM)) and Adam run in
torch on MPS -- they are fused later only if profiling shows them hot. The
rasterizer is the fused-Metal fwd+bwd path. Densification is MCMC (fixed
budget, relocation + exploration noise) rather than ADC growth, so step time
stays flat.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from metal_gauss import render
from metal_gauss.dataset import Scene, downscaled, load_scene
from metal_gauss.geometry_loss import (depth_loss, depth_normal_loss, flatten_loss,
                                      normal_loss, normals_from_depth, splat_normals_cam)
from metal_gauss.priors import decode_depth, decode_normal
from metal_gauss.schedule import auto_budget  # noqa: F401  (re-exported)
from metal_gauss.appearance import AppearanceModel
from metal_gauss.mcmc import add_noise, grow, relocate
from metal_gauss.mipfilter import apply_3d_filter, compute_3d_filter

C0 = 0.28209479177387814


# ---------------------------------------------------------------- SSIM (torch)

def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device="mps", groups: int = 15):
    """1D gaussian window, shaped for a batched separable grouped conv2d.

    `groups` is 15 because SSIM needs five blurred quantities (x, y, x^2, y^2,
    x*y) of three channels each, and they are blurred in ONE grouped
    convolution rather than five separate calls.
    """
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return (g / g.sum()).view(1, 1, 1, size).expand(groups, 1, 1, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """a, b: (H,W,3) in [0,1]. SSIM, 11-tap sigma=1.5 gaussian.

    Two optimisations over the textbook form, neither of which changes the
    result: the gaussian window is separable, so an 11x11 convolution becomes
    two 11-tap passes; and the five quantities SSIM needs are stacked into one
    15-channel grouped convolution, so the whole thing costs 2 dispatches
    instead of 10. Profiling put this loss at 39% of a training step -- more
    than the rasteriser -- almost all of it dispatch overhead rather than
    arithmetic.
    """
    return ssim_map(a, b, kernel).mean()


def ssim_map(a: torch.Tensor, b: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Per-pixel SSIM, before the mean. `ssim_tail_forward` returns (1,3,H,W) -- NOT
    the (H,W,3) layout the images are in -- so anything multiplied into it has to be
    broadcast to that layout, which `_bcast` does."""
    return _SSIMFused.apply(a.contiguous(), b.contiguous(),
                            kernel[0, 0, 0].contiguous())


def _bcast(mask01: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """(H,W) keep-mask reshaped to multiply `like` elementwise.

    (H,W,3) images take a trailing axis; the SSIM map's (1,3,H,W) takes two LEADING
    axes. Using the trailing form on the SSIM map silently mis-broadcasts -- and on a
    square image it does not even raise, it just multiplies the wrong pixels.
    """
    if like.ndim == 4:                      # (1,3,H,W) SSIM map
        return mask01[None, None]
    if like.ndim == 3:                      # (H,W,3) image
        return mask01[..., None]
    return mask01


def photometric_loss(rgb_c: torch.Tensor, gt: torch.Tensor,
                     mask01: torch.Tensor | None, kernel: torch.Tensor,
                     return_terms: bool = False):
    """0.8 L1 + 0.2 (1 - SSIM), per pixel, multiplied by the KEEP mask (1 = keep),
    then meaned over ALL pixels.

    The mask zeroes the NUMERATOR only -- there is deliberately no division by
    coverage here, matching Brush's `image_loss`. `masked_mse` below DOES divide,
    because a PSNR that did not would be inflated by -10 log10(coverage). The
    asymmetry is intentional: a training loss is a descent direction whose overall
    scale is absorbed by the learning rate, while a reported metric is a number
    someone compares against a 24 dB gate.

    Masking cannot be exact at a mask boundary: SSIM's 11-tap window means a kept
    pixel within 5 px of a dropped one still sees it. That is a property of windowed
    SSIM, not of this implementation (CLAUDE.md Stage 4 says the same of masked SSIM).
    """
    l1 = (rgb_c - gt).abs()
    ds = 1.0 - ssim_map(rgb_c, gt, kernel)
    if mask01 is not None:
        l1 = l1 * _bcast(mask01, l1)
        ds = ds * _bcast(mask01, ds)
    terms = {"l1": l1.mean(), "ssim": ds.mean()}
    loss = 0.8 * terms["l1"] + 0.2 * terms["ssim"]
    return (loss, terms) if return_terms else loss


def masked_mse(err2: torch.Tensor, mask01: torch.Tensor | None) -> torch.Tensor:
    """mean(err2 * keep) / mean(keep). `mask01` None -> plain mean.

    The divisor is the whole point. Without it, masked-out pixels leave the numerator
    but not the denominator, so PSNR reads -10 log10(coverage) too high: +2.3 dB at
    59% coverage. Coverage is clamped so a fully dropped frame scores 0 rather than
    an infinite PSNR that would drag the held-out mean up.
    """
    if mask01 is None:
        return err2.mean()
    cov = mask01.mean().clamp_min(1e-6)
    return (err2 * _bcast(mask01, err2)).mean() / cov


def mask_void_warning(masks_supplied: bool, coverage: float) -> str | None:
    """The one-line instrument that makes CLAUDE.md's 49.561 dB impossible to repeat.

    `mean coverage 100.0%` on a run that supplied masks means the masks never reached
    the loss: the trainer scored every pixel, trained the operator and the monopod in,
    and its "masked" PSNR is identical to its unmasked one by construction. That run
    (splat_3840v2_full) spent 47,037 s of GPU and its 2,500-step authorising gate was
    measured on the same maskless dataset.

    Silent when no masks were supplied -- an unmasked dataset legitimately reads 100%,
    and a warning there just teaches operators to ignore the line.
    """
    if not masks_supplied or coverage < 0.9999:
        return None
    return ("  [MASKS] mean coverage 100.0% on a run that supplied masks: the masks "
            "did not reach the loss. Treat this run as VOID "
            "(see CLAUDE.md Stage 3, splat_3840v2_full).")


class _SSIMFused(torch.autograd.Function):
    """Whole SSIM in Metal: stack construction, both blur passes, and the tail.

    Replaces cat + two grouped conv2d + the tail. Three things go away:
      * the (1,15,H,W) stack, 86 MB at 900x1600, materialised every step;
      * the permute to (1,3,H,W) and its contiguous copy for BOTH inputs, since
        the kernels read (H,W,3) -- the layout render() and the ground truth
        already have;
      * torch's four convolution dispatches and their autograd graph.

    A symmetric gaussian is self-adjoint, so `ssim_blur15` is reused unchanged
    for the forward vertical pass and for both adjoint passes in the backward.
    Only the stack construction needs a separate forward and chain kernel.

    Zero padding matches F.conv2d exactly -- taps off the edge contribute
    nothing. A mismatch there only shows in a 5px border, which a mean-reduced
    loss hides well.
    """

    @staticmethod
    def forward(ctx, a, b, w):
        from metal_gauss.metal_backend import _load
        ext = _load()
        H, W = a.shape[0], a.shape[1]
        inter = ext.ssim_stack_blur_h(a, b, w, H, W)
        blurred = ext.ssim_blur15(inter, w, 1, H, W)
        ctx.save_for_backward(a, b, w, blurred)
        return ext.ssim_tail_forward(blurred, H, W)

    @staticmethod
    def backward(ctx, grad_out):
        from metal_gauss.metal_backend import _load
        ext = _load()
        a, b, w, blurred = ctx.saved_tensors
        H, W = a.shape[0], a.shape[1]
        d_blur = ext.ssim_tail_backward(blurred, grad_out.contiguous(), H, W)
        d_inter = ext.ssim_blur15(d_blur, w, 1, H, W)     # vertical adjoint
        d_stack = ext.ssim_blur15(d_inter, w, 0, H, W)    # horizontal adjoint
        return ext.ssim_chain_backward(a, b, d_stack, H, W), None, None


# ---------------------------------------------------------------- init

def scene_extent(scene: Scene) -> float:
    """Radius of the camera cloud -- the scale every position LR is relative to."""
    C = np.stack([(-v.viewmat[:3, :3].T @ v.viewmat[:3, 3]).numpy()
                  for v in scene.train])
    return float(np.linalg.norm(C - C.mean(0), axis=1).max())


def init_params(scene: Scene, budget: int, device: str) -> dict:
    pts = scene.points
    cols = scene.colors
    if len(pts) > budget:
        sel = np.random.default_rng(0).choice(len(pts), budget, replace=False)
        pts, cols = pts[sel], cols[sel]
    n = len(pts)

    # knn-ish scale init: median distance to 3 nearest of a random subset
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=4)
    s0 = np.clip(d[:, 1:].mean(1), 1e-4, None)
    # Isolated sparse points get huge knn distances -> screen-covering splats
    # that made early steps ~10x slower (every tile walks them every frame).
    s0 = np.minimum(s0, np.quantile(s0, 0.9))

    means = torch.tensor(pts, dtype=torch.float32, device=device)
    if n < budget:                      # fill the budget with jittered copies
        extra = budget - n
        idx = torch.randint(0, n, (extra,), device=device)
        jit = torch.randn(extra, 3, device=device) * \
            torch.tensor(s0, dtype=torch.float32, device=device)[idx][:, None]
        means = torch.cat([means, means[idx] + jit])
        cols = np.concatenate([cols, cols[np.asarray(idx.cpu())]])
        s0 = np.concatenate([s0, s0[np.asarray(idx.cpu())]])

    sh = torch.zeros(budget, 16, 3, device=device)
    sh[:, 0] = (torch.tensor(cols, dtype=torch.float32, device=device) - 0.5) / C0

    return {
        "means": means.requires_grad_(True),
        "log_scales": torch.log(torch.tensor(
            np.repeat(s0[:, None], 3, 1), dtype=torch.float32, device=device)
        ).requires_grad_(True),
        "quats": torch.cat([torch.ones(budget, 1, device=device),
                            torch.zeros(budget, 3, device=device)], 1).requires_grad_(True),
        "logit_opac": torch.full((budget,), -2.0, device=device).requires_grad_(True),
        "sh": sh.requires_grad_(True),
    }



def split_sh(p: dict) -> dict:
    """DC band and bands 1+ as separate leaves, for their different LRs."""
    p["sh_dc"] = p["sh"][:, :1].detach().clone().requires_grad_(True)
    p["sh_rest"] = p["sh"][:, 1:].detach().clone().requires_grad_(True)
    del p["sh"]
    return p


def make_optimizer(p: dict, lr_means0: float, *, lr_opac: float = 1e-2,
                   sh_lr_div: float = 20.0, selective: bool = False,
                   fused: bool = True):
    groups = [
        {"params": [p["means"]], "lr": lr_means0, "name": "means"},
        {"params": [p["log_scales"]], "lr": 5e-3, "name": "scales"},
        {"params": [p["quats"]], "lr": 1e-3, "name": "quats"},
        {"params": [p["logit_opac"]], "lr": lr_opac, "name": "opac"},
        {"params": [p["sh_dc"]], "lr": 2.5e-3, "name": "sh_dc"},
        {"params": [p["sh_rest"]], "lr": 2.5e-3 / sh_lr_div, "name": "sh_rest"},
    ]
    if selective:
        from metal_gauss.selective_adam import SelectiveAdam
        return SelectiveAdam(groups, eps=1e-15)
    if fused:
        from metal_gauss.fused_adam import FusedAdam
        return FusedAdam(groups, eps=1e-15)
    return torch.optim.Adam(groups, eps=1e-15)


def geometry_view_coverage(args, views) -> dict:
    """{term: (n_views_actually_supervised, n_views)} for each ENABLED geometry term.

    The startup check proves at least ONE view carries each prior; it says nothing about
    how many. A dataset with 10 of 196 views carrying depth passes it, trains with depth
    supervision on 5% of steps, and produces a log indistinguishable from full coverage.
    CLAUDE.md records exactly this in production -- "24 of 276 faces trained
    photometric-only", caused by a stale gate and found long after the fact.

    `depth_normal` compares the render against itself and needs no prior, so it is always
    fully covered.
    """
    n = len(views)
    cov = {}
    if args.depth_loss_weight > 0:
        cov["depth"] = (sum(v.depth is not None for v in views), n)
    if args.normal_loss_weight > 0:
        cov["normal"] = (sum(v.normal is not None for v in views), n)
    if args.depth_normal_weight > 0:
        cov["depth_normal"] = (n, n)
    return cov


def geometry_coverage_warning(cov: dict) -> str | None:
    """Loud line when a term supervises only some views. Silent at full coverage, as the
    mask warning is silent on an unmasked dataset -- a warning that always fires is one
    operators learn to skip."""
    partial = [(k, a, b) for k, (a, b) in cov.items() if 0 < a < b]
    if not partial:
        return None
    bits = ", ".join(f"{k} {a}/{b} views ({100.0 * a / b:.1f}%)" for k, a, b in partial)
    return ("  [PRIORS] PARTIAL geometry supervision: " + bits +
            ". The remaining views train photometric-only and nothing else in this log "
            "says so (see CLAUDE.md Stage 3, '24 of 276 faces trained photometric-only').")


def geometry_aux(means, quats, scales, viewmat):
    """The two aux attribute maps the geometry recipe composites: camera-frame splat
    normals and view-space z.

    `scales` MUST be the same tensor handed to `render` (post-`filter_3d`), so the
    thin-axis choice describes the ellipsoid actually rasterized.

    z comes from torch, NEVER from the preprocess's own `depth` output.
    `_PreprocessMetal.backward` accepts `d_depth` and DROPS it (metal_backend.py:267-279),
    so a depth map built on that output has no gradient path: it trains nothing and does
    not error. Pinned by tests/test_aux_render.py.
    """
    vmd = viewmat.to(means.device)
    n_cam = splat_normals_cam(means, quats, scales, vmd)
    z = (means @ vmd[:3, :3].T + vmd[:3, 3])[:, 2:3].expand(-1, 3)
    return [n_cam, z]


def geometry_terms(args, aux_maps, alpha, K, gt_depth=None, gt_normal=None, keep=None):
    """The depth / normal / depth-normal terms over one view's aux maps.

    ALPHA IS DETACHED. The terms divide by coverage to recover an attribute from its
    alpha-weighted composite; a differentiable divisor would make FADING A SPLAT OUT a
    cheaper way to satisfy a depth loss than moving it. That is the mechanism behind
    Brush's banned `--depth-source plane-fused`, where opacity p50 collapses ~30%.

    `keep` is a BINARY mask. Multiplying a metric depth by a fractional alpha would invent
    depths between 0 and the truth, and a smaller depth is a different surface, not a less
    certain one.
    """
    alpha_d = alpha.detach()
    n_img, z_img = aux_maps
    n_img = n_img / alpha_d.clamp_min(1e-10)[..., None]
    n_img = n_img / n_img.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    depth_img = z_img[..., 0] / alpha_d.clamp_min(1e-10)
    terms = {}
    if args.depth_loss_weight > 0 and gt_depth is not None:
        gt_d = gt_depth if keep is None else gt_depth * keep
        terms["depth"] = depth_loss(depth_img, gt_d, args.depth_loss_space)
    if args.normal_loss_weight > 0 and gt_normal is not None:
        gt_n = gt_normal if keep is None else gt_normal * keep[..., None]
        terms["normal"] = normal_loss(n_img, gt_n)
    if args.depth_normal_weight > 0:
        n_d = normals_from_depth(depth_img, K[0, 0].item(), K[1, 1].item(),
                                 K[0, 2].item(), K[1, 2].item())
        terms["depth_normal"] = depth_normal_loss(
            n_d, n_img, alpha_d if keep is None else alpha_d * keep)
    return terms


def render_view(p: dict, v, active: int, sh_deg: int = 3,
                background=(0.0, 0.0, 0.0), antialias: bool = False,
                absgrad_out=None, filter_3d=None, want_geometry: bool = False):
    """One rendered view, exactly as training renders it.

    v.K and v.viewmat stay on the HOST on purpose: the kernel reads both there,
    and passing MPS tensors makes every call drain the queue.
    """
    H, W = v.image.shape[:2]
    scales = torch.exp(p["log_scales"][:active])
    opac = torch.sigmoid(p["logit_opac"][:active])
    if filter_3d is not None:
        # Mip-Splatting's 3D low-pass. Applied here rather than in the kernel
        # because it is a plain reparameterisation of scale and opacity, so
        # autograd carries it and no adjoint has to be written.
        scales, opac = apply_3d_filter(scales, opac, filter_3d[:active])
    aux = geometry_aux(p["means"][:active], p["quats"][:active], scales,
                       v.viewmat) if want_geometry else None
    return render(
        p["means"][:active], p["quats"][:active],
        scales, opac, p["sh_dc"][:active],
        v.K, v.viewmat, W, H,
        sh_degree=sh_deg, backend="metal", sh_rest=p["sh_rest"][:active],
        background=background, antialias=antialias, absgrad_out=absgrad_out,
        aux_colors=aux)


# ---------------------------------------------------------------- training

def train(args, scene: Scene | None = None) -> dict:
    env = _env_snapshot()          # BEFORE any work: this is the provenance of the run
    device = "mps"
    # Seed every torch RNG the run touches: the init jitter, relocate/grow's
    # multinomial, and add_noise's randn. The numpy streams (point subsample,
    # view order) were already fixed at 0.
    #
    # The point is PAIRED comparison. Two policies run at the same seed share
    # their draws, so the only difference between them is the thing under test.
    # Unpaired, the 10k/1600p configuration has a 0.49 dB run-to-run spread --
    # wider than any lever measured here, and wide enough that a single sample
    # produced two confidently wrong conclusions in one session.
    #
    # This does NOT make runs bit-reproducible. rasterize_backward accumulates
    # through atomics whose order is not controllable, worth ~1e-10 a step, and
    # 10k steps of that compounds. Seeding removes the large term, not all of it.
    torch.manual_seed(args.seed)
    if hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(args.seed)
    if scene is not None:
        # A pre-built Scene (tests, and any caller that already has one). Blender's white
        # background is not implied here; a supplied scene uses the COLMAP convention.
        bg = (0.0, 0.0, 0.0)
    elif args.blender:
        from metal_gauss.blender import load_blender
        scene = load_blender(args.blender, args.max_resolution)
        # The Blender PNGs are composited over WHITE, so the renderer must
        # fill unoccupied pixels with white too. Leaving the default black
        # background here cost ~10 dB on lego and looks exactly like a broken
        # model rather than a convention mismatch.
        bg = (1.0, 1.0, 1.0)
    else:
        scene = load_scene(args.colmap, args.images, args.max_resolution,
                           args.eval_split_every,
                           masks_dir=getattr(args, "masks", None),
                           mask_polarity=getattr(args, "mask_polarity", "auto"),
                           depth_dir=getattr(args, "depth_dir", None),
                           normal_dir=getattr(args, "normal_dir", None),
                           prior_resident=getattr(args, "prior_resident", "quantized"),
                           use_priors=not getattr(args, "no_priors", False),
                           init_ply=getattr(args, "init_ply", None))
        bg = (0.0, 0.0, 0.0)
    print(f"{len(scene.train)} train views, {len(scene.heldout)} held out, "
          f"{len(scene.points):,} sparse points, budget {args.budget:,}")
    # "supplied" means a mask source was present, from EITHER convention: an explicit
    # --masks directory, or alpha baked into the images (which needs no flag at all).
    masks_supplied = any(v.mask is not None for v in scene.train + scene.heldout)
    if masks_supplied:
        n_m = sum(v.mask is not None for v in scene.train + scene.heldout)
        print(f"masked supervision active on {n_m}/"
              f"{len(scene.train) + len(scene.heldout)} views")

    # Refuse to start rather than silently train without the supervision the run was
    # configured for: a run that quietly drops a term looks exactly like one that kept it.
    for flag, attr in (("--depth-loss-weight", "depth"), ("--normal-loss-weight", "normal")):
        w = getattr(args, flag[2:].replace("-", "_"))
        if w > 0 and not any(getattr(v, attr) is not None for v in scene.train):
            raise RuntimeError(
                f"{flag} {w} but no view carries a {attr} prior -- refusing to train "
                f"without the supervision it was configured for. Point --{attr}-dir at "
                f"the priors, or set the weight to 0.")
    want_geometry = (args.depth_loss_weight > 0 or args.normal_loss_weight > 0
                     or args.depth_normal_weight > 0)
    term_cov = geometry_view_coverage(args, scene.train)
    if term_cov:
        print("geometry supervision: " + ", ".join(
            f"{k} {a}/{b} views" for k, (a, b) in term_cov.items()))
        cov_warn = geometry_coverage_warning(term_cov)
        if cov_warn:
            print(cov_warn, flush=True)

    # Selective Adam was built, validated, and MEASURED SLOWER here: at this
    # scene's per-view visibility (well above 50%), 15 gather/scatter launches
    # (or the masked-dense where-chain) cost more than dense Adam's 69 ms.
    # It stays available via --selective-adam for low-visibility regimes;
    # measurement-first means the default is the thing that is actually fast.
    p = init_params(scene, args.budget, device)
    # Position LR is relative to scene scale, and SH bands 1+ train 20x colder
    # than DC: with ~160 views, hot high-band SH memorises per-view shading and
    # costs held-out PSNR. Both conventions come from Inria 3DGS / Brush.
    extent = scene_extent(scene)
    lr_means0 = args.lr_means * extent if args.lr_scene_scaled else args.lr_means
    print(f"scene extent {extent:.2f}, initial means lr {lr_means0:.2e}")

    p = split_sh(p)
    opt = make_optimizer(p, lr_means0, lr_opac=args.lr_opac,
                         sh_lr_div=args.sh_lr_div,
                         selective=args.selective_adam, fused=args.fused_adam)

    appearance = None
    if args.appearance != "off":
        appearance = AppearanceModel(len(scene.train), args.appearance, device)
        opt.add_param_group({"params": list(appearance.parameters()),
                             "lr": args.lr_appearance, "name": "appearance"})
        print(f"appearance correction: {args.appearance} "
              f"({sum(x.numel() for x in appearance.parameters())} params "
              f"over {len(scene.train)} training images)")

    kernel = _gaussian_kernel(device=device)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(scene.train))
    oi = 0
    t0 = time.perf_counter()
    log = []

    # Capacity ramp: start small (cheap early steps) and grow to the cap by
    # grow_until. This is ADC's curriculum without ADC's bookkeeping -- and it
    # is a speed lever as much as a quality one.
    active = min(args.start_active, args.budget) if args.grow else args.budget
    grow_until = int(args.grow_until_frac * args.steps)
    lr_decay = (args.lr_means_end / max(args.lr_means, 1e-12)) ** (1.0 / args.steps)
    relocate_until = int(args.relocate_until_frac * args.steps)

    # Screen-space positional gradient, accumulated per gaussian between
    # densification events. 3DGS-ADC's signal: a gaussian whose projected
    # centre is being pulled hard is straddling detail it cannot represent, and
    # is the one worth splitting. Averaged over the views that actually saw it,
    # because a gaussian visible in 3 views must not outrank one visible in 30.
    use_absgrad = args.densify_weight == "absgrad"
    filter_3d = None
    f3d_every = args.filter_3d_every or args.relocate_every
    if args.filter_3d:
        filter_3d = compute_3d_filter(p["means"], scene.train)
    grad_sum = torch.zeros(args.budget, device=device)
    grad_hits = torch.zeros(args.budget, device=device)

    step_times: list[float] = []
    for step in range(1, args.steps + 1):
        t_step = time.perf_counter()
        if oi >= len(order):
            order = rng.permutation(len(scene.train))
            oi = 0
        view_idx = int(order[oi])
        v = scene.train[view_idx]
        oi += 1

        # anneal the position LR (and with it the exploration noise)
        cur_lr_means = lr_means0 * (lr_decay ** step)
        opt.param_groups[0]["lr"] = cur_lr_means
        # SH degree warmup: +1 band per warmup interval
        sh_deg = min(3, step // args.sh_warmup) if args.sh_warmup else 3

        # Coarse-to-fine: start at 1/2^num_downscales and double on a schedule.
        # Off by default (num_downscales=0) until the Pareto curve says it wins;
        # msplat's speed is roughly half this and half per-step cost.
        if args.num_downscales > 0:
            lvl = max(0, args.num_downscales - step // max(1, args.resolution_schedule))
            v = downscaled(v, 1 << lvl)
        H, W = v.image.shape[:2]
        # absgrad is accumulated by the backward kernel straight into grad_sum,
        # which is exactly the accumulator the other criteria fill by hand and
        # is zeroed on the same schedule.
        if filter_3d is not None and step % f3d_every == 0:
            # the gaussians have moved since the last refresh
            filter_3d = compute_3d_filter(p["means"], scene.train)
        rgb, alpha, info = render_view(p, v, active, sh_deg, bg,
                                       antialias=args.antialias,
                                       absgrad_out=grad_sum if use_absgrad else None,
                                       filter_3d=filter_3d,
                                       want_geometry=want_geometry)
        gt = v.image.to(device).float() / 255.0
        # Correct the RENDER (not the ground truth) so the exported splat stays
        # in the true photometric space and held-out views need no transform.
        rgb_c = appearance(rgb, view_idx) if appearance is not None else rgb
        m01 = None if v.mask is None else v.mask.to(device).float() / 255.0

        loss, terms = photometric_loss(rgb_c, gt, m01, kernel, return_terms=True)
        # PlanarGS flatten. NOT ramped down with `aux`: it is a geometry prior on the
        # final model, not an early-growth regulariser, and Brush applies it at constant
        # weight for the whole schedule. ADDED EXACTLY ONCE -- a second add lived below
        # the MCMC regularisers until 2026-09-02 and doubled every flatten run in this
        # project; tests pin term multiplicity now.
        if args.flatten_loss_weight > 0.0:
            terms["flatten"] = flatten_loss(p["log_scales"][:active])
            loss = loss + args.flatten_loss_weight * terms["flatten"]
        if want_geometry:
            keep = None if m01 is None else (m01 > 0.5)
            gt_d = decode_depth(v.depth.to(device)) if v.depth is not None else None
            gt_n = decode_normal(v.normal.to(device)) if v.normal is not None else None
            g = geometry_terms(args, info["aux"], alpha, v.K, gt_d, gt_n, keep)
            terms.update(g)
            for name, w in (("depth", args.depth_loss_weight),
                            ("normal", args.normal_loss_weight),
                            ("depth_normal", args.depth_normal_weight)):
                if name in g:
                    loss = loss + w * g[name]
        if appearance is not None:
            loss = loss + args.appearance_reg * appearance.regulariser()
        # MCMC regularisers: keep opacity and scale mass in check
        # Regularisers ramp down: they exist to keep early growth honest, not
        # to fight the reconstruction at convergence (Brush's aux_loss_weight).
        aux = max(0.0, 1.0 - step / (0.9 * args.steps))
        loss = loss + aux * (args.opac_reg * torch.sigmoid(p["logit_opac"][:active]).mean()
                             + args.scale_reg * torch.exp(p["log_scales"][:active]).mean())

        opt.zero_grad(set_to_none=True) if not args.selective_adam else opt.zero_grad()
        loss.backward()
        opt.step(info["valid_mask"]) if args.selective_adam else opt.step()

        if args.densify_weight != "opacity":
            with torch.no_grad():
                if use_absgrad:
                    # grad_sum was filled by the kernel; only the view count is
                    # left to record.
                    grad_hits[:active] += info["valid_mask"].to(grad_sum.dtype)
                else:
                    guv = info["uv"].grad
                    if guv is not None:
                        grad_sum[:active] += guv.norm(dim=1)
                        grad_hits[:active] += info["valid_mask"].to(grad_sum.dtype)

        with torch.no_grad():
            add_noise(p, cur_lr_means, args.noise_weight, active=active)
            densify_now = step % args.relocate_every == 0
            w = None
            if densify_now and args.densify_weight != "opacity":
                avg_grad = grad_sum / grad_hits.clamp_min(1.0)
                if args.densify_weight in ("grad", "absgrad"):
                    # same consumer: absgrad only changes how grad_sum was
                    # filled, not what relocation does with it.
                    w = avg_grad
                elif args.densify_weight == "opacity_grad":
                    # Opacity keeps MCMC's requirement that a target be a real
                    # gaussian worth splitting; the gradient says where the
                    # reconstruction is straining. Normalised so the product is
                    # not dominated by whichever term happens to be larger.
                    g = avg_grad / avg_grad.max().clamp_min(1e-12)
                    w = torch.sigmoid(p["logit_opac"]) * g
                else:                                   # "grad_gate"
                    # ADC densifies everything ABOVE a gradient threshold, and
                    # treats those equally. Sampling proportional to the
                    # gradient instead is far more concentrated, and gets worse
                    # with resolution: the p99/p50 ratio of the per-gaussian
                    # gradient is 93 at 270p and 212 at 1600p, so the top 1% of
                    # gaussians absorb 25% of the sampling mass at 270p but 37%
                    # at 1600p. That is the mechanism behind opacity_grad
                    # winning at 270p and losing at 1600p. A gate keeps the
                    # "where" signal without the concentration.
                    live = avg_grad[avg_grad > 0]
                    if live.numel() > 0:
                        thr = torch.quantile(live.float(),
                                             1.0 - args.densify_gate_frac)
                        w = torch.sigmoid(p["logit_opac"]) * (avg_grad > thr)
                    else:
                        w = None
            if densify_now and step < relocate_until:
                moved = relocate(p, opt=opt, active=active, weights=w)
                if moved:
                    log.append({"step": step, "relocated": moved})
            if args.grow and densify_now and step <= grow_until:
                target = int(args.start_active + (args.budget - args.start_active)
                             * min(1.0, step / max(grow_until, 1)))
                new_active = grow(p, target, active, opt=opt, weights=w)
                if new_active != active:
                    log.append({"step": step, "active": new_active})
                    active = new_active
            if densify_now:
                grad_sum.zero_(); grad_hits.zero_()

        # MPS driver-side allocations grow without bound if the queue never
        # drains: a 10k-step run died at ~25GB of "other allocations" before
        # the first eval. Sync + drain periodically; costs ~ms, saves the run.
        if step % 50 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()
            if step % 200 == 0:
                da = torch.mps.driver_allocated_memory() / 1e9
                if da > 20.0:
                    print(f"  [mem] driver {da:.1f} GB at step {step}", flush=True)

        # Stall detector. A step that suddenly takes far longer than the
        # running median means something outside the model changed: the machine
        # swapping, another job taking the GPU, or -- as happened here -- the
        # Mac going to sleep mid-run and the log simply stopping. Any of those
        # is worth a loud line, because a process that has quietly stopped
        # logging looks identical to one that is merely slow, and that
        # ambiguity already produced one confidently wrong diagnosis.
        # Long unattended runs should go under `caffeinate -i`.
        if step > 20:
            dt_step = time.perf_counter() - t_step
            step_times.append(dt_step)
            if len(step_times) > 50:
                step_times.pop(0)
            med = sorted(step_times)[len(step_times) // 2]
            if dt_step > max(10.0 * med, 30.0):
                print(f"  [STALL] step {step} took {dt_step:.1f}s against a "
                      f"{med * 1000:.0f} ms median. The machine is very likely "
                      f"swapping: {active:,} active gaussians at {W}x{H}. "
                      f"Reduce --budget or --max-resolution.", flush=True)

        if step == 100 or step % 500 == 0 and step % args.eval_every != 0:
            dt = time.perf_counter() - t0
            print(f"step {step:>6}  loss {loss.item():.4f}  ({1000 * dt / step:.0f} ms/step)",
                  flush=True)
        if args.export and args.export_every and step % args.export_every == 0:
            # written before the eval so the mtime reflects training time, not
            # training plus a 200-view evaluation
            ck = Path(args.export).with_suffix(f".step{step:06d}.ply")
            export_ply({k: (v[:active] if torch.is_tensor(v) else v)
                        for k, v in p.items()}, str(ck), filter_3d=filter_3d)

        if step % args.eval_every == 0 or step == args.steps:
            torch.mps.empty_cache()
            ev = evaluate(p, scene, device, sh_degree=sh_deg, active=active,
                          background=bg, antialias=args.antialias,
                          filter_3d=filter_3d,
                          dump_dir=getattr(args, "eval_dump", None))
            dt = time.perf_counter() - t0
            print(f"step {step:>6}  loss {loss.item():.4f}  "
                  f"heldout masked PSNR {ev['psnr_masked']:.2f} dB | "
                  f"unmasked (legacy) PSNR {ev['psnr']:.2f} | "
                  f"mean coverage {100 * ev['coverage']:.1f}%  "
                  f"{active/1000:.0f}k splats  {dt:.0f}s  ({1000 * dt / step:.0f} ms/step)",
                  flush=True)
            warn = mask_void_warning(masks_supplied, ev["coverage"])
            if warn:
                print(warn, flush=True)
            term_vals = {k: float(t.detach()) for k, t in terms.items()}
            print("  terms  " + "  ".join(f"{k} {t:.5f}" for k, t in term_vals.items()),
                  flush=True)
            log.append({"step": step, "psnr": ev["psnr"],
                        "psnr_masked": ev["psnr_masked"], "coverage": ev["coverage"],
                        "wall_s": round(dt, 1), "active": active, "terms": term_vals})

    out = _run_report(args, log, time.perf_counter() - t0, active, env=env)
    out["metrics"]["term_view_coverage"] = {k: [a, b] for k, (a, b) in term_cov.items()}
    out["metrics"]["term_coverage_warning"] = geometry_coverage_warning(term_cov)
    for dest in (args.out, getattr(args, "report", None)):
        if dest:
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_text(json.dumps(out, indent=2, default=str))
    if args.export:
        export_ply({k: (v[:active] if torch.is_tensor(v) else v) for k, v in p.items()},
                   args.export, filter_3d=filter_3d)
        print(f"exported {args.export}")
    return out


@torch.no_grad()
def evaluate(p, scene: Scene, device: str, max_views: int | None = None,
             background=(0.0, 0.0, 0.0),
             sh_degree: int = 3, active: int | None = None,
             antialias: bool = False, filter_3d=None, dump_dir=None) -> dict:
    """Held-out PSNR over ALL held-out views by default.

    This used to default to the first 10 of 21 views, which biased every
    number we reported by up to ~0.5 dB depending on which views happened to
    be easy. Metrics get fixed before levers get credited.

    Returns masked PSNR (the gate figure -- scored over KEPT pixels only), the
    legacy unmasked PSNR (every pixel, kept comparable with every number recorded
    before masks existed), and mean coverage. On an unmasked dataset all three are
    the old number, 1.0 and each other.
    """
    psnrs_m, psnrs_u, covs = [], [], []
    for v in (scene.heldout if max_views is None else scene.heldout[:max_views]):
        H, W = v.image.shape[:2]
        a = len(p["means"]) if active is None else active
        sh_full = torch.cat([p["sh_dc"], p["sh_rest"]], dim=1) if "sh_dc" in p else p["sh"]
        sc_e = torch.exp(p["log_scales"][:a])
        op_e = torch.sigmoid(p["logit_opac"][:a])
        if filter_3d is not None:
            sc_e, op_e = apply_3d_filter(sc_e, op_e, filter_3d[:a])
        rgb, _, _ = render(p["means"][:a], p["quats"][:a], sc_e, op_e, sh_full[:a],
                           v.K, v.viewmat, W, H,
                           sh_degree=sh_degree, backend="metal", background=background,
                           antialias=antialias)
        gt = v.image.to(device).float() / 255.0
        m01 = None if v.mask is None else v.mask.to(device).float() / 255.0
        err2 = (rgb.clamp(0, 1) - gt) ** 2
        psnrs_u.append(-10 * math.log10(max(err2.mean().item(), 1e-10)))
        psnrs_m.append(-10 * math.log10(max(masked_mse(err2, m01).item(), 1e-10)))
        covs.append(1.0 if m01 is None else m01.mean().item())
        if dump_dir is not None:
            from PIL import Image as _Im
            d = Path(dump_dir); d.mkdir(parents=True, exist_ok=True)
            stem = Path(v.name).stem
            def _u8(t):
                return (t.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            _Im.fromarray(_u8(rgb)).save(d / f"{stem}_render.png")
            _Im.fromarray(_u8(gt)).save(d / f"{stem}_gt.png")
            if v.mask is not None:
                _Im.fromarray(v.mask.cpu().numpy()).save(d / f"{stem}_mask.png")
    return {"psnr_masked": float(np.mean(psnrs_m)),
            "psnr": float(np.mean(psnrs_u)),
            "coverage": float(np.mean(covs))}


@torch.no_grad()
def export_ply(p, path: str, filter_3d=None) -> None:
    """Write an INRIA-convention ply.

    With `filter_3d`, Mip-Splatting's 3D low-pass is BAKED IN: the widened
    scales and dimmed opacity are written in the file's pre-activation space,
    so a viewer that knows nothing about the filter renders the model exactly
    as this trainer does. That is the property the 2D Mip filter lacks -- it is
    screen-space and cannot be folded into a ply at all, which is why
    --antialias cannot be a default while this can.
    """
    import plyfile

    n = len(p["means"])
    names = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(45)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.zeros(n, dtype=[(nm, "f4") for nm in names])
    m = p["means"].detach().cpu().numpy()
    data["x"], data["y"], data["z"] = m.T
    sh = (torch.cat([p["sh_dc"], p["sh_rest"]], dim=1) if "sh_dc" in p
          else p["sh"]).detach().cpu().numpy()
    for c in range(3):
        data[f"f_dc_{c}"] = sh[:, 0, c]
        for b in range(15):
            data[f"f_rest_{c * 15 + b}"] = sh[:, b + 1, c]
    logit = p["logit_opac"].detach()
    log_scales = p["log_scales"].detach()
    if filter_3d is not None:
        f = filter_3d[:n].detach()
        sc, op = apply_3d_filter(torch.exp(log_scales), torch.sigmoid(logit), f)
        log_scales = torch.log(sc.clamp_min(1e-12))
        # back to logit space; clamp keeps a fully suppressed gaussian finite
        op = op.clamp(1e-6, 1.0 - 1e-6)
        logit = torch.log(op / (1.0 - op))
    data["opacity"] = logit.cpu().numpy()
    ls = log_scales.cpu().numpy()
    for i in range(3):
        data[f"scale_{i}"] = ls[:, i]
    q = p["quats"].detach().cpu().numpy()
    for i in range(4):
        data[f"rot_{i}"] = q[:, i]
    plyfile.PlyData([plyfile.PlyElement.describe(data, "vertex")]).write(path)


def _env_snapshot() -> dict:
    """What code is about to run, captured BEFORE it runs.

    `_run_report` used to query git when it WROTE the report, which answers "what was
    checked out when this stopped?" -- not "what produced this number?". On 2026-09-02 an
    arm that started at 16:53 and finished at 17:09 recorded a commit made at 16:59, six
    minutes after Python had already imported the module it was executing. The report was
    the provenance mechanism for a five-arm measurement protocol, so it has to be right.
    """
    def _git(*a):
        try:
            return subprocess.run(("git",) + a, cwd=Path(__file__).resolve().parent,
                                  capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except Exception:
            return None

    return {
        "git": _git("rev-parse", "--short", "HEAD") or None,
        "dirty": bool(_git("status", "--porcelain")),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        # Microseconds, not seconds: a short run starts and finishes inside the same
        # second, and a timestamp pair that cannot order two events is not provenance.
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }


def _run_report(args, log, wall_s, active, env=None):
    """Everything needed to reproduce this run, recorded by the process that ran it.

    Records ALL of vars(args), not a curated subset. Curation is how knobs go
    unrecorded, and an unrecorded knob is how this project twice published a
    number produced by settings other than the ones it claimed: --steps-scaler
    once, then --budget, where a harness forwarded 300k to every child and
    auto_budget() never ran in any 8-scene sweep. Both were invisible because
    the harness recorded its own argparse namespace as "the protocol".

    args is read AFTER main() has resolved auto_budget, resolution_schedule,
    start_active clamping and the steps-scaler, so these are the values that
    actually ran, by construction rather than by convention.
    """
    env = dict(env) if env is not None else _env_snapshot()
    env["finished_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    resolved = {k: v for k, v in sorted(vars(args).items())}
    ms = (1000.0 * wall_s / args.steps) if args.steps else None
    return {
        "schema": 1,
        "resolved": resolved,
        "env": env,
        "metrics": {
            # `psnr` stays the UNMASKED number: readers of --out predate masks.
            "psnr": log[-1]["psnr"] if log else None,
            "psnr_masked": log[-1].get("psnr_masked") if log else None,
            "coverage": log[-1].get("coverage") if log else None,
            "terms": log[-1].get("terms") if log else None,
            "wall_s": round(wall_s, 1),
            "n_splats": int(active),
            "ms_per_step": round(ms, 2) if ms else None,
        },
        # kept at the old top-level keys so existing readers of --out still work
        "steps": args.steps,
        "budget": args.budget,
        "wall_clock_s": round(wall_s, 1),
        "final_psnr": log[-1]["psnr"] if log else None,
        "log": log,
    }


def build_parser() -> argparse.ArgumentParser:
    """The CLI, as a value. Tests build arms through THIS so they inherit every default
    from the one place the command line does -- a hand-written namespace is how a sweep
    once ran with settings other than the ones it reported."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap")
    ap.add_argument("--blender", help="NeRF-synthetic scene dir (transforms_*.json). "
                                      "Uses the published train/test split, for "
                                      "absolute calibration against reported numbers.")
    ap.add_argument("--images")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--budget", type=int, default=None,
                    help="gaussian capacity. Default scales with --steps; see "
                         "auto_budget(). Set explicitly to override.")
    ap.add_argument("--max-resolution", type=int, default=1600)
    ap.add_argument("--eval-split-every", type=int, default=8)
    ap.add_argument("--masks", default=None,
                    help="directory of sidecar masks, one per image by stem. Polarity "
                         "is decided for the DIRECTORY (see --mask-polarity). Images "
                         "that carry a baked RGBA alpha need no flag and must NOT also "
                         "have a sidecar -- that combination is an error, not a merge.")
    ap.add_argument("--mask-polarity", choices=["auto", "drop", "keep"], default="auto",
                    help="what 255 means in --masks. Every masks*/ directory in "
                         "earthbyte/slam is 255 = DROP; 'auto' reads the median white "
                         "fraction over a sample and picks, then prints what it picked.")
    ap.add_argument("--depth-dir", default=None,
                    help="directory of depth priors, one per image by stem (tiff-f32, "
                         "tiff-f32-deflate or png-quantized; formats may mix per frame). "
                         "Default: the sibling <images>/../depth if it exists. A prior "
                         "whose size differs from the LOADED image size is a hard error.")
    ap.add_argument("--normal-dir", default=None,
                    help="directory of normal priors. Default: the sibling "
                         "<images>/../normal if it exists.")
    ap.add_argument("--prior-resident", choices=["quantized", "float32"],
                    default="quantized",
                    help="how priors are held in RAM. 'quantized' is uint16 mm / uint8 "
                         "codes, 5 bytes/px against 16, and is training-equivalent per "
                         "the WS-G gate (0.0128 dB vs a 0.0317 dB same-seed repeat). "
                         "'float32' is the lossless escape hatch.")
    ap.add_argument("--init-ply", default=None,
                    help="seed the gaussians from this ply instead of the COLMAP model's "
                         "points3D. Needed when poses and seed live apart -- ARKitScenes "
                         "keeps 656 posed images with ZERO points3D and a separate 1.13M "
                         "point seed.ply. Colour is read from f_dc_* (SH DC) or red/green/blue.")
    ap.add_argument("--no-priors", action="store_true",
                    help="ignore depth/normal priors entirely, including the sibling "
                         "auto-detect. Without this, a dataset that HAS priors refuses to "
                         "start at any --max-resolution below the prior size; this is the "
                         "no-prior control arm.")
    ap.add_argument("--depth-loss-weight", type=float, default=0.0,
                    help="L1 on rendered vs prior depth. earthbyte indoor recipe: 1.0")
    ap.add_argument("--depth-loss-space", choices=["disparity", "metric"],
                    default="disparity",
                    help="disparity (1/z, Brush's default) weights near geometry more")
    ap.add_argument("--normal-loss-weight", type=float, default=0.0,
                    help="component L1 on rendered vs prior normals. Recipe: 0.2")
    ap.add_argument("--depth-normal-weight", type=float, default=0.0,
                    help="1 - cos between normals differentiated from the rendered depth "
                         "and the rendered normals. Needs NO prior data, so it is the "
                         "cheapest of the three to switch on. Recipe: 0.05")
    ap.add_argument("--eval-dump", default=None,
                    help="write <stem>_render.png / _gt.png / _mask.png per held-out "
                         "view at every eval, for LPIPS and for looking at.")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--relocate-every", type=int, default=100)
    ap.add_argument("--lr-means", type=float, default=2e-4)
    ap.add_argument("--noise-weight", type=float, default=4e4)
    ap.add_argument("--opac-reg", type=float, default=0.01)
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument("--flatten-loss-weight", type=float, default=0.0,
                    help="PlanarGS flatten: weight on the mean smallest ACTIVATED scale. "
                         "The earthbyte indoor recipe is 1.0, applied at constant weight "
                         "with no metric normalisation. Dominant measured geometry lever "
                         "in Brush (-14.3 deg thin-axis on playroom, -8.9 on ARKitScenes).")
    ap.add_argument("--appearance", choices=["off", "gain_bias", "affine"],
                    default="off",
                    help="per-training-image photometric correction; held-out "
                         "views always render with the identity transform")
    ap.add_argument("--lr-appearance", type=float, default=1e-3)
    ap.add_argument("--appearance-reg", type=float, default=1e-2)
    ap.add_argument("--lr-means-end", type=float, default=2e-6)
    ap.add_argument("--lr-scene-scaled", action="store_true", default=True)
    ap.add_argument("--lr-opac", type=float, default=1e-2)
    ap.add_argument("--sh-lr-div", type=float, default=20.0)
    ap.add_argument("--sh-warmup", type=int, default=1000,
                    help="steps per SH band; 0 disables warmup")
    ap.add_argument("--grow", action="store_true", default=True)
    ap.add_argument("--no-grow", dest="grow", action="store_false")
    ap.add_argument("--start-active", type=int, default=150_000)
    ap.add_argument("--grow-until-frac", type=float, default=0.7)
    ap.add_argument("--relocate-until-frac", type=float, default=0.85)
    # LichtFeld's steps_scaler: scale EVERY schedule together so a short run is
    # a proportional miniature of a long one, not a truncation. Without this,
    # rankings measured on a 1k-step run do not transfer to 10k.
    ap.add_argument("--steps-scaler", type=float, default=1.0)
    ap.add_argument("--selective-adam", action="store_true")
    ap.add_argument("--densify-weight",
                    choices=["opacity", "grad", "opacity_grad", "grad_gate",
                             "absgrad"],
                    default="opacity_grad",
                    help="what relocation and growth sample proportional to. "
                         "'opacity' is the 3DGS-MCMC policy; 'grad' is ADC's "
                         "screen-space positional gradient; 'opacity_grad' is "
                         "their product; 'grad_gate' restricts opacity "
                         "sampling to the top --densify-gate-frac by gradient, "
                         "which is closer to what ADC actually does.")
    ap.add_argument("--filter-3d", action="store_true",
                    help="Mip-Splatting's 3D smoothing filter: band-limit each "
                         "gaussian to the sampling rate of the views that see "
                         "it. Unlike --antialias this is view-independent, so "
                         "it is BAKED INTO THE EXPORTED PLY and any viewer "
                         "renders it correctly without cooperation.")
    ap.add_argument("--filter-3d-every", type=int, default=0,
                    help="steps between recomputing the 3D filter; 0 means use "
                         "--relocate-every. Gaussians move under MCMC "
                         "relocation, so a filter computed once at init goes "
                         "stale, but recomputing every step is waste.")
    ap.add_argument("--antialias", action="store_true",
                    help="Mip-Splatting / gsplat antialiased rasterisation: "
                         "rescale opacity so the low-pass dilation preserves "
                         "energy instead of eroding sub-pixel gaussians. "
                         "Matters most when render and training resolution "
                         "differ, which --num-downscales makes the default.")
    ap.add_argument("--num-downscales", type=int, default=2,
                    help="start training at 1/2^N resolution and double on "
                         "--resolution-schedule. 0 disables (full res throughout).")
    ap.add_argument("--resolution-schedule", type=int, default=None,
                    help="steps between resolution doublings. Default steps//3, "
                         "so full resolution is reached two thirds through.")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds torch RNG. Use the SAME seed on both arms of an "
                         "A/B so they share draws and only the tested variable "
                         "differs; atomics still make runs non-bit-reproducible.")
    ap.add_argument("--densify-gate-frac", type=float, default=0.30,
                    help="grad_gate: fraction of gaussians, by gradient, that "
                         "are eligible densification targets.")
    ap.add_argument("--fused-adam", action="store_true", default=True,
                    help="Adam in one Metal pass instead of torch's five (2.2x)")
    ap.add_argument("--no-fused-adam", dest="fused_adam", action="store_false")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None,
                    help="write the resolved config, env and metrics as JSON. "
                         "Benchmarks must read this instead of parsing stdout.")
    ap.add_argument("--export", default=None)
    ap.add_argument("--export-every", type=int, default=0,
                    help="also write a checkpoint ply every N steps, named "
                         "<export>.stepNNNNNN.ply. Used to build the wall-clock "
                         "convergence comparison: the checkpoint's mtime gives "
                         "its real elapsed time, which beats interpolating from "
                         "the step count when per-step cost is not constant.")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    if not args.blender and not (args.colmap and args.images):
        ap.error("need --blender, or --colmap with --images")
    if args.resolution_schedule is None:
        # Reach full resolution two thirds of the way in, whatever the run
        # length. Measured on lego at a 100k budget, scored on the official
        # test split:
        #     iters   full-res throughout   coarse-to-fine
        #      1000   23.89 @ 1.13 min      23.65 @ 0.74 min
        #      2000   26.32 @ 1.80          26.20 @ 1.17
        #      4000   28.49 @ 2.99          28.72 @ 1.93
        # ~35% faster everywhere, and the quality cost shrinks with run length
        # (-0.24, -0.12, +0.23 dB), so by 4k steps it wins on BOTH axes.
        args.resolution_schedule = max(1, args.steps // 3)
    if args.budget is None:
        args.budget = auto_budget(args.steps)
        print(f"budget {args.budget:,} (auto, from {args.steps} steps)")
    if args.start_active > args.budget:
        # the parameter tensors are preallocated at `budget`; an `active` above
        # that reads past the end.
        args.start_active = max(1000, args.budget // 2)
    if args.steps_scaler != 1.0:
        k = args.steps_scaler
        args.steps = max(1, int(args.steps * k))
        args.relocate_every = max(1, int(args.relocate_every * k))
        args.eval_every = max(1, int(args.eval_every * k))
        args.sh_warmup = max(1, int(args.sh_warmup * k)) if args.sh_warmup else 0
        print(f"steps_scaler {k}: {args.steps} steps, relocate every "
              f"{args.relocate_every}, sh warmup {args.sh_warmup}")
    train(args)


if __name__ == "__main__":
    main()
