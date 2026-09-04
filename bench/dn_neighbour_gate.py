#!/usr/bin/env python
"""Task 20 step 2: BOUND the depth-normal term's neighbour-gating semantic on real maps.

`keep` and `alpha > 0.5` gate the BASE pixel only, so `n_d(v,u)` can be built from a
dropped or uncovered neighbour's depth (research/depth-normal-loss-adjoint.md 2.5). This
measures how often that happens on REAL rendered maps and what it would cost to change,
against the pre-registered 0.5%-of-in-loss-pixels line. It trains nothing.

Per view it records:
  (a) the fraction of IN-LOSS dn pixels with at least one gated neighbour, split by cause
      (alpha vs keep; the two overlap and all three are reported);
  (b) both dn losses and both in-loss counts -- the gate changes the DENOMINATOR as well
      as the numerator, so a bare relative delta would conflate them;
  (c) `rel` (max|d|/max|ref|) and cosine of dL/d(z_img) and dL/d(n_sum) between the two
      rules, for the dn term alone AND for the weighted geometry loss the trainer actually
      backpropagates. The comparison line is the fused kernel's own measured f32 distance
      from truth, 7.02e-6 (research/metal-gauss.md 11.1).

PROVENANCE. Every output carries `kind`, `synthetic`, the dataset paths, the ply and its
sha256 prefix, the git commit, and the view names. `--summary` REFUSES any file that is
synthetic or lacks provenance, and it must be told its scenes by name: a summary that
globs a directory will happily average a smoke fixture into a result, which is how a
fabricated pass/regress pair once reached a Task 19 table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

KIND = "dn_neighbour_gate_measurement"
SCHEMA = 1
F32_KERNEL_DISTANCE_FROM_TRUTH = 7.02e-6      # research/metal-gauss.md 11.1
AFFECTED_FRACTION_LINE = 0.005                # pre-registered, Task 20 step 1


# ------------------------------------------------------------------ model construction

def params_from_ply(path: str, device: str) -> tuple[dict, int]:
    """The trainer's own parameter dict, read back from an INRIA ply in ITS OWN
    pre-activation space -- `opacity` is a logit, `scale_*` a log, `rot_*` unnormalised,
    exactly as `train.export_ply` wrote them. `io.load_ply` activates all three, so it is
    the wrong reader here: re-inverting a sigmoid loses the value at the rails.
    """
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]

    def col(n):
        return torch.from_numpy(np.asarray(v[n], dtype=np.float32).copy())

    n = len(v)
    sh = torch.zeros(n, 16, 3)
    for c in range(3):
        sh[:, 0, c] = col(f"f_dc_{c}")
        for b in range(15):
            sh[:, b + 1, c] = col(f"f_rest_{c * 15 + b}")
    p = {
        "means": torch.stack([col("x"), col("y"), col("z")], 1).to(device),
        "log_scales": torch.stack([col(f"scale_{i}") for i in range(3)], 1).to(device),
        "quats": torch.stack([col(f"rot_{i}") for i in range(4)], 1).to(device),
        "logit_opac": col("opacity").to(device),
        "sh_dc": sh[:, :1].to(device),
        "sh_rest": sh[:, 1:].to(device),
    }
    return p, n


def params_at_step0(scene, budget: int, seed: int, device: str) -> tuple[dict, int]:
    """The trainer's state at step 0, reproduced: `torch.manual_seed(seed)` then
    `init_params` then `split_sh` (train.py:461, :481-ish), and `active` set the way the
    growth ramp sets it, so this is the model that renders at step 1 and not a
    full-capacity model that never exists."""
    from metal_gauss.train import init_params, split_sh
    torch.manual_seed(seed)
    if hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    p = split_sh(init_params(scene, budget, device))
    return p, min(150_000, budget)              # train.py: min(args.start_active, budget)


# ------------------------------------------------------------------- the per-view maps

def view_maps(p, v, active, sh_deg, device):
    """`geometry_terms`' torch path, up to the point where the two rules diverge.

    n_sum and z_img are LEAVES here so the two rules' gradients can be compared; the
    rendered aux maps are detached from the splats first, which changes nothing about the
    pixel-space gradients being compared and keeps 500k-splat autograd out of the loop.
    """
    from metal_gauss.train import render_view
    with torch.no_grad():
        _rgb, alpha, info = render_view(p, v, active, sh_deg, want_geometry=True)
    n_sum, z_img = info["aux"][0].detach(), info["aux"][1].detach()
    # STRUCTURAL CHECK on the aux ORDER, which `geometry_aux` fixes as [n_cam, z]. `z` is
    # broadcast to three identical channels before compositing, so the composited z map has
    # three identical channels and the normal map does not. Swapping the two maps is
    # otherwise undetectable -- both are (H,W,3) float32 -- and would silently make every
    # number here describe a different quantity.
    if not torch.equal(z_img[..., 0], z_img[..., 1]):
        raise RuntimeError("aux[1] is not a 3x-replicated z map: aux order changed")
    if torch.equal(n_sum[..., 0], n_sum[..., 1]):
        raise RuntimeError("aux[0] has identical channels: it is not a normal map")
    return (n_sum.clone().requires_grad_(True),
            z_img.clone().requires_grad_(True), alpha.detach())


def _terms(n_sum, z_img, alpha, keep, K, gt_d, gt_n, gate: bool, space="disparity"):
    from metal_gauss.geometry_loss import (depth_loss, depth_normal_loss, normal_loss,
                                           normals_from_depth)
    a = alpha.clamp_min(1e-10)
    n_img = n_sum / a[..., None]
    n_img = n_img / n_img.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    depth_img = z_img[..., 0] / a
    k = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
    n_d = normals_from_depth(depth_img, *k)
    out = {"depth_normal": depth_normal_loss(
        n_d, n_img, alpha if keep is None else alpha * keep, gate_neighbours=gate)}
    if gt_d is not None:
        out["depth"] = depth_loss(depth_img, gt_d if keep is None else gt_d * keep, space)
    if gt_n is not None:
        out["normal"] = normal_loss(n_img, gt_n if keep is None else gt_n * keep[..., None])
    return out, n_d, n_img


def _grads(n_sum, z_img, loss):
    # retain_graph: each rule's graph is differentiated TWICE -- once for the dn term
    # alone and once for the weighted geometry loss the trainer backpropagates.
    g = torch.autograd.grad(loss, [z_img, n_sum], retain_graph=True, allow_unused=True)
    return [torch.zeros_like(t) if x is None else x.detach()
            for x, t in zip(g, (z_img, n_sum))]


def _rel_cos(got, ref):
    """`rel` (max-norm), `rel_l2` and cosine.

    `rel` is the repo's own bar shape (tests/test_fused_geom_loss.py `_bars`) and is what
    the plan asked for, so it is reported unchanged. Read it knowing what it can and
    cannot say HERE: it is the right statistic for two implementations of the SAME
    function, and a poor one for a rule change that REMOVES PIXELS FROM THE LOSS. A
    removed pixel's gradient becomes exactly zero, so if the largest-magnitude gradient in
    the frame sits on a removed pixel, `rel` is exactly 1.0 however few pixels moved. It
    cannot be compared to the kernel's 7.02e-6 f32 distance from truth in any useful way.

    `rel_l2` = ||got - ref|| / ||ref|| over the whole field is the one that answers "how
    much of the gradient field changed", and it is reported beside it. ADDED AFTER seeing
    the first max-norm result; it is extra information and it changes no threshold -- the
    pre-registered decision rule is on the affected FRACTION alone.
    """
    # .cpu() BEFORE .double(): MPS has no float64 and raises rather than casting.
    got, ref = got.detach().cpu().double(), ref.detach().cpu().double()
    ref_max, ref_l2 = float(ref.abs().max()), float(ref.norm())
    rel = float((got - ref).abs().max() / max(ref_max, 1e-30))
    rel_l2 = float((got - ref).norm() / max(ref_l2, 1e-30))
    cos = float(torch.nn.functional.cosine_similarity(
        got.flatten(), ref.flatten(), dim=0)) if ref_max > 0 else float("nan")
    return rel, rel_l2, cos


def affected_stats(in_loss, covered, alpha_ok, keep_ok) -> dict:
    """Counts for (a), separated from rendering so they can be unit-tested directly.

    `in_loss` is the shipped rule's pixel set. A pixel is AFFECTED when at least one of
    its two `normals_from_depth` neighbours -- (v,u+1) and (v+1,u) -- is not `covered`.
    The cause split is by which predicate the bad neighbour failed; a neighbour can fail
    both, so the three counts satisfy inclusion-exclusion and that identity is ASSERTED,
    not assumed.
    """
    from metal_gauss.geometry_loss import dn_neighbours_covered

    def _nb_bad(ok):
        b = torch.ones_like(ok)                     # border: no neighbours at all
        b[:-1, :-1] = (~ok[:-1, 1:]) | (~ok[1:, :-1])
        return b

    nb_ok = dn_neighbours_covered(covered)
    nb_alpha_bad, nb_keep_bad = _nb_bad(alpha_ok), _nb_bad(keep_ok)
    affected = in_loss & ~nb_ok
    n_in = int(in_loss.sum())
    out = {
        "in_loss_px": n_in,
        "affected_px": int(affected.sum()),
        "affected_by_alpha_px": int((affected & nb_alpha_bad).sum()),
        "affected_by_keep_px": int((affected & nb_keep_bad).sum()),
        "affected_by_both_px": int((affected & nb_alpha_bad & nb_keep_bad).sum()),
        "affected_frac": (int(affected.sum()) / n_in) if n_in else float("nan"),
    }
    if (out["affected_by_alpha_px"] + out["affected_by_keep_px"]
            - out["affected_by_both_px"]) != out["affected_px"]:
        raise RuntimeError(f"cause split does not partition the affected set: {out}")
    return out


def measure_view(p, v, active, sh_deg, device, weights) -> dict:
    from metal_gauss.priors import decode_depth, decode_normal

    n_sum, z_img, alpha = view_maps(p, v, active, sh_deg, device)
    keep = None if v.mask is None else (v.mask.to(device).float() / 255.0 > 0.5)
    gt_d = decode_depth(v.depth.to(device)) if v.depth is not None else None
    gt_n = decode_normal(v.normal.to(device)) if v.normal is not None else None
    K = v.K

    # --- (a) the population, under the SHIPPED rule
    keep_ok = torch.ones_like(alpha, dtype=torch.bool) if keep is None else keep
    alpha_ok = alpha > 0.5
    covered = (alpha if keep is None else alpha * keep) > 0.5
    with torch.no_grad():
        _t, n_d, n_img = _terms(n_sum, z_img, alpha, keep, K, gt_d, gt_n, gate=False)
        in_loss = (covered & (n_d.norm(dim=-1) > 0.5) & (n_img.norm(dim=-1) > 0.5))
        # INTEGRITY: the border can never be "affected", because normals_from_depth emits
        # exactly zero there so those pixels are not in-loss. If this ever fires, the
        # affected fraction is being inflated by a border artefact.
        border = torch.zeros_like(in_loss)
        border[-1, :] = True
        border[:, -1] = True
        if bool((in_loss & border).any()):
            raise RuntimeError("in-loss pixel on the border: normals_from_depth changed")
        n_in = int(in_loss.sum())
        rec = {
            "name": v.name, "H": int(alpha.shape[0]), "W": int(alpha.shape[1]),
            "has_mask": keep is not None,
            "has_depth_prior": gt_d is not None, "has_normal_prior": gt_n is not None,
            "coverage_frac": float(covered.float().mean()),
            "keep_drop_frac": float((~keep_ok).float().mean()),
            "alpha_uncovered_frac": float((~alpha_ok).float().mean()),
        }
        rec.update(affected_stats(in_loss, covered, alpha_ok, keep_ok))

    # --- (b) the two losses, and (c) the two gradient fields
    t_u, _, _ = _terms(n_sum, z_img, alpha, keep, K, gt_d, gt_n, gate=False)
    t_g, _, _ = _terms(n_sum, z_img, alpha, keep, K, gt_d, gt_n, gate=True)
    rec["loss_dn_ungated"] = float(t_u["depth_normal"].detach())
    rec["loss_dn_gated"] = float(t_g["depth_normal"].detach())
    d = rec["loss_dn_gated"] - rec["loss_dn_ungated"]
    rec["loss_dn_abs_delta"] = abs(d)
    rec["loss_dn_rel_delta"] = abs(d) / max(abs(rec["loss_dn_ungated"]), 1e-30)
    rec["in_loss_px_gated"] = n_in - rec["affected_px"]

    gz_u, gn_u = _grads(n_sum, z_img, t_u["depth_normal"])
    gz_g, gn_g = _grads(n_sum, z_img, t_g["depth_normal"])
    rec["dn_only"] = {}
    for lab, a_, b_ in (("dL_dz_img", gz_g, gz_u), ("dL_dn_sum", gn_g, gn_u)):
        r, r2, c = _rel_cos(a_, b_)
        rec["dn_only"][lab] = {"rel": r, "rel_l2": r2, "cos": c}

    # The weighted geometry loss the trainer actually backpropagates: a dn-only rel is
    # the term's own change, which is NOT what the splats feel at dn weight 0.05.
    def _tot(t):
        s = 0.0
        for k, w in weights.items():
            if k in t:
                s = s + w * t[k]
        return s
    gz_u2, gn_u2 = _grads(n_sum, z_img, _tot(t_u))
    gz_g2, gn_g2 = _grads(n_sum, z_img, _tot(t_g))
    rec["weighted_total"] = {}
    for lab, a_, b_ in (("dL_dz_img", gz_g2, gz_u2), ("dL_dn_sum", gn_g2, gn_u2)):
        r, r2, c = _rel_cos(a_, b_)
        rec["weighted_total"][lab] = {"rel": r, "rel_l2": r2, "cos": c}
    return rec


# --------------------------------------------------------------------------- the driver

def _sha256_head(path: str, n: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:16]


def _git(*a):
    try:
        return subprocess.run(("git",) + a, cwd=Path(__file__).resolve().parent.parent,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return None


def run(args) -> dict:
    from metal_gauss.dataset import load_scene
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scene = load_scene(args.colmap, args.images, max_resolution=args.max_resolution,
                       eval_split_every=args.eval_split_every, masks_dir=args.masks,
                       depth_dir=args.depth_dir, normal_dir=args.normal_dir)
    views = scene.train if args.split == "train" else scene.heldout
    if not views:
        raise RuntimeError(f"no views in split {args.split}")
    step = max(1, len(views) // args.views)
    sel = views[::step][:args.views]

    if args.ply:
        p, active = params_from_ply(args.ply, device)
        model = {"kind": "ply", "path": args.ply, "sha256_head": _sha256_head(args.ply),
                 "splats": active}
    else:
        p, active = params_at_step0(scene, args.budget, args.seed, device)
        model = {"kind": "step0_seed", "budget": args.budget, "seed": args.seed,
                 "active": active, "sparse_points": int(len(scene.points))}

    weights = {"depth": args.depth_loss_weight, "normal": args.normal_loss_weight,
               "depth_normal": args.depth_normal_weight}
    per_view = [measure_view(p, v, active, args.sh_degree, device, weights) for v in sel]

    tot_in = sum(r["in_loss_px"] for r in per_view)
    if tot_in == 0:
        raise RuntimeError("no in-loss dn pixels in any view: nothing was measured")
    agg = {
        "views": len(per_view),
        "in_loss_px_total": tot_in,
        "affected_px_total": sum(r["affected_px"] for r in per_view),
        "affected_frac_overall": sum(r["affected_px"] for r in per_view) / tot_in,
        "affected_by_alpha_frac_overall":
            sum(r["affected_by_alpha_px"] for r in per_view) / tot_in,
        "affected_by_keep_frac_overall":
            sum(r["affected_by_keep_px"] for r in per_view) / tot_in,
        "affected_by_both_frac_overall":
            sum(r["affected_by_both_px"] for r in per_view) / tot_in,
        "affected_frac_min": min(r["affected_frac"] for r in per_view),
        "affected_frac_median": float(np.median([r["affected_frac"] for r in per_view])),
        "affected_frac_max": max(r["affected_frac"] for r in per_view),
        "loss_dn_rel_delta_median":
            float(np.median([r["loss_dn_rel_delta"] for r in per_view])),
        "loss_dn_rel_delta_max": max(r["loss_dn_rel_delta"] for r in per_view),
    }
    for grp in ("dn_only", "weighted_total"):
        for lab in ("dL_dz_img", "dL_dn_sum"):
            agg[f"{grp}.{lab}.rel_max"] = max(r[grp][lab]["rel"] for r in per_view)
            agg[f"{grp}.{lab}.rel_median"] = float(
                np.median([r[grp][lab]["rel"] for r in per_view]))
            agg[f"{grp}.{lab}.rel_l2_max"] = max(r[grp][lab]["rel_l2"] for r in per_view)
            agg[f"{grp}.{lab}.rel_l2_median"] = float(
                np.median([r[grp][lab]["rel_l2"] for r in per_view]))
            agg[f"{grp}.{lab}.cos_min"] = min(r[grp][lab]["cos"] for r in per_view)
    agg["affected_frac_line"] = AFFECTED_FRACTION_LINE
    agg["affected_frac_under_line"] = bool(
        agg["affected_frac_overall"] < AFFECTED_FRACTION_LINE)
    agg["grad_rel_max_over_all"] = max(
        agg[f"{g}.{l}.rel_max"] for g in ("dn_only", "weighted_total")
        for l in ("dL_dz_img", "dL_dn_sum"))
    agg["grad_rel_l2_max_over_all"] = max(
        agg[f"{g}.{l}.rel_l2_max"] for g in ("dn_only", "weighted_total")
        for l in ("dL_dz_img", "dL_dn_sum"))
    agg["grad_cos_min_over_all"] = min(
        agg[f"{g}.{l}.cos_min"] for g in ("dn_only", "weighted_total")
        for l in ("dL_dz_img", "dL_dn_sum"))
    agg["f32_kernel_distance_from_truth"] = F32_KERNEL_DISTANCE_FROM_TRUTH
    agg["grad_under_kernel_f32_error"] = bool(
        agg["grad_rel_max_over_all"] < F32_KERNEL_DISTANCE_FROM_TRUTH)

    return {
        "kind": KIND, "schema": SCHEMA, "synthetic": bool(args.synthetic),
        "scene": args.scene, "step_label": args.step_label,
        "dataset": {"colmap": args.colmap, "images": args.images, "masks": args.masks,
                    "depth_dir": args.depth_dir, "normal_dir": args.normal_dir,
                    "max_resolution": args.max_resolution,
                    "eval_split_every": args.eval_split_every, "split": args.split,
                    "n_train": len(scene.train), "n_heldout": len(scene.heldout)},
        "model": model, "weights": weights,
        "git": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "aggregate": agg, "per_view": per_view,
    }


def summarise(paths: list[str], scenes: list[str]) -> dict:
    """Cross-scene decision from the measurement files ALONE, and only from the ones named.

    Refuses a synthetic file, a file of the wrong kind, a duplicate scene/step, and any
    named scene that is missing -- a summary that silently averages what it happens to
    find is how a smoke fixture becomes a result.
    """
    got: dict[tuple[str, str], dict] = {}
    for pth in paths:
        d = json.loads(Path(pth).read_text())
        if d.get("kind") != KIND:
            raise RuntimeError(f"{pth}: kind {d.get('kind')!r}, not {KIND}")
        if d.get("synthetic"):
            raise RuntimeError(f"{pth}: synthetic:true -- a fixture is not a measurement")
        key = (d["scene"], d["step_label"])
        if key in got:
            raise RuntimeError(f"duplicate {key} ({pth})")
        got[key] = d
    missing = [s for s in scenes if not any(k[0] == s for k in got)]
    if missing:
        raise RuntimeError(f"named scenes with no measurement: {missing}")
    extra = sorted({k[0] for k in got} - set(scenes))
    if extra:
        raise RuntimeError(f"measurements for scenes not named: {extra}")
    rows = [{"scene": k[0], "step": k[1],
             "affected_frac_overall": d["aggregate"]["affected_frac_overall"],
             "by_alpha": d["aggregate"]["affected_by_alpha_frac_overall"],
             "by_keep": d["aggregate"]["affected_by_keep_frac_overall"],
             "affected_frac_max_view": d["aggregate"]["affected_frac_max"],
             "loss_dn_rel_delta_median": d["aggregate"]["loss_dn_rel_delta_median"],
             "grad_rel_max": d["aggregate"]["grad_rel_max_over_all"],
             "under_line": d["aggregate"]["affected_frac_under_line"],
             "grad_under_f32": d["aggregate"]["grad_under_kernel_f32_error"]}
            for k, d in sorted(got.items())]
    worst = max(r["affected_frac_overall"] for r in rows)
    all_under = all(r["under_line"] for r in rows)
    return {
        "kind": KIND + "_summary", "scenes": sorted(scenes), "rows": rows,
        "worst_affected_frac_overall": worst,
        "line": AFFECTED_FRACTION_LINE,
        "verdict": "CLOSE" if all_under else "ESCALATE",
        "falsifier_refuted": bool(not all_under),
        "falsifier_supported": bool(all_under and all(r["grad_under_f32"] for r in rows)),
        "grad_rel_max_over_all_scenes": max(r["grad_rel_max"] for r in rows),
    }


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colmap"); ap.add_argument("--images")
    ap.add_argument("--masks", default=None)
    ap.add_argument("--depth-dir", default=None); ap.add_argument("--normal-dir", default=None)
    ap.add_argument("--max-resolution", type=int, default=1920)
    ap.add_argument("--eval-split-every", type=int, default=8)
    ap.add_argument("--split", choices=["train", "heldout"], default="train",
                    help="the dn term is computed on TRAINING views; that is the default")
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--ply", default=None, help="a trained export; omit for the step-0 seed")
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--depth-loss-weight", type=float, default=1.0)
    ap.add_argument("--normal-loss-weight", type=float, default=0.2)
    ap.add_argument("--depth-normal-weight", type=float, default=0.05)
    ap.add_argument("--scene", default=None, help="scene label, required to write output")
    ap.add_argument("--step-label", default=None, help="e.g. step0 or 30k")
    ap.add_argument("--synthetic", action="store_true",
                    help="mark the output as a fixture; --summary then REFUSES it")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", nargs="+", default=None, metavar="FILE")
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scene labels the summary must cover, exactly")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.summary:
        if not args.scenes:
            raise SystemExit("--summary requires --scenes: a summary must NAME its scenes")
        out = summarise(args.summary, args.scenes.split(","))
    else:
        for req in ("colmap", "images", "scene", "step_label"):
            if getattr(args, req) is None:
                raise SystemExit(f"--{req.replace('_', '-')} is required")
        out = run(args)
    print(json.dumps(out["aggregate"] if "aggregate" in out else out, indent=2,
                     sort_keys=True))
    if args.out:
        # Written LAST and only on success: a result file that exists is a result file
        # that was computed. Never truncate a previous arm's output on the way in.
        Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"wrote {args.out}", file=sys.stderr)
    return out


if __name__ == "__main__":
    main()
