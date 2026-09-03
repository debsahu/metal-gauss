"""Task 11: where does Tier 1's time actually go? Measured from scratch.

Three earlier number-sets exist (7.04x/2.91x, 4.80x/2.22x, 2.57x/2.60x) and ALL THREE are
superseded, not reconciled: every one was taken at the default --num-downscales 2 (a mean
0.4375 pixel fraction, so none of them is a resolution you can name), and every one
predates the aux weight-path fix, which changed what the aux backward computes.

Blocks, per the plan's Task 11 step 1:
  a  fused RGB fwd+bwd alone
  b  a + the O(N) torch aux VALUE construction (no aux passes)
  c  a + aux passes FORWARD only
  d  full: RGB + aux passes, fwd+bwd
  e  the loss arithmetic on the maps (normals_from_depth + the three losses), fwd+bwd

Median of 15 after 5 warm-ups, torch.mps.synchronize() around each.
"""
import json, sys, time
import torch
from metal_gauss import render
from metal_gauss.geometry_loss import (depth_loss, depth_normal_loss, normal_loss,
                                       normals_from_depth, splat_normals_cam)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
CASES = [("full 1920x1440", 1920, 1440), ("half 960x720", 960, 720), ("quarter 480x360", 480, 360)]
REPS, WARM = 15, 5


def leaves(n, dev="mps"):
    torch.manual_seed(0)
    L = dict(m=torch.randn(n, 3) * 0.6 + torch.tensor([0., 0., 4.]),
             q=torch.randn(n, 4), s=torch.rand(n, 3) * 0.03 + 0.005,
             o=torch.rand(n) * 0.7 + 0.15, sh=torch.randn(n, 16, 3) * 0.25)
    return {k: v.to(dev).requires_grad_(True) for k, v in L.items()}


def timeit(fn):
    for _ in range(WARM):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(REPS):
        torch.mps.synchronize(); t = time.perf_counter()
        fn(); torch.mps.synchronize()
        ts.append((time.perf_counter() - t) * 1000.0)
    return sorted(ts)[len(ts) // 2]


def run(W, H):
    L = leaves(N)
    K = torch.eye(3); K[0, 0] = K[1, 1] = 0.8 * max(W, H); K[0, 2], K[1, 2] = W / 2, H / 2
    vm = torch.eye(4); vmd = vm.to("mps")
    gt_d = torch.rand(H, W, device="mps") * 3 + 1.0
    gt_n = torch.zeros(H, W, 3, device="mps"); gt_n[..., 2] = -1.0

    def zero():
        for v in L.values():
            v.grad = None

    def aux_vals():
        return [splat_normals_cam(L["m"], L["q"], L["s"], vmd),
                (L["m"] @ vmd[:3, :3].T + vmd[:3, 3])[:, 2:3].expand(-1, 3)]

    def rgb_only():
        zero()
        rgb, _, _ = render(L["m"], L["q"], L["s"], L["o"], L["sh"][:, :1].contiguous(),
                           K, vm, W, H, sh_degree=3, sh_rest=L["sh"][:, 1:].contiguous(),
                           backend="metal")
        rgb.square().mean().backward()

    def rgb_plus_values():
        zero()
        a = aux_vals()
        rgb, _, _ = render(L["m"], L["q"], L["s"], L["o"], L["sh"][:, :1].contiguous(),
                           K, vm, W, H, sh_degree=3, sh_rest=L["sh"][:, 1:].contiguous(),
                           backend="metal")
        (rgb.square().mean() + a[0].sum() * 0 + a[1].sum() * 0).backward()

    def aux_fwd_only():
        zero()
        with torch.no_grad():
            a = [x.detach() for x in aux_vals()]
            render(L["m"].detach(), L["q"].detach(), L["s"].detach(), L["o"].detach(),
                   L["sh"][:, :1].detach().contiguous(), K, vm, W, H, sh_degree=3,
                   sh_rest=L["sh"][:, 1:].detach().contiguous(), backend="metal",
                   aux_colors=a, aux_detach_weights=[True, True])
        rgb_only()

    def full():
        zero()
        rgb, alpha, info = render(L["m"], L["q"], L["s"], L["o"], L["sh"][:, :1].contiguous(),
                                  K, vm, W, H, sh_degree=3,
                                  sh_rest=L["sh"][:, 1:].contiguous(), backend="metal",
                                  aux_colors=aux_vals(), aux_detach_weights=[True, True])
        (rgb.square().mean() + info["aux"][0].square().mean()
         + info["aux"][1].square().mean()).backward()

    # (e) the loss chain alone, on maps detached from the render so only the loss is timed
    rgb0, alpha0, info0 = render(L["m"], L["q"], L["s"], L["o"], L["sh"][:, :1].contiguous(),
                                 K, vm, W, H, sh_degree=3,
                                 sh_rest=L["sh"][:, 1:].contiguous(), backend="metal",
                                 aux_colors=aux_vals(), aux_detach_weights=[True, True])
    n_map = info0["aux"][0].detach().requires_grad_(True)
    z_map = info0["aux"][1].detach().requires_grad_(True)
    a_det = alpha0.detach()

    def loss_chain():
        if n_map.grad is not None: n_map.grad = None
        if z_map.grad is not None: z_map.grad = None
        ni = n_map / a_det.clamp_min(1e-10)[..., None]
        ni = ni / ni.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        di = z_map[..., 0] / a_det.clamp_min(1e-10)
        nd = normals_from_depth(di, K[0, 0].item(), K[1, 1].item(), K[0, 2].item(), K[1, 2].item())
        (depth_loss(di, gt_d) + 0.2 * normal_loss(ni, gt_n)
         + 0.05 * depth_normal_loss(nd, ni, a_det)).backward()

    return {"a_rgb_fwd_bwd": timeit(rgb_only),
            "b_plus_aux_values": timeit(rgb_plus_values),
            "c_plus_aux_fwd": timeit(aux_fwd_only),
            "d_full_fwd_bwd": timeit(full),
            "e_loss_chain": timeit(loss_chain)}


out = {"splats": N, "reps": REPS, "warmups": WARM,
       "note": "single resolution per row; NOT a --num-downscales schedule average",
       "cases": {}}
for name, W, H in CASES:
    r = run(W, H)
    aux_cost = r["d_full_fwd_bwd"] - r["a_rgb_fwd_bwd"] - (r["b_plus_aux_values"] - r["a_rgb_fwd_bwd"])
    r["aux_pass_cost"] = aux_cost
    r["value_cost"] = r["b_plus_aux_values"] - r["a_rgb_fwd_bwd"]
    r["ratio_d_over_a"] = r["d_full_fwd_bwd"] / r["a_rgb_fwd_bwd"]
    # Target A (plan): recover >= 75% of the aux-pass cost.
    r["target_A_ms"] = (r["a_rgb_fwd_bwd"] + r["value_cost"] + r["e_loss_chain"]
                        + 0.25 * (r["d_full_fwd_bwd"] - r["a_rgb_fwd_bwd"]
                                  - r["value_cost"] - r["e_loss_chain"]))
    out["cases"][name] = r
    print(f"\n=== {name}, {N:,} splats ===")
    for k in ("a_rgb_fwd_bwd", "b_plus_aux_values", "c_plus_aux_fwd", "d_full_fwd_bwd",
              "e_loss_chain", "value_cost", "aux_pass_cost", "target_A_ms"):
        print(f"  {k:<20} {r[k]:8.2f} ms")
    print(f"  {'d/a ratio':<20} {r['ratio_d_over_a']:8.2f}x")
    print(f"  loss chain vs aux passes: {r['e_loss_chain']:.2f} vs {r['aux_pass_cost']:.2f} ms"
          f"  -> {'LOSS DOMINATES (do the loss kernel first)' if r['e_loss_chain'] > r['aux_pass_cost'] else 'aux passes dominate (fuse them first)'}")
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/task11.json", "w"), indent=1)
