#!/usr/bin/env python3
"""Task 21 step 1: rescore every arm's --eval-dump per view, under both nets.

    bench/lpips_rescore.py score  --dump DIR --scene S --arm A --out-root R [--nets vgg,alex]
    bench/lpips_rescore.py summary --out-root R --scenes s1,s2 [--require s/a,s/a,...]

WHY NOT scripts/lpips_eval.py. That tool writes `lpips.json` INTO the dump,
unconditionally. Task 19's only defect that reached a number was a re-grade
silently overwriting a different arm's verdict, both files well-formed
(research/metal-gauss.md 12.5), so nothing in this task rewrites an arm's
published result. The rescoring lands in its own tree and refuses to overwrite.

THE SELF-CHECK IS THE POINT OF `score`. The recomputed VGG mean must reproduce
the arm's committed lpips.json. If it does not, this scoring chain is not the
one that produced research/metal-gauss.md 8.3's numbers and every comparison
downstream of it is void. It is reported, not assumed, and `--summary` refuses a
result whose self-check did not run.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
from PIL import Image                                                 # noqa: E402

from bench import lpips_attr as LA                                    # noqa: E402

SELF_CHECK_TOL = 1e-6


def load_rgb(p: Path) -> torch.Tensor:
    """[H, W, 3] float32 in [0, 1] -- the PNG as it sits on disk."""
    return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0)


def to_lpips(img: torch.Tensor) -> torch.Tensor:
    """scripts/lpips_eval.py's convention: [-1, 1], NCHW."""
    return (img * 2.0 - 1.0).permute(2, 0, 1)[None]


def make_metric(net: str):
    import lpips
    fn = lpips.LPIPS(net=net, verbose=False).eval()

    def m(a: torch.Tensor, b: torch.Tensor) -> float:
        with torch.no_grad():
            return float(fn(to_lpips(a), to_lpips(b)))
    return m


def pairs(dump: Path) -> list[tuple[str, Path, Path, Path | None]]:
    out = []
    for r in sorted(dump.glob("*_render.png")):
        stem = r.name[: -len("_render.png")]
        g = r.with_name(stem + "_gt.png")
        if not g.exists():
            raise FileNotFoundError(f"{dump}: no ground truth for {r.name}")
        m = r.with_name(stem + "_mask.png")
        out.append((stem, r, g, m if m.exists() else None))
    if not out:
        raise FileNotFoundError(f"{dump}: no *_render.png")
    return out


def dist(vals: list[float]) -> dict:
    v = sorted(vals)
    n = len(v)

    def q(f):
        return v[min(n - 1, max(0, int(round(f * (n - 1)))))]
    mean = statistics.fmean(v)
    return {"n": n, "mean": mean, "std": statistics.pstdev(v) if n > 1 else 0.0,
            "min": v[0], "p10": q(0.10), "p25": q(0.25), "p50": q(0.50),
            "p75": q(0.75), "p90": q(0.90), "max": v[-1],
            "spread": v[-1] - v[0],
            "frac_above_mean_plus_0p1": sum(1 for x in v if x > mean + 0.1) / n}


def composite(render: torch.Tensor, gt: torch.Tensor, mask_path) -> torch.Tensor:
    """Replace the DROPPED region with ground truth, so it contributes nothing.

    `View.mask` is uint8 with 255 = KEEP, so `m01 = mask/255` is 1 where the
    pixel is trained on. The shipped LPIPS convention is full-frame with those
    regions rendered BLACK against real GT content, which on a masked capture
    charges the metric for pixels the trainer was told to ignore. The difference
    between the two numbers is the size of that convention effect.

    CAVEAT, stated because it bounds what the number means: LPIPS is not
    pixel-local, so pasting GT in leaves a seam at the mask boundary that VGG
    still sees. This is an estimate of the convention's contribution, not an
    exact removal of it.
    """
    m = load_rgb(mask_path)[..., :1]
    return render * m + gt * (1.0 - m)


def score(args) -> None:
    dump = Path(args.dump)
    ps = pairs(dump)
    nets = args.nets.split(",")
    mode = getattr(args, "mask_mode", "full")
    if mode == "composite" and not any(m is not None for _, _, _, m in ps):
        raise ValueError(f"{dump}: --mask-mode composite but the dump has no *_mask.png")
    per_view: dict[str, dict[str, float]] = {n: {} for n in nets}
    black = {}
    for net in nets:
        metric = make_metric(net)
        for stem, rp, gp, mp in ps:
            r, g = load_rgb(rp), load_rgb(gp)
            if r.shape != g.shape:
                raise ValueError(f"{stem}: render {tuple(r.shape)} vs gt {tuple(g.shape)}")
            if mode == "composite":
                r = composite(r, g, mp)
            per_view[net][stem] = metric(r, g)
            if net == nets[0]:
                black[stem] = float((r.sum(-1) == 0).float().mean())

    committed = dump / "lpips.json"
    self_check = {"file": str(committed), "present": committed.exists()}
    if committed.exists() and "vgg" in per_view and mode == "full":
        c = json.loads(committed.read_text())
        got = statistics.fmean(per_view["vgg"].values())
        self_check.update(committed_net=c.get("net"), committed_mean=c.get("mean"),
                          recomputed_mean=got, abs_diff=abs(got - float(c["mean"])),
                          tol=SELF_CHECK_TOL,
                          passed=bool(c.get("net") == "vgg"
                                      and abs(got - float(c["mean"])) <= SELF_CHECK_TOL))
    else:
        self_check["passed"] = None

    out = LA.write_json(
        Path(args.out_root) / "step1" /
        f"{args.scene}__{args.arm}{'' if mode == 'full' else '__' + mode}.json",
        {"stage": "step1_rescore", "scene": args.scene, "arm": args.arm,
         "mask_mode": mode,
         "dump": str(dump), "n_views": len(ps),
         "has_mask_files": any(m is not None for _, _, _, m in ps),
         "image_hw": list(load_rgb(ps[0][1]).shape[:2]),
         "self_check": self_check,
         "per_view": per_view,
         "dist": {n: dist(list(per_view[n].values())) for n in nets},
         "black_render_pixel_frac": {"per_view": black,
                                     "mean": statistics.fmean(black.values())}})
    for n in nets:
        d = out and dist(list(per_view[n].values()))
        print(f"{args.scene}/{args.arm} {n}: mean {d['mean']:.4f} "
              f"p50 {d['p50']:.4f} min {d['min']:.4f} max {d['max']:.4f} n={d['n']}")
    print(f"  self-check vs committed lpips.json: {self_check.get('passed')}"
          f" (|d| {self_check.get('abs_diff')})")
    print(f"  wrote {out}")


def summary(args) -> None:
    root = Path(args.out_root) / "step1"
    files = sorted(root.glob("*.json"))
    seen: dict[tuple[str, str], Path] = {}
    rows = []
    for f in files:
        d = LA.read_result(f)                     # refuses a foreign kind/schema
        if d.get("stage") != "step1_rescore":
            raise ValueError(f"{f}: stage {d.get('stage')!r} is not step1_rescore")
        key = (d["scene"], d["arm"], d.get("mask_mode", "full"))
        if key in seen:
            raise ValueError(f"duplicate (scene, arm) {key}: {seen[key]} and {f}")
        seen[key] = f
        if d["self_check"].get("passed") is False:
            raise ValueError(f"{f}: self-check FAILED -- the scoring chain does not "
                             f"reproduce the committed lpips.json; no summary is valid")
        rows.append(d)
    if args.require:
        want = {(a, b, "full") for a, b in
                (x.split("/", 1) for x in args.require.split(","))}
        missing = want - set(seen)
        if missing:
            raise ValueError(f"missing required (scene, arm): {sorted(missing)}")
    if args.scenes:
        keep = set(args.scenes.split(","))
        rows = [r for r in rows if r["scene"] in keep]
        if not rows:
            raise ValueError(f"no results for scenes {sorted(keep)}")
    hdr = (f"{'scene':<9} {'arm':<18} {'net':<5} {'n':>4} {'mean':>8} {'p50':>8} "
           f"{'min':>8} {'max':>8} {'spread':>8} {'>m+.1':>6} {'blk%':>6}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["scene"], x["arm"])):
        for net, d in r["dist"].items():
            print(f"{r['scene']:<9} {r['arm']:<18} {net:<5} {d['n']:>4} "
                  f"{d['mean']:>8.4f} {d['p50']:>8.4f} {d['min']:>8.4f} {d['max']:>8.4f} "
                  f"{d['spread']:>8.4f} {d['frac_above_mean_plus_0p1']:>6.2f} "
                  f"{100 * r['black_render_pixel_frac']['mean']:>6.3f}")
    print(f"\n{len(rows)} arm(s); every self-check present and passing.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--dump", required=True)
    s.add_argument("--scene", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--out-root", required=True)
    s.add_argument("--nets", default="vgg,alex")
    s.add_argument("--mask-mode", choices=["full", "composite"], default="full",
                   help="full = the shipped convention (masked pixels render black "
                        "against GT content); composite = dropped region replaced by GT")
    s.set_defaults(fn=score)
    m = sub.add_parser("summary")
    m.add_argument("--out-root", required=True)
    m.add_argument("--scenes", default="")
    m.add_argument("--require", default="")
    m.set_defaults(fn=summary)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
