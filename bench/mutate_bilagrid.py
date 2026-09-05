"""Task 22 mutation battery for metal_gauss/bilagrid.py.

For each mutant: (1) apply it, (2) PROVE it changes behaviour -- a mutant that is
behaviourally identical proves nothing, and this project has recorded a "killed"
mutant that had in fact survived -- (3) run the suite, (4) assert the EXPECTED
TEST NAMES are among the failures. Never a failure COUNT: a count is satisfied by
any failure, including an import error the mutant caused by accident.
"""
import json, re, subprocess, sys, pathlib

WT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
SRC = WT / "metal_gauss" / "bilagrid.py"
PY = str(WT / ".venv/bin/python")
ORIG = SRC.read_text()

# (name, old, new, tests that MUST fail)
MUTANTS = [
 ("identity_column_major",
  "IDENTITY_CHANNELS = (0, 5, 10)", "IDENTITY_CHANNELS = (0, 4, 8)",
  ["test_identity_grid_reproduces_the_render"]),
 ("align_corners_false",
  'padding_mode="border", align_corners=True', 'padding_mode="border", align_corners=False',
  ["test_slice_matches_the_independent_eight_corner_transcription"]),
 # NOTE padding_mode alone is UNREACHABLE while the explicit clamp stands: the
 # sample never leaves the volume, so border and zeros are behaviourally
 # identical. Recorded here as a null rather than deleted, because "this mutant
 # is not killable" is a fact about the code worth keeping.
 ("padding_zeros__EXPECTED_NULL",
  'mode="bilinear",\n                         padding_mode="border"',
  'mode="bilinear",\n                         padding_mode="zeros"', []),
 ("xy_swapped_in_sampler",
  "return torch.stack([xn[None, :].expand(h, w),\n                        yn[:, None].expand(h, w), zn], dim=-1)[None, None]",
  "return torch.stack([yn[:, None].expand(h, w),\n                        xn[None, :].expand(h, w), zn], dim=-1)[None, None]",
  ["test_slice_matches_the_independent_eight_corner_transcription"]),
 # Likewise: deleting the clamp alone is identical, because border saturates.
 ("guidance_clamp_removed__EXPECTED_NULL",
  "zn = (2.0 * lum - 1.0).clamp(-1.0, 1.0)", "zn = (2.0 * lum - 1.0)", []),
 # The REACHABLE failure is losing both guards at once.
 ("guidance_unclamped_and_padding_zeros",
  'zn = (2.0 * lum - 1.0).clamp(-1.0, 1.0)', 'zn = (2.0 * lum - 1.0)  # PAD:zeros',
  ["test_guidance_is_clamped_and_the_clamp_kills_the_gradient"]),
 ("luma_r_and_b_swapped",
  "LUMA_R, LUMA_G, LUMA_B = 0.299, 0.587, 0.114",
  "LUMA_R, LUMA_G, LUMA_B = 0.114, 0.587, 0.299",
  ["test_slice_matches_the_independent_eight_corner_transcription"]),
 ("tv_sum_of_sums",
  "return dx.mean() + dy.mean() + dz.mean()", "return dx.sum() + dy.sum() + dz.sum()",
  ["test_tv_is_a_sum_of_three_means_not_a_mean_of_the_concatenation"]),
 ("tv_dy_on_wrong_axis",
  "dy = (grid[..., 1:, :] - grid[..., :-1, :]) ** 2",
  "dy = (grid[..., 1:] - grid[..., :-1]) ** 2",
  ["test_tv_is_a_sum_of_three_means_not_a_mean_of_the_concatenation"]),
 ("regulariser_always_global",
  "g = self.grids if idx is None else self.grids[idx:idx + 1]", "g = self.grids",
  ["test_tv_defaults_to_the_active_view_only", "test_tv_gradient_reaches_only_the_active_view"]),
 ("forward_ignores_index",
  "return slice_apply(self.grids[idx:idx + 1], rgb)", "return slice_apply(self.grids[0:1], rgb)",
  ["test_forward_uses_the_indexed_view_and_no_other",
   "test_photometric_gradient_lands_only_in_the_drawn_view"]),
 ("render_detached_in_affine",
  "col = torch.cat([rgb.permute(2, 0, 1),", "col = torch.cat([rgb.detach().permute(2, 0, 1),",
  ["test_gradient_reaches_the_render_through_the_AFFINE_and_not_only_the_guidance"]),
 ("einsum_transposed",
  'torch.einsum("rchw,chw->hwr", coef, col)', 'torch.einsum("rchw,chw->hwr", coef.transpose(0, 1).reshape(3, 4, h, w), col)',
  ["test_slice_matches_the_independent_eight_corner_transcription"]),
 ("warmup_off_by_one",
  "t = (step + 1.0) / warmup", "t = step / warmup",
  ["test_warmup_exp_lr_matches_an_independent_transcription"]),
 ("dims_guard_removed",
  "if min(gw, gh, gl) < 2:", "if False:",
  ["test_dims_below_two_are_rejected"]),
]

PROBE = r'''
import json, sys, torch
sys.path.insert(0, %r)
from metal_gauss.bilagrid import identity_grids, slice_apply, tv_loss, warmup_exp_lr, BilateralGrid
gen = torch.Generator().manual_seed(11)
g = torch.randn(1, 12, 5, 3, 4, generator=gen, dtype=torch.float64)
rgb = torch.rand(7, 5, 3, generator=gen, dtype=torch.float64) * 3.0 - 1.0
out = {"slice": float(slice_apply(g, rgb).sum()),
       "tv": float(tv_loss(g)),
       "ident": float(identity_grids(2, (4, 3, 5), device="cpu", dtype=torch.float64).sum()),
       "lr": [warmup_exp_lr(s, 2e-3, decay_steps=30000) for s in (0, 500, 5000)]}
m = BilateralGrid(3, (4, 3, 5), device="cpu")
with torch.no_grad(): m.grids += torch.randn(m.grids.shape, generator=gen) * 0.1
out["fwd1"] = float(m(rgb.float(), 1).sum())
out["reg1"] = float(m.regulariser(1))
r = rgb.clone().requires_grad_(True)
slice_apply(g, r).sum().backward()
out["drgb"] = float(r.grad.abs().sum())
g2 = g.clone().requires_grad_(True)
slice_apply(g2, rgb).sum().backward()
out["dgrid"] = float(g2.grad.abs().sum())
try:
    identity_grids(1, (16, 16, 1), device="cpu"); out["guard"] = "no-raise"
except ValueError: out["guard"] = "raised"
print(json.dumps(out))
''' % str(WT)

def probe():
    r = subprocess.run([PY, "-c", PROBE], capture_output=True, text=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    if r.returncode != 0:
        return {"__error__": r.stderr.strip().splitlines()[-1][:200]}
    return json.loads(r.stdout)

def run_suite():
    r = subprocess.run([PY, "-m", "pytest", "tests/test_bilagrid.py", "-q", "--tb=no",
                        "-p", "no:cacheprovider"], cwd=WT, capture_output=True, text=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    return set(re.findall(r"FAILED tests/test_bilagrid\.py::(\w+)", r.stdout))

base = probe()
assert "__error__" not in base, base
print(f"pristine probe: {base}\n")
results = []
for name, old, new, expect in MUTANTS:
    assert ORIG.count(old) == 1, f"{name}: anchor appears {ORIG.count(old)} times, need exactly 1"
    mutated = ORIG.replace(old, new)
    if name == "guidance_unclamped_and_padding_zeros":
        mutated = mutated.replace('padding_mode="border"', 'padding_mode="zeros"')
    SRC.write_text(mutated)
    p = probe()
    changed = ("__error__" in p) or any(p.get(k) != base.get(k) for k in base)
    failed = run_suite()
    ok = (set(expect) <= failed and changed) if expect else (not changed)
    results.append((name, changed, sorted(failed), ok))
    print(f"{'KILL ' if ok else 'MISS '} {name:32s} behaviour-changed={changed}  "
          f"failed={sorted(failed)}")
    SRC.write_text(ORIG)
assert SRC.read_text() == ORIG
print(f"\n{sum(r[3] for r in results)}/{len(results)} mutants killed by the NAMED tests")
json.dump([{"mutant": n, "behaviour_changed": c, "failed_tests": f, "killed": k}
           for n, c, f, k in results], open(sys.argv[2], "w"), indent=1)
