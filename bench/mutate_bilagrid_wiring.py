"""Task 22 wiring mutation battery. Same rules: prove the mutant changes
behaviour, then assert on the FAILING TEST'S NAME."""
import json, re, subprocess, sys, pathlib
WT = pathlib.Path(sys.argv[1]); PY = str(WT / ".venv/bin/python")
FILES = {f: (WT / f).read_text() for f in ("metal_gauss/train.py", "metal_gauss/appearance.py")}
T = "tests/test_appearance_bilagrid_wiring.py"

MUT = [
 # 1. appearance never reaches the photometric loss (the classic dead flag)
 ("loss_uses_raw_render", "metal_gauss/train.py",
  "loss, terms = photometric_loss(rgb_c, gt, m01, kernel, return_terms=True)",
  "loss, terms = photometric_loss(rgb, gt, m01, kernel, return_terms=True)",
  ["test_bilagrid_reaches_the_photometric_loss_and_NOT_the_aux_maps"]),
 # 2. the corrected render leaks into the aux/geometry path
 ("aux_sees_the_corrected_render", "metal_gauss/train.py",
  'g = geometry_terms(args, info["aux"], alpha, v.K, gt_d, gt_n, keep)',
  'g = geometry_terms(args, [rgb_c - rgb + a for a in info["aux"]], alpha, v.K, gt_d, gt_n, keep)',
  ["test_bilagrid_reaches_the_photometric_loss_and_NOT_the_aux_maps"]),
 # 3. THE ONE THAT MATTERS MOST: appearance applied at eval, Brush's apply_eval
 ("appearance_applied_at_eval", "metal_gauss/train.py",
  '''                          dump_dir=getattr(args, "eval_dump", None))''',
  '''                          dump_dir=getattr(args, "eval_dump", None),
                          _appearance=appearance)''',
  ["test_heldout_eval_never_applies_the_appearance_model"]),
 # 4. TV applied at --appearance-reg (1e-2) instead of Brush's 10.0
 ("tv_at_appearance_reg", "metal_gauss/appearance.py",
  'self.reg_weight = float(tv_weight if mode == "bilagrid" else reg_weight)',
  'self.reg_weight = float(reg_weight)',
  ["test_bilagrid_is_regularised_by_the_TV_WEIGHT_and_not_by_appearance_reg"]),
 # 5. the lr schedule is computed and never written to the group
 ("lr_schedule_dropped", "metal_gauss/train.py",
  '                    grp["lr"] = appearance_lr_at(step - 1, args.bilagrid_lr, args.steps)',
  '                    _ = appearance_lr_at(step - 1, args.bilagrid_lr, args.steps)',
  ["test_the_appearance_group_gets_the_brush_lr_schedule_and_nothing_else_does"]),
 # 6. the schedule catches every param group, not just appearance
 ("lr_schedule_hits_every_group", "metal_gauss/train.py",
  '                if grp.get("name") == "appearance":',
  '                if True:',
  ["test_the_appearance_group_gets_the_brush_lr_schedule_and_nothing_else_does"]),
 # 7. the report describes the configuration instead of the learned state
 ("report_state_is_static", "metal_gauss/appearance.py",
  '"max_abs_dev": float((g - ident).abs().max()),',
  '"max_abs_dev": 0.0,',
  ["test_report_records_what_the_grid_actually_did"]),
 # 8. dims read in the wrong order (guidance, y, x instead of x, y, guidance)
 ("dims_order_reversed", "metal_gauss/train.py",
  'dims=tuple(getattr(args, "bilagrid_dims", (16, 16, 8))))',
  'dims=tuple(reversed(getattr(args, "bilagrid_dims", (16, 16, 8)))))',
  ["test_report_records_what_the_grid_actually_did"]),
]

def patch_eval_signature(txt):
    """Mutant 3 needs evaluate() to accept and apply the model."""
    txt = txt.replace(
        "             antialias: bool = False, filter_3d=None, dump_dir=None) -> dict:",
        "             antialias: bool = False, filter_3d=None, dump_dir=None,\n"
        "             _appearance=None) -> dict:")
    return txt.replace(
        "        gt = v.image.to(device).float() / 255.0\n"
        "        m01 = None if v.mask is None else v.mask.to(device).float() / 255.0\n"
        "        err2 = (rgb.clamp(0, 1) - gt) ** 2",
        "        if _appearance is not None:\n"
        "            rgb = _appearance(rgb, 0)\n"
        "        gt = v.image.to(device).float() / 255.0\n"
        "        m01 = None if v.mask is None else v.mask.to(device).float() / 255.0\n"
        "        err2 = (rgb.clamp(0, 1) - gt) ** 2")

def run():
    r = subprocess.run([PY, "-m", "pytest", T, "-q", "--tb=no", "-p", "no:cacheprovider"],
                       cwd=WT, capture_output=True, text=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    return set(re.findall(r"FAILED " + re.escape(T) + r"::(\w+)", r.stdout)), r.stdout

_, base_out = run()
assert "failed" not in base_out.split("=")[-1], f"suite not green before mutating: {base_out[-400:]}"
res = []
for name, f, old, new, expect in MUT:
    src = FILES[f]
    assert src.count(old) == 1, f"{name}: anchor count {src.count(old)}"
    mutated = src.replace(old, new)
    if name == "appearance_applied_at_eval":
        mutated = patch_eval_signature(mutated)
    (WT / f).write_text(mutated)
    failed, out = run()
    changed = failed != set() or "passed" not in out
    ok = (set(expect) <= failed) if expect else (failed == set())
    res.append((name, sorted(failed), ok, bool(expect)))
    print(f"{'KILL ' if ok else 'MISS '} {name:32s} failed={sorted(failed)}")
    (WT / f).write_text(src)
for f, src in FILES.items():
    assert (WT / f).read_text() == src, f"{f} not restored"
print(f"\n{sum(r[2] for r in res)}/{len(res)} resolved")
json.dump([{"mutant": n, "failed_tests": ft, "resolved": ok, "expected_kill": e}
           for n, ft, ok, e in res], open(sys.argv[2], "w"), indent=1)
