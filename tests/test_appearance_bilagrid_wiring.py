"""Task 22 wiring: does `--appearance bilagrid` actually reach the loss, and does
it stay OUT of the two places it must never reach?

Unit tests on the grid cannot see any of this. This repo's failure log is a list
of flags that parse and do nothing -- LFS's `--train` is a no-op, its `--init=`
is dead, and a harness once forwarded `--budget` so `auto_budget()` never ran in
an eight-scene sweep. These tests go through `train()`.

Every assertion below reads a REPORT DICT, never stdout.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.test_train_recipe import _args, _synthetic_scene

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _non_identity_patch(monkeypatch, bias=0.25):
    """Force every view's grid off identity AT CONSTRUCTION, so step 1's forward
    already carries the correction. Without this the grid is the identity at init
    and an appearance model that was never called is INDISTINGUISHABLE from one
    that was -- which is the whole difficulty of testing this flag."""
    from metal_gauss import appearance as A
    real = A.AppearanceModel.__init__

    def patched(self, n_images, mode="gain_bias", device="mps", **kw):
        real(self, n_images, mode, device, **kw)
        with torch.no_grad():
            self.grid.grids[:, 3] += bias        # red bias in every cell
    monkeypatch.setattr(A.AppearanceModel, "__init__", patched)


@mps
def test_bilagrid_reaches_the_photometric_loss_and_NOT_the_aux_maps(monkeypatch):
    """One test, two halves, and each half is the other's control.

    With the grid forced OFF identity at construction, at step 1 -- the only step
    whose forward runs on the untouched seed -- the PHOTOMETRIC terms must differ
    from an `--appearance off` run (proving the model reached
    `photometric_loss`), while the GEOMETRY terms must be BIT-IDENTICAL (proving
    it did not reach the aux maps).

    Fails if: the flag parses and does nothing (photometric half); or the
    corrected render is handed to `geometry_terms` instead of the raw `info`
    (geometry half). Brush routes depth around appearance explicitly
    (brush-train/src/train.rs:1479-1491); here the aux maps are separate tensors
    and this test is what pins that it stays that way.

    Neither half alone is sufficient. A model that is never called passes the
    geometry half; a model wired into everything passes the photometric half.
    """
    from metal_gauss import train as T
    kw = dict(steps=1, eval_every=1, flatten_loss_weight=1.0, depth_loss_weight=1.0,
              normal_loss_weight=0.2, depth_normal_weight=0.05)
    off = T.train(_args(**kw), scene=_synthetic_scene())["log"][-1]["terms"]
    _non_identity_patch(monkeypatch)
    on = T.train(_args(appearance="bilagrid", **kw), scene=_synthetic_scene())["log"][-1]["terms"]

    for k in ("l1", "ssim"):
        assert on[k] != off[k], f"appearance never reached the photometric loss ({k} unmoved)"
    for k in ("depth", "normal", "depth_normal"):
        assert on[k] == off[k], (
            f"the appearance-corrected render leaked into the aux path: "
            f"{k} {off[k]!r} -> {on[k]!r}")
    # the geometry half is only meaningful if those terms are actually alive
    assert all(on[k] > 0 for k in ("depth", "normal", "depth_normal")), \
        "geometry terms are zero, so 'unchanged' proves nothing"


@mps
def test_heldout_eval_never_applies_the_appearance_model(monkeypatch, tmp_path):
    """The discipline the whole task rests on. Brush's `apply_eval`
    (brush-appearance/src/train_state.rs:422) DOES apply per-view parameters at
    eval; that is a per-view cheat here and is the easiest way to manufacture a
    spectacular, worthless held-out number.

    The appearance model is replaced by one whose forward returns PURE WHITE
    regardless of input, and the HELD-OUT DUMP is inspected directly: if
    `evaluate()` applied the model, every dumped render would be uniformly 255.

    THE FIRST VERSION OF THIS TEST COMPARED HELD-OUT PSNR AGAINST AN `--appearance
    off` RUN and asserted equality, on the reasoning that a white forward has no
    dependence on `rgb` and so leaves the gaussians untouched. That reasoning is
    WRONG, and the test failed on correct code: `torch.ones_like(rgb)` has no
    graph edge to `rgb` at all, so the render receives NO gradient rather than a
    zero one -- the two arms train differently by construction and their eval
    renders have no reason to agree. It failed by 0.025 dB, which is small enough
    to have been argued away as noise. Comparing the dump against the property it
    must have needs no second arm and cannot be confounded by a trajectory.
    """
    from PIL import Image
    from metal_gauss import appearance as A, train as T

    def white(self, rgb, idx):
        return torch.ones_like(rgb)
    monkeypatch.setattr(A.AppearanceModel, "forward", white)
    a = _args(appearance="bilagrid", steps=1, eval_every=1)
    a.eval_dump = str(tmp_path / "dump")
    on = T.train(a, scene=_synthetic_scene())["log"][-1]

    dumps = sorted((tmp_path / "dump").glob("*_render.png"))
    assert dumps, "no held-out render was dumped, so nothing was checked"
    for f in dumps:
        arr = np.asarray(Image.open(f)).astype(np.float32)
        assert arr.std() > 1.0, (
            f"{f.name} is a CONSTANT image -- the appearance model reached the "
            f"eval path (mean {arr.mean():.1f})")
        assert arr.mean() < 250.0, f"{f.name} is saturated white: {arr.mean():.1f}"
    # control: the white forward must really have been installed at TRAINING time,
    # or the eval assertion is satisfied by a model that was never called at all.
    off = T.train(_args(steps=1, eval_every=1), scene=_synthetic_scene())["log"][-1]
    assert on["terms"]["l1"] != off["terms"]["l1"], \
        "the white forward never ran, so the eval assertion proves nothing"


@mps
def test_bilagrid_is_regularised_by_the_TV_WEIGHT_and_not_by_appearance_reg():
    """Brush's TV weight is 10.0 (brush-train/src/config.rs:1010); this trainer's
    `--appearance-reg` default is 1e-2. Reusing the latter would apply the
    regulariser at 1/1000 of its ported strength -- and Task 21 measured TV to be
    LOAD-BEARING rather than a knob: the unregularised grid recovers a roughly
    scene-independent 7.4% of baseline on lego, i.e. nuisance capacity, against
    0.7% regularised.

    Fails silently in exactly the way that matters: nothing errors, the arm runs,
    and it is the nuisance model rather than the ported one.
    """
    from metal_gauss.train import build_parser
    a = build_parser().parse_args(["--colmap", "x", "--images", "y",
                                   "--appearance", "bilagrid"])
    assert a.bilagrid_tv_weight == 10.0
    assert a.bilagrid_lr == 2e-3
    assert a.bilagrid_dims == [16, 16, 8]
    from metal_gauss.appearance import AppearanceModel
    m = AppearanceModel(3, "bilagrid", device="cpu",
                        tv_weight=a.bilagrid_tv_weight, dims=a.bilagrid_dims)
    assert m.reg_weight == 10.0
    # and the OLD modes must be untouched by this change
    b = build_parser().parse_args(["--colmap", "x", "--images", "y",
                                   "--appearance", "affine"])
    assert AppearanceModel(3, "affine", device="cpu",
                           reg_weight=b.appearance_reg).reg_weight == 0.01


@mps
def test_report_records_what_the_grid_actually_did():
    """Every claim in the results section must trace to a report JSON, never to
    stdout. Fails if the appearance block is absent, or if it reports the
    configuration rather than the learned state -- `max_abs_dev` is the number
    that separates "the grid trained" from "the grid was constructed"."""
    from metal_gauss import train as T
    out = T.train(_args(appearance="bilagrid", steps=40, eval_every=40),
                  scene=_synthetic_scene())
    ap = out["metrics"]["appearance"]
    assert ap["mode"] == "bilagrid"
    assert ap["params"] == 5 * 12 * 8 * 16 * 16          # 5 training views
    # The PARAMETER COUNT CANNOT CATCH A DIMS-ORDER BUG -- 16x16x8 and 8x16x16 have
    # the same product, and the mutation battery duly found `dims_order_reversed`
    # surviving. `--bilagrid-dims` is (x, y, guidance), Brush's order
    # (brush-train/src/config.rs:1001), and the tensor is [N, 12, guidance, y, x].
    assert ap["dims"] == [16, 16, 8]
    assert ap["max_abs_dev"] > 0.0, "the grid never moved off identity in 40 steps"
    assert ap["tv"] > 0.0
    off = T.train(_args(steps=40, eval_every=40), scene=_synthetic_scene())
    assert off["metrics"]["appearance"] is None


@mps
def test_the_appearance_group_gets_the_brush_lr_schedule_and_nothing_else_does():
    """The grid's LR is warmup-then-exponential-decay (train_state.rs:16-19,
    44-53); the gaussian groups have their own schedule and must not be touched
    by it. Fails if the schedule is applied to every param group, or to none."""
    from metal_gauss.bilagrid import warmup_exp_lr
    from metal_gauss.train import appearance_lr_at
    base, steps = 2e-3, 30000
    assert appearance_lr_at(0, base, steps) == warmup_exp_lr(0, base, decay_steps=steps)
    assert appearance_lr_at(0, base, steps) < 0.05 * base       # warmup really warms
    assert appearance_lr_at(999, base, steps) > 0.99 * base
    assert appearance_lr_at(29999, base, steps) < 0.02 * base   # and really decays

    # ...AND that the schedule is WRITTEN ONTO THE GROUP, not merely computed. A
    # schedule the training loop calculates and drops on the floor is invisible to
    # every assertion above, so this reads the LR the optimiser actually finished
    # on, out of the report.
    from metal_gauss import train as T
    out = T.train(_args(appearance="bilagrid", steps=40, eval_every=40),
                  scene=_synthetic_scene())
    want = appearance_lr_at(39, 2e-3, 40)
    got = out["metrics"]["appearance"]["lr_final"]
    assert abs(got - want) < 1e-12, f"group lr {got} is not the scheduled {want}"
    assert abs(got - 2e-3) > 1e-9, "the fixture cannot tell a live schedule from a constant lr"
    # the global modes must NOT be touched by it
    aff = T.train(_args(appearance="affine", steps=40, eval_every=40),
                  scene=_synthetic_scene())
    assert aff["metrics"]["appearance"]["lr_final"] == 1e-3

    # ...AND NEITHER MUST THE GAUSSIANS. The mutation battery caught this: dropping
    # the `name == "appearance"` guard applies the appearance schedule to means,
    # scales, quats, opacity and both SH groups -- a run-destroying change that
    # every assertion above is blind to, because they are all about the appearance
    # group. Compare the other groups against an `--appearance off` run.
    off = T.train(_args(steps=40, eval_every=40), scene=_synthetic_scene())
    on_g, off_g = out["metrics"]["lr_groups"], off["metrics"]["lr_groups"]
    for name in ("means", "scales", "quats", "opac", "sh_dc", "sh_rest"):
        assert on_g[name] == off_g[name], (
            f"the appearance LR schedule reached the {name!r} group: "
            f"{off_g[name]} -> {on_g[name]}")
    # control: those groups must carry DIFFERENT lrs from each other and from the
    # appearance group, or "unchanged" would be satisfiable by a constant.
    assert len({on_g[n] for n in ("means", "scales", "opac", "sh_rest")}) == 4
    assert on_g["appearance"] not in {off_g[n] for n in off_g if n != "appearance"}
