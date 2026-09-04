"""bench/dn_neighbour_gate.py -- the Task 20 measurement tool.

The tool produces the numbers a pre-registered decision rests on, so its counting, its
ply reader and its refusals are tested like production code. CPU-only, so CI runs them.
"""
import json

import pytest
import torch

from bench.dn_neighbour_gate import KIND, affected_stats, params_from_ply, summarise


# ------------------------------------------------------------------------ the counting

def _grid(h=4, w=4):
    return torch.ones(h, w, dtype=torch.bool)


def test_affected_stats_counts_zero_when_nothing_is_uncovered():
    """CATCHES a statistic that is nonzero by construction (a border artefact, an
    inverted predicate). The interior of a fully covered frame must be untouched."""
    c = _grid()
    in_loss = torch.zeros(4, 4, dtype=torch.bool)
    in_loss[:-1, :-1] = True                       # exactly what normals_from_depth emits
    s = affected_stats(in_loss, c, c, c)
    assert s["in_loss_px"] == 9 and s["affected_px"] == 0 and s["affected_frac"] == 0.0


def test_affected_stats_finds_a_pixel_whose_right_neighbour_is_uncovered():
    """CATCHES the statistic being unable to fire at all -- the failure mode where a
    'measurement' returns 0 because it is measuring nothing. The fixture's discriminating
    power is asserted: this case and the all-covered case must differ."""
    c = _grid()
    c[0, 1] = False                                 # (0,0)'s right neighbour
    alpha_ok = c.clone()
    in_loss = torch.zeros(4, 4, dtype=torch.bool)
    in_loss[:-1, :-1] = True
    in_loss = in_loss & c                           # (0,1) itself is not in loss
    s = affected_stats(in_loss, c, alpha_ok, _grid())
    assert s["affected_px"] >= 1
    assert s["affected_px"] != affected_stats(in_loss, _grid(), _grid(), _grid())["affected_px"]
    assert s["affected_by_alpha_px"] == s["affected_px"]
    assert s["affected_by_keep_px"] == 0            # no mask involved


def test_affected_stats_separates_the_two_causes():
    """CATCHES a cause split that reports one bucket for both, or that reads the base
    pixel's cause instead of the neighbour's. One hole from alpha, one from keep, in
    different places, and the two buckets must name them separately."""
    alpha_ok, keep_ok = _grid(), _grid()
    alpha_ok[0, 1] = False                          # gates (0,0), cause = alpha
    keep_ok[3, 2] = False                           # gates (2,2) via (v+1,u), cause = keep
    covered = alpha_ok & keep_ok
    in_loss = torch.zeros(4, 4, dtype=torch.bool)
    in_loss[:-1, :-1] = True
    in_loss = in_loss & covered
    s = affected_stats(in_loss, covered, alpha_ok, keep_ok)
    assert s["affected_by_alpha_px"] >= 1 and s["affected_by_keep_px"] >= 1
    assert s["affected_by_both_px"] == 0
    assert (s["affected_by_alpha_px"] + s["affected_by_keep_px"]
            - s["affected_by_both_px"]) == s["affected_px"]


def test_affected_stats_raises_when_the_cause_split_does_not_partition():
    """CATCHES a silently wrong split. The inclusion-exclusion identity is the only thing
    tying the three cause counts to the total, so it must be a hard failure, not a note."""
    alpha_ok, keep_ok = _grid(), _grid()
    covered = _grid()
    covered[0, 1] = False                # covered says gated; NEITHER cause explains it
    in_loss = torch.zeros(4, 4, dtype=torch.bool)
    in_loss[:-1, :-1] = True
    with pytest.raises(RuntimeError, match="partition"):
        affected_stats(in_loss, covered, alpha_ok, keep_ok)


# ------------------------------------------------------------------------- the ply read

def test_params_from_ply_round_trips_export_ply_bit_for_bit(tmp_path):
    """CATCHES the easy wrong reader. `io.load_ply` ACTIVATES opacity, scale and quat;
    the trainer's parameters are the pre-activation values, and re-inverting a sigmoid
    loses the rails. A measurement rendered from a mis-read ply is not a measurement of
    that ply."""
    from metal_gauss.train import export_ply
    g = torch.Generator().manual_seed(11)
    n = 37
    p = {"means": torch.randn(n, 3, generator=g),
         "log_scales": torch.randn(n, 3, generator=g),
         "quats": torch.randn(n, 4, generator=g),
         "logit_opac": torch.randn(n, generator=g) * 6,      # includes near-rail values
         "sh_dc": torch.randn(n, 1, 3, generator=g),
         "sh_rest": torch.randn(n, 15, 3, generator=g)}
    f = tmp_path / "x.ply"
    export_ply(p, str(f))
    q, m = params_from_ply(str(f), "cpu")
    assert m == n
    for k in p:
        assert torch.equal(q[k], p[k]), k


# -------------------------------------------------------------------------- the summary

def _meas(scene, step, frac, grad=1e-9, synthetic=False):
    return {"kind": KIND, "schema": 1, "synthetic": synthetic, "scene": scene,
            "step_label": step,
            "aggregate": {"affected_frac_overall": frac,
                          "affected_by_alpha_frac_overall": frac,
                          "affected_by_keep_frac_overall": 0.0,
                          "affected_frac_max": frac * 2,
                          "loss_dn_rel_delta_median": 1e-4,
                          "grad_rel_max_over_all": grad,
                          "affected_frac_under_line": frac < 0.005,
                          "grad_under_kernel_f32_error": grad < 7.02e-6}}


def _write(tmp_path, name, d):
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def test_summary_refuses_a_synthetic_file(tmp_path):
    """THE ONE THAT MATTERS FOR PROVENANCE. Task 19's summary globbed its own smoke
    fixtures, including a deliberately fabricated pass/regress pair. A fixture must not be
    able to reach a verdict even when it is handed over explicitly."""
    f = _write(tmp_path, "a.json", _meas("pgeom", "30k", 0.001, synthetic=True))
    with pytest.raises(RuntimeError, match="synthetic"):
        summarise([f], ["pgeom"])


def test_summary_refuses_a_file_of_the_wrong_kind(tmp_path):
    d = _meas("pgeom", "30k", 0.001)
    d["kind"] = "something_else"
    with pytest.raises(RuntimeError, match="kind"):
        summarise([_write(tmp_path, "a.json", d)], ["pgeom"])


def test_summary_refuses_a_named_scene_with_no_measurement(tmp_path):
    """CATCHES a verdict computed over fewer scenes than the rule requires -- 'CLOSE on
    every real dataset measured' is a claim about a set, so the set must be checked."""
    f = _write(tmp_path, "a.json", _meas("pgeom", "30k", 0.001))
    with pytest.raises(RuntimeError, match="no measurement"):
        summarise([f], ["pgeom", "pmask"])


def test_summary_refuses_a_scene_that_was_not_named(tmp_path):
    """CATCHES the opposite leak: an unnamed file quietly widening the verdict."""
    fs = [_write(tmp_path, "a.json", _meas("pgeom", "30k", 0.001)),
          _write(tmp_path, "b.json", _meas("mystery", "30k", 0.9))]
    with pytest.raises(RuntimeError, match="not named"):
        summarise(fs, ["pgeom"])


def test_summary_refuses_duplicate_scene_step(tmp_path):
    fs = [_write(tmp_path, "a.json", _meas("pgeom", "30k", 0.001)),
          _write(tmp_path, "b.json", _meas("pgeom", "30k", 0.004))]
    with pytest.raises(RuntimeError, match="duplicate"):
        summarise(fs, ["pgeom"])


def test_summary_escalates_when_any_single_scene_is_over_the_line(tmp_path):
    """THE RULE, and it is an ANY not an average: one scene over 0.5% escalates even when
    the other is far enough under to drag the MEAN below the line.

    The fixture's discriminating power is asserted first, and it is the whole point of
    these particular numbers: an earlier version used 0.0001 and 0.02, whose mean is
    0.01005 -- still over the line -- so a mean-based implementation passed it and the
    mutant survived. A rule tested only where it cannot bind is not tested.
    """
    a, b = 0.0001, 0.006
    assert (a + b) / 2 < 0.005 < b, "fixture no longer separates 'any' from 'mean'"
    fs = [_write(tmp_path, "a.json", _meas("pgeom", "30k", a)),
          _write(tmp_path, "b.json", _meas("pmask", "30k", b))]
    out = summarise(fs, ["pgeom", "pmask"])
    assert out["verdict"] == "ESCALATE"
    assert out["falsifier_refuted"] is True and out["falsifier_supported"] is False
    assert out["worst_affected_frac_overall"] == b


def test_summary_closes_only_when_every_scene_is_under_the_line(tmp_path):
    fs = [_write(tmp_path, "a.json", _meas("pgeom", "30k", 0.0001)),
          _write(tmp_path, "b.json", _meas("pmask", "30k", 0.0049))]
    out = summarise(fs, ["pgeom", "pmask"])
    assert out["verdict"] == "CLOSE"
    assert out["falsifier_supported"] is True and out["falsifier_refuted"] is False


def test_summary_does_not_call_the_falsifier_supported_on_a_large_gradient(tmp_path):
    """CATCHES conflating the two halves of the falsifier. Under the line on the affected
    fraction is NOT enough: the gradient difference must also be under the kernel's own
    f32 distance from truth, or the framing survives."""
    fs = [_write(tmp_path, "a.json", _meas("pgeom", "30k", 0.0001, grad=1e-3))]
    out = summarise(fs, ["pgeom"])
    assert out["verdict"] == "CLOSE" and out["falsifier_supported"] is False


def test_main_does_not_write_an_output_file_when_it_fails(tmp_path):
    """CATCHES the defect Task 19 shipped and caught only by luck: a result file written
    unconditionally, so a failed or mis-configured run leaves a well-formed file carrying
    somebody else's numbers. The file must not exist at all."""
    from bench.dn_neighbour_gate import main
    f = _write(tmp_path, "a.json", _meas("pgeom", "30k", 0.001))
    out = tmp_path / "summary.json"
    with pytest.raises(RuntimeError):
        main(["--summary", f, "--scenes", "pgeom,pmask", "--out", str(out)])
    assert not out.exists()


def test_main_writes_the_output_file_when_it_succeeds(tmp_path):
    """The guard above must not pass by never writing anything at all."""
    from bench.dn_neighbour_gate import main
    f = _write(tmp_path, "a.json", _meas("pgeom", "30k", 0.001))
    out = tmp_path / "summary.json"
    main(["--summary", f, "--scenes", "pgeom", "--out", str(out)])
    assert json.loads(out.read_text())["verdict"] == "CLOSE"


# ------------------------------------------------------- end to end, through a real render

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _synthetic_scene(root, W=64, H=48, n_views=6, stripe=(20, 29)):
    """A tiny PINHOLE scene: n_views identity-rotation cameras 3 m in front of a plane of
    splats, and a sidecar mask that DROPS a vertical stripe. The stripe is the point --
    it is the only way to make `keep` fire through a full render rather than in a unit
    fixture."""
    import numpy as np
    from PIL import Image
    (root / "sparse").mkdir(); (root / "images").mkdir(); (root / "masks").mkdir()
    (root / "sparse" / "cameras.txt").write_text(
        f"1 PINHOLE {W} {H} {W} {W} {W / 2} {H / 2}\n")
    lines, rng = [], np.random.default_rng(0)
    for i in range(n_views):
        tx = -0.3 + 0.12 * i
        lines.append(f"{i + 1} 1 0 0 0 {tx} 0 3 1 v{i}.png\n\n")
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(
            root / "images" / f"v{i}.png")
        m = np.zeros((H, W), np.uint8)
        m[:, stripe[0]:stripe[1]] = 255                  # 255 = DROP, this repo's polarity
        Image.fromarray(m).save(root / "masks" / f"v{i}.png")
    (root / "sparse" / "images.txt").write_text("".join(lines))
    pts = rng.uniform(-1.2, 1.2, (60, 3)); pts[:, 2] = 0.0
    (root / "sparse" / "points3D.txt").write_text("".join(
        f"{i} {p[0]} {p[1]} {p[2]} 200 200 200 0\n" for i, p in enumerate(pts)))


def _synthetic_ply(path, n=4000, seed=5):
    """An opaque plane of splats at world z=0, so alpha really exceeds 0.5 and the in-loss
    set is non-empty. A step-0 seed at logit_opac -2 would not guarantee that."""
    from metal_gauss.train import export_ply
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g) - 0.5) * 3.0
    means = torch.cat([xy, torch.zeros(n, 1)], 1)
    export_ply({"means": means,
                "log_scales": torch.full((n, 3), float(torch.log(torch.tensor(0.045)))),
                "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
                "logit_opac": torch.full((n,), 6.0),
                "sh_dc": torch.zeros(n, 1, 3), "sh_rest": torch.zeros(n, 15, 3)}, str(path))


@mps
def test_end_to_end_measures_a_real_render_and_keep_reaches_the_statistic(tmp_path):
    """THE INTEGRATION TEST. Everything above is a unit fixture; this one renders. It
    asserts (i) the measurement is NOT vacuous -- in-loss pixels exist, so a zero affected
    fraction would mean something; (ii) a mask stripe actually produces keep-caused
    affected pixels, which is the only cause that cannot be exercised on P-GEOM; and
    (iii) the record is marked synthetic and the summary REFUSES it, so this fixture can
    never be averaged into a result.
    """
    from bench.dn_neighbour_gate import build_parser, run, summarise
    _synthetic_scene(tmp_path)
    ply = tmp_path / "m.ply"
    _synthetic_ply(ply)
    args = build_parser().parse_args([
        "--colmap", str(tmp_path / "sparse"), "--images", str(tmp_path / "images"),
        "--masks", str(tmp_path / "masks"), "--max-resolution", "64",
        "--eval-split-every", "1000", "--views", "6", "--ply", str(ply),
        "--scene", "synthetic", "--step-label", "fixture", "--synthetic"])
    rec = run(args)
    a = rec["aggregate"]
    assert rec["synthetic"] is True
    assert a["in_loss_px_total"] > 0, "vacuous: nothing was in the loss to be affected by"
    assert 0.0 <= a["affected_frac_overall"] <= 1.0
    assert a["affected_by_keep_frac_overall"] > 0.0, (
        "the mask never reached the statistic through a real render")
    assert all(v["has_mask"] for v in rec["per_view"])
    with pytest.raises(RuntimeError, match="synthetic"):
        summarise_inline(rec, tmp_path)


def summarise_inline(rec, tmp_path):
    from bench.dn_neighbour_gate import summarise
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(rec))
    return summarise([str(p)], ["synthetic"])
