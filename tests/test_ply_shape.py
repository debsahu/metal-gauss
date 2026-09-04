"""`scripts/ply_shape.py` -- the hard-needle column, recovered from an exported ply.

The tool exists because `hard_needle_frac` was added to `shape_metrics` after Task 19's ten
arms were trained, and eight hours of GPU is not the way to recover a reported-only column.

Both things that could be wrong here fail SILENTLY, producing a plausible number for a
different quantity: the ply field order, and the median convention. Each gets a test that a
wrong implementation passes with a number, not with a crash.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ply_shape import (HARD_NEEDLE_ASPECT, cross_check, lower_median,  # noqa: E402
                       read_ply_scales, shape_from_ply)

# THE ORIGINAL FIXTURE HERE WAS THE INRIA ORDER and did not know it: with 11 properties,
# `x y z opacity scale_0..2 rot_0..3` puts the scales at exactly `[-7:-4]`, which is what a
# fixed-offset parser reads. A mutant that replaced the by-name lookup with `rows[:, -7:-4]`
# therefore SURVIVED -- the fixture could not tell the two implementations apart.
#
# `PROPS` below breaks that offset deliberately: the scales sit early and two trailing
# fields shift `[-7:-4]` onto rot/extra columns. `INRIA_PROPS` is kept so the real layout is
# still covered, and a test asserts the fixture's discriminating power directly.
PROPS = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity",
         "rot_0", "rot_1", "rot_2", "rot_3", "f_extra_0", "f_extra_1"]
INRIA_PROPS = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
               "rot_0", "rot_1", "rot_2", "rot_3"]


def test_the_fixture_layout_can_actually_SEPARATE_by_name_from_by_offset():
    """Assert the fixture's discriminating power, so a future reordering cannot quietly
    re-pin the parser test the way the original one did."""
    assert [PROPS.index(f"scale_{i}") for i in range(3)] != [len(PROPS) - 7 + i
                                                             for i in range(3)]
    assert [INRIA_PROPS.index(f"scale_{i}") for i in range(3)] == \
        [len(INRIA_PROPS) - 7 + i for i in range(3)], \
        "INRIA_PROPS is supposed to BE the offset layout; if not, it covers nothing"


def _write_ply(path: Path, scales: np.ndarray, props=PROPS) -> None:
    n = scales.shape[0]
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    hdr += [f"property float {p}" for p in props] + ["end_header", ""]
    rows = np.zeros((n, len(props)), dtype="<f4")
    for j, p in enumerate(props):
        # every non-scale column gets a distinctive decoy value, so reading the wrong
        # offset produces a WRONG NUMBER rather than a crash -- which is the failure mode
        rows[:, j] = {"x": 11.0, "y": 22.0, "z": 33.0, "opacity": 44.0,
                      "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
                      "f_extra_0": 55.0, "f_extra_1": 66.0}.get(p, 0.0)
    for j, name in enumerate(("scale_0", "scale_1", "scale_2")):
        rows[:, props.index(name)] = scales[:, j]
    path.write_bytes("\n".join(hdr).encode() + rows.tobytes())


@pytest.mark.parametrize("props", [PROPS, INRIA_PROPS], ids=["shifted", "inria"])
def test_the_parser_reads_scales_BY_NAME_not_by_a_fixed_offset(tmp_path, props):
    """Would catch a parser that assumes the INRIA layout. The decoy columns are finite and
    plausible (11/22/33/44/55/66 and a unit quaternion), so a wrong offset yields WRONG
    SCALES rather than a crash -- and the `shifted` case is the one that separates the two
    implementations, the `inria` case only proving the real layout still reads."""
    log_s = np.log(np.array([[0.01, 0.02, 0.04], [0.001, 0.03, 0.05]], dtype=np.float32))
    p = tmp_path / "a.ply"
    _write_ply(p, log_s.astype("<f4"), props)
    got = read_ply_scales(p)
    assert got == pytest.approx(log_s, abs=1e-6)


def test_the_median_is_torchs_LOWER_median_not_numpys_average(tmp_path):
    """`torch.median` returns the lower of the two middle values on an even count; np.median
    averages them. At 500,000 splats those are different numbers, and the trainer's is the
    one the report holds.

    Would catch `np.median`: the two differ by 0.5 on this fixture and the wrong one is
    still a perfectly plausible median.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert lower_median(x) == 2.0
    assert np.median(x) == 2.5


def test_hard_needles_are_counted_at_the_DELIVERY_threshold_not_the_needle_one(tmp_path):
    """`needle_frac` is aspect < 0.1 and `hard_needle_frac` is aspect < 0.01 -- a 10x
    different bar. Would catch the two thresholds being swapped or shared, which would make
    the new column a duplicate of the old one and change no verdict while looking measured.

    The fixture is built so the two counts DIFFER: 1 of 4 below 0.01, 3 of 4 below 0.1.
    """
    smax = 1.0
    smids = [0.005, 0.05, 0.09, 0.5]           # aspects: hard, needle, needle, neither
    log_s = np.log(np.array([[1e-4, m, smax] for m in smids], dtype=np.float32))
    p = tmp_path / "b.ply"
    _write_ply(p, log_s.astype("<f4"))
    got = shape_from_ply(p)
    assert got["hard_needle_frac"] == pytest.approx(0.25)
    assert got["needle_frac"] == pytest.approx(0.75)
    assert HARD_NEEDLE_ASPECT == 0.01


def test_aspect_uses_smid_over_smax_regardless_of_the_stored_axis_ORDER(tmp_path):
    """The three scale columns are not sorted in the ply. Would catch `scale_1 / scale_2`
    taken literally, which is right only when the file happens to be sorted."""
    log_s = np.log(np.array([[0.04, 0.001, 0.02], [0.001, 0.02, 0.04]], dtype=np.float32))
    p = tmp_path / "c.ply"
    _write_ply(p, log_s.astype("<f4"))
    got = shape_from_ply(p)
    assert got["aspect_p50"] == pytest.approx(0.5, rel=1e-5)      # 0.02 / 0.04, both rows


def _report(tmp_path, **shape) -> Path:
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"metrics": {"shape": shape}}))
    return f


def test_the_cross_check_REFUSES_when_the_recomputation_disagrees(tmp_path):
    """The guard that makes a shape sidecar trustworthy at all. Would catch a tool that
    writes its number and reports the disagreement as a warning."""
    got = {"aspect_p50": 0.5, "needle_frac": 0.25, "smid_p50_mm": 20.0,
           "smax_p50_mm": 40.0, "hard_needle_frac": 0.1}
    cross_check(got, _report(tmp_path, aspect_p50=0.5, needle_frac=0.25,
                             smid_p50_mm=20.0, smax_p50_mm=40.0))
    with pytest.raises(SystemExit, match="needle_frac"):
        cross_check(got, _report(tmp_path, aspect_p50=0.5, needle_frac=0.30,
                                 smid_p50_mm=20.0, smax_p50_mm=40.0))


def test_the_cross_check_tolerance_ADMITS_float32_storage_but_REJECTS_one_splat(tmp_path):
    """The tolerance has to sit between two known error sizes, and this asserts both ends.

    Below: the report holds float32, so `needle_frac` 0.157376 reads back as
    0.1573760062456131. The tool's first version rejected that on 8 of 10 real arms.
    Above: the smallest REAL disagreement is one misclassified splat out of 500,000, i.e.
    2e-6 relative. A tolerance that admits that admits anything.
    """
    got = {"aspect_p50": 0.5, "needle_frac": float(np.float32(0.157376)),
           "smid_p50_mm": 20.0, "smax_p50_mm": 40.0}
    cross_check(got, _report(tmp_path, needle_frac=0.157376))          # f32 noise: fine
    one_splat = {**got, "needle_frac": 0.157376 + 1.0 / 500_000}
    with pytest.raises(SystemExit, match="needle_frac"):
        cross_check(one_splat, _report(tmp_path, needle_frac=0.157376))


def test_an_ascii_ply_is_REFUSED_rather_than_parsed_as_binary(tmp_path):
    p = tmp_path / "d.ply"
    p.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float scale_0\n"
                  b"property float scale_1\nproperty float scale_2\nend_header\n0 0 0\n")
    with pytest.raises(SystemExit, match="binary_little_endian"):
        read_ply_scales(p)


def test_a_ply_with_no_scale_fields_is_REFUSED(tmp_path):
    p = tmp_path / "e.ply"
    p.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                  b"property float x\nend_header\n" + struct.pack("<f", 1.0))
    with pytest.raises(SystemExit, match="scale_0"):
        read_ply_scales(p)


def test_a_truncated_body_is_REFUSED_rather_than_silently_short(tmp_path):
    """Would catch `np.fromfile` without a count check, which returns whatever is there and
    computes a median over a prefix of the splats."""
    log_s = np.log(np.array([[0.01, 0.02, 0.04]] * 4, dtype=np.float32))
    p = tmp_path / "f.ply"
    _write_ply(p, log_s.astype("<f4"))
    p.write_bytes(p.read_bytes()[:-8])
    with pytest.raises(SystemExit, match="body holds"):
        read_ply_scales(p)


# ------------------------------------------- the sidecar contract plane_aux_arms relies on

def test_an_UNVERIFIED_sidecar_is_refused_by_the_battery(tmp_path):
    """`hard_needle_from_sidecar` must not read a shape file that does not record a passing
    cross-check. Would catch a `.get("hard_needle_frac")` that trusts any well-formed JSON
    -- and a shape file computed from another arm's ply is well-formed."""
    from plane_aux_arms import hard_needle_from_sidecar
    (tmp_path / "P0.shape.json").write_text(json.dumps({"hard_needle_frac": 0.9}))
    with pytest.raises(SystemExit, match="cross-check"):
        hard_needle_from_sidecar(tmp_path, "P0")
    (tmp_path / "P0.shape.json").write_text(
        json.dumps({"hard_needle_frac": 0.9, "verified_against_report": True}))
    assert hard_needle_from_sidecar(tmp_path, "P0") == 0.9
    assert hard_needle_from_sidecar(tmp_path, "MISSING") is None


def test_the_ten_ARCHIVED_sidecars_all_record_a_passing_cross_check():
    """Provenance for every number this re-grade reports. Each sidecar reproduces its arm's
    own `aspect_p50`, `needle_frac`, `smid_p50_mm` and `smax_p50_mm` from the ply, so the
    hard-needle column beside them describes THAT reconstruction and not another."""
    root = Path(__file__).resolve().parents[1] / "bench/results/plane_aux"
    seen = 0
    for scene in ("pgeom", "arkit"):
        for tag in ("F0", "F1", "F2", "P0", "M0"):
            f = root / scene / f"{tag}.shape.json"
            assert f.exists(), f
            d = json.loads(f.read_text())
            assert d["verified_against_report"] is True
            assert d["n_splats"] == 500_000
            rep = json.loads((root / scene / f"{tag}.report.json").read_text())
            assert d["reproduced"]["needle_frac"] == rep["metrics"]["shape"]["needle_frac"]
            seen += 1
    assert seen == 10


def test_the_committed_TIER1_shape_columns_use_the_AVERAGE_median_convention():
    """TWO MEDIAN CONVENTIONS EXIST IN THIS PROJECT'S OWN ARTIFACTS and they are not
    interchangeable past ~5 significant figures. Established by recomputing three Tier 1
    arms both ways and matching to 10 digits:

      lower    torch.median -- the TRAINER's `metrics.shape` (all ten Task 19 arms)
      average  np.median    -- Tier 1's `collected.json` `ply.*` columns, and therefore
                              research/metal-gauss.md section 8.1's aspect/needle figures

    On B0a they differ by 1.9e-6 relative on smid, which is far below any verdict and far
    above float noise -- so a cross-check tight enough to catch a misread field trips over
    it, and it did, on all eight Tier 1 arms.

    Would catch the convention argument being ignored: `lower` on B0a gives aspect
    0.2956616021 against the committed 0.2956620955.
    """
    from ply_shape import shape_from_ply, MEDIANS
    assert set(MEDIANS) == {"lower", "average"}
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert MEDIANS["lower"](x) == 2.0 and MEDIANS["average"](x) == 2.5
    row = json.loads((Path(__file__).resolve().parents[1] /
                      "bench/results/plane_aux/tier1_void_row.json").read_text())
    assert row["median_convention"].startswith("average")
    # B0a, both conventions, as measured -- the committed value must match ONE of them,
    # and the assertion is that it is the average one, to a precision that separates them.
    assert row["arms"]["B0a"]["values"]["run.aspect_p50"] == pytest.approx(
        0.2956620955, abs=1e-9)
    assert abs(0.2956620955 - 0.2956616021) > 4e-7, \
        "if the two conventions agreed here this test would prove nothing"


def test_the_convention_argument_REACHES_the_computed_columns(tmp_path):
    """The test above pins the CONSTANTS and the committed file; this one pins the WIRING.

    A mutant that hard-coded `MEDIANS["lower"]` inside `shape_from_ply` survived the
    constants test entirely -- the dict was still right, the committed JSON was still right,
    and the argument was ignored. Four splats with an even count make the two conventions
    give different medians, so a `--median average` that quietly computes `lower` fails.

    The fixture's discriminating power is asserted first, so a future edit that makes the
    two agree fails here rather than going slack.
    """
    from ply_shape import shape_from_ply
    smids = [0.10, 0.20, 0.30, 0.40]
    log_s = np.log(np.array([[1e-4, m_, 1.0] for m_ in smids], dtype=np.float32))
    f = tmp_path / "conv.ply"
    _write_ply(f, log_s.astype("<f4"))
    lo, av = shape_from_ply(f, "lower"), shape_from_ply(f, "average")
    assert lo["median_convention"] == "lower" and av["median_convention"] == "average"
    assert lo["smid_p50_mm"] == pytest.approx(200.0, rel=1e-5)      # lower of 0.20/0.30
    assert av["smid_p50_mm"] == pytest.approx(250.0, rel=1e-5)      # their average
    assert abs(lo["aspect_p50"] - av["aspect_p50"]) > 0.04, \
        "if the fixture's two conventions agreed it could not separate the wiring"
    # and the fractions are convention-free, which is why hard_needle_frac is comparable
    # between Tier 1 (average) and Task 19 (lower)
    assert lo["hard_needle_frac"] == av["hard_needle_frac"]
    assert lo["needle_frac"] == av["needle_frac"]


def test_a_reference_with_NO_shape_columns_cannot_stamp_a_file_as_verified(tmp_path):
    """THE DEFECT THIS TOOL SHIPPED AND THE TIER 1 ARMS EXPOSED. `cross_check` skips
    columns the reference lacks, so a report with an EMPTY `metrics.shape` compared nothing
    at all, found nothing wrong, and the file was written stamped `verified_against_report:
    true`. Five Tier 1 arms -- the VOID row among them -- have exactly such reports, because
    `metrics.shape` was added to the trainer mid-batch.

    A check that passes because it did not run is the failure CLAUDE.md names. `main` now
    refuses unless the two columns that pin field order and median convention were both
    compared.

    Would catch the required-checks list being emptied.
    """
    import subprocess
    log_s = np.log(np.array([[0.01, 0.02, 0.04]] * 4, dtype=np.float32))
    ply = tmp_path / "a.ply"
    _write_ply(ply, log_s.astype("<f4"))
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({"metrics": {"shape": {}}}))
    assert cross_check({"aspect_p50": 0.5, "needle_frac": 0.25,
                        "median_convention": "lower"}, rep) == {}, \
        "an empty reference compares nothing -- that is the hazard, not the bug"
    r = subprocess.run([sys.executable,
                        str(Path(__file__).resolve().parents[1] / "scripts/ply_shape.py"),
                        str(ply), "--report", str(rep), "--out", str(tmp_path / "o.json")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "verified" in (r.stdout + r.stderr)
    assert not (tmp_path / "o.json").exists(), \
        "nothing may be written when the verification did not happen"
