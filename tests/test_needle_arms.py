"""The arm table in `scripts/run_tier1_arms.sh` is a config file that nothing checked.

WHAT THIS CATCHES. A mistyped flag in `arm_flags` does not error: `metal_gauss.train`
rejects an unknown option, so a typo aborts the arm -- but a flag that EXISTS and is not
the one intended (`--filter-3d-every` for `--filter-3d`, `--scale-reg 0.01` for `0.0`)
trains a different configuration, writes a report, and is tabulated as the arm it is
named after. That is the shape this project keeps paying for: a check something other
than the thing being checked can satisfy. Every needle arm is a single-flag arm, so its
entire meaning is one string.

It is a SHELL function, so the test extracts it by brace matching and sources the
extracted text. Sourcing the runner itself is not possible: it is `set -euo pipefail`
with top-level argument parsing that exits without `--out`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_tier1_arms.sh"

# arm -> the EXACT flag string the arm must emit. Deliberately spelled out rather than
# derived from the runner, so the test and the thing tested cannot drift together.
NEEDLE_ARMS = {
    "N1": "--filter-3d",
    "N2": "--antialias",
    "N3": "--scale-reg 0.0",
    "N2b": "--antialias",
    "N4": "--num-downscales 0",
    "I0": "--inplane-isotropy-weight 0.01",
    "I1": "--inplane-isotropy-weight 0.1",
    "I2": "--inplane-isotropy-weight 1.0",
    "I3": "--inplane-isotropy-weight 10.0",
}

# The isotropy sweep must actually SPAN decades -- three arms that differ in the third
# decimal would look like a sweep in the table and be one weight in the data.
ISOTROPY_ARMS = ("I0", "I1", "I2", "I3")


def _extract_arm_flags(text: str) -> str:
    """The `arm_flags() { ... }` block, by brace matching from its opening line."""
    start = text.index("arm_flags() {")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise AssertionError("arm_flags() is not brace-balanced in " + str(RUNNER))


def _arm_flags(arm: str) -> subprocess.CompletedProcess:
    body = _extract_arm_flags(RUNNER.read_text())
    return subprocess.run(["bash", "-c", body + f'\narm_flags {arm}'],
                          capture_output=True, text=True)


@pytest.mark.parametrize("arm,expected", sorted(NEEDLE_ARMS.items()))
def test_needle_arm_emits_exactly_its_one_flag(arm, expected):
    r = _arm_flags(arm)
    assert r.returncode == 0, f"{arm} is not wired into arm_flags: {r.stderr.strip()}"
    assert r.stdout.strip() == expected, (
        f"{arm} emits {r.stdout.strip()!r}, not {expected!r}. A single-flag arm's entire "
        f"meaning is that string; a wrong one trains a different configuration and is "
        f"still tabulated under this name.")


def test_the_needle_arms_are_single_flag_arms():
    """Their comparability to the recorded Tier 1 arms rests on differing in ONE thing.
    N3 and N4 are `--flag value` (two tokens); N1 and N2 are bare switches (one)."""
    for arm, expected in NEEDLE_ARMS.items():
        assert expected.count("--") == 1, f"{arm} moves more than one flag: {expected}"


def test_N2b_repeats_N2_EXACTLY_and_differs_only_in_the_binary():
    """N2b exists to re-run --antialias on a binary where it does not emit NaN gradients.
    If its flag string ever drifts from N2's, the comparison stops being about the binary
    and becomes about the configuration, silently."""
    a, b = _arm_flags("N2"), _arm_flags("N2b")
    assert a.returncode == 0 and b.returncode == 0
    assert a.stdout.strip() == b.stdout.strip() == "--antialias", (a.stdout, b.stdout)


def test_an_unknown_arm_still_fails_loudly():
    """The guard the needle arms were inserted next to must survive their insertion --
    otherwise a typo in an --arms list trains a silent baseline under a treatment name."""
    r = _arm_flags("N9_does_not_exist")
    assert r.returncode != 0, "an unknown arm returned success"
    assert "unknown arm" in r.stderr


def test_baseline_arms_emit_nothing():
    """A floor arm that acquired a flag would poison every floor in the batch."""
    for arm in ("B0a", "B0b", "B0c"):
        r = _arm_flags(arm)
        assert r.returncode == 0 and r.stdout.strip() == "", (
            f"{arm} is meant to be a bare baseline but emits {r.stdout.strip()!r}")


def test_the_isotropy_sweep_spans_decades():
    """CATCHES a sweep that is not one: four arms whose weights differ by less than an
    order of magnitude would be four repeats wearing a sweep's name, and the batch could
    not distinguish 'the term does nothing' from 'the weight was wrong'."""
    ws = []
    for arm in ISOTROPY_ARMS:
        r = _arm_flags(arm)
        assert r.returncode == 0, f"{arm} is not wired: {r.stderr.strip()}"
        ws.append(float(r.stdout.strip().split()[-1]))
    assert ws == sorted(ws), f"the sweep is not monotone: {ws}"
    for lo, hi in zip(ws, ws[1:]):
        assert hi / lo >= 9.9, f"{lo} -> {hi} is less than a decade"


# ------------------------------------------------- the splatstats path (2026-09-06)

def _run_runner(tmp_path, mg_root, *extra):
    """Invoke the runner for real, with a MG_ROOT that has no sibling analyze/splatstats.

    Runs the actual script rather than an extracted fragment, because the defect this
    covers lives in the interaction between the derived SPLATSTATS default and `set -e`
    thirty-seven minutes later -- not in any one line."""
    ds = tmp_path / "ds"
    (ds / "sparse" / "0").mkdir(parents=True)
    (ds / "images").mkdir()
    seed = tmp_path / "seed.txt"; seed.write_text("1 0 0 0 0 0 0 0\n")
    out = tmp_path / "out"
    return subprocess.run(
        ["bash", str(RUNNER), "--dataset", str(ds), "--out", str(out),
         "--seed-cloud", str(seed), "--arms", "B0a", "--floors", "B0a,B0b,B0c",
         *extra],
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path),
             "MG_ROOT": str(mg_root)})


def test_an_unresolvable_splatstats_fails_BEFORE_any_arm_trains(tmp_path):
    """THE DEFECT THIS COVERS, and it cost 37 minutes of GPU on 2026-09-06.

    `SPLATSTATS` defaults to `$MG/../../analyze/splatstats`, i.e. it assumes metal-gauss
    is checked out at <repo>/compute/metal-gauss. On a standalone clone that path does not
    exist, `$(cd ... || echo "")` yields the EMPTY STRING, `cd ""` silently stays in the
    current directory, and the scorer runs `$MG/scripts/splat_stats.py`, which is not
    there. `set -e` then killed the driver -- AFTER all three floor arms had trained and
    BEFORE anything was scored. Nothing failed at launch; the batch simply stopped
    existing thirty-seven minutes later.

    The guard must fire during argument handling, so this test asserts on the ABSENCE of
    any PHASE line as well as on the exit status: a run that trains an arm and then
    complains has not been fixed."""
    mg = tmp_path / "not_a_repo"; (mg / "scripts").mkdir(parents=True)
    r = _run_runner(tmp_path, mg)
    assert r.returncode != 0, "an unresolvable splatstats was accepted"
    assert "splatstats" in (r.stderr + r.stdout).lower(), (r.stdout, r.stderr)
    assert "PHASE 1" not in r.stdout, (
        "the guard fired too late -- an arm had already been launched:\n" + r.stdout)


@pytest.mark.parametrize("kind", ["absent", "exists_but_empty"])
def test_an_EXPLICIT_splatstats_that_cannot_SCORE_is_refused(tmp_path, kind):
    """--splatstats is the escape hatch for a standalone clone, so a wrong value must not
    reintroduce the same silent failure by another door.

    THE SECOND CASE EXISTS BECAUSE A MUTANT SURVIVED WITHOUT IT. Weakening the guard from
    `-f "$SPLATSTATS/scripts/splat_stats.py"` to `-d "$SPLATSTATS"` passed the whole file,
    because the only value under test was a path that was neither. The realistic operator
    error is not a typo -- it is pointing at the repo root, or at `analyze/`, instead of
    `analyze/splatstats`: a directory that exists and cannot score. Only the file check
    separates the two, and only this case makes the test say so."""
    mg = tmp_path / "not_a_repo"; (mg / "scripts").mkdir(parents=True)
    if kind == "absent":
        target = tmp_path / "nope"
    else:
        target = tmp_path / "wrong_level"
        (target / "scripts").mkdir(parents=True)      # exists, but no splat_stats.py
        (target / "scripts" / "something_else.py").write_text("")
    r = _run_runner(tmp_path, mg, "--splatstats", str(target))
    assert r.returncode != 0, f"{kind}: an unusable --splatstats was accepted"
    assert "splatstats" in (r.stderr + r.stdout).lower(), (r.stdout, r.stderr)
    assert "PHASE 1" not in r.stdout


def test_a_VALID_explicit_splatstats_is_accepted(tmp_path):
    """The control for the two refusals above: without it, a guard that refused every
    value whatsoever would pass both of them and block every real run."""
    mg = tmp_path / "not_a_repo"; (mg / "scripts").mkdir(parents=True)
    ss = tmp_path / "splatstats"; (ss / "scripts").mkdir(parents=True)
    (ss / "scripts" / "splat_stats.py").write_text("")
    r = _run_runner(tmp_path, mg, "--splatstats", str(ss))
    assert "splatstats does not resolve" not in (r.stderr + r.stdout), (
        "the guard refused a splatstats directory that has the script in it:\n" + r.stderr)
    assert "PHASE 1" in r.stdout, "the run did not get past argument handling"


def test_a_run_with_NO_seed_cloud_does_not_need_splatstats_at_all(tmp_path):
    """The guard must be conditional. A scene with no reference cloud skips splatstats by
    design (`score_arm` says so and reports the geometry metrics as UNDEFINED), and a
    guard that demanded it anyway would block every such scene -- lego among them."""
    mg = tmp_path / "not_a_repo"; (mg / "scripts").mkdir(parents=True)
    ds = tmp_path / "ds2"; (ds / "sparse" / "0").mkdir(parents=True); (ds / "images").mkdir()
    r = subprocess.run(
        ["bash", str(RUNNER), "--dataset", str(ds), "--out", str(tmp_path / "out2"),
         "--arms", "B0a", "--floors", "B0a,B0b,B0c"],
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path),
             "MG_ROOT": str(mg)})
    assert "splatstats" not in (r.stderr + r.stdout).lower(), (
        "the guard fired on a run that never needed splatstats:\n" + r.stderr)
