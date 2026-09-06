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
