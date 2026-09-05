"""The run report must record WHICH LOSS CHAIN ACTUALLY RAN, observed at the branch.

Task 20's escalation arm is 8 training arms whose treatment variable is
`MG_DN_GATE_NEIGHBOURS=1`, and every arm in a pair must also be forced onto the torch
loss chain with `MG_TORCH_LOSS=1` or gate-vs-no-gate is confounded with torch-vs-fused.
Before this file, `_run_report` recorded `vars(args)` and `_env_snapshot()` and NEITHER
carries an environment variable -- so an arm whose `MG_TORCH_LOSS=1` was lost across a
`nohup`, a typo or a non-inheriting shell would have been graded as a torch arm on the
strength of nothing at all. That is precisely the failure `_run_report`'s own docstring
records twice (`--steps-scaler`, then `--budget`).

READING `os.environ` AT REPORT TIME WOULD NOT FIX IT. That answers "what was requested",
which is the question that was already answerable and already wrong. `_use_fused_loss()`
returning True is not sufficient either: `geometry_terms` takes the fused branch only if
several other preconditions hold too (all three weights positive, MPS device, float32
aux), so `MG_TORCH_LOSS=0` does not imply the fused kernel ran. The counters are bumped
AT THE BRANCH, by the branch, and the two disagreement tests below are the ones that
separate an observation from a re-read of the environment.

COUNTERS, NOT BOOLEANS: a boolean cannot distinguish "every view took the torch path"
from "one view did". And every key is always present as an integer, so a report of all
zeros -- no geometry term ran at all, a failure on a geometry-recipe arm -- can never be
read as "ran ungated".

Each test says what it CATCHES and each was confirmed to FAIL against a wrong
implementation, asserted on the failing test's NAME, never on a failure count. Needs
`PYTHONDONTWRITEBYTECODE=1` to be trustworthy (research/metal-gauss.md section 12.5).
"""
import argparse

import pytest
import torch

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

KEYS = ("fused_calls", "torch_calls", "dn_gated_calls", "dn_ungated_calls",
        "dn_skipped_calls")


def _args(**kw):
    d = dict(depth_loss_weight=1.0, normal_loss_weight=0.2, depth_normal_weight=0.05,
             depth_loss_space="disparity", depth_source="center")
    d.update(kw)
    return argparse.Namespace(**d)


def _tiny_maps(dev="cpu"):
    """The same fixture as tests/test_dn_neighbour_gate.py, deliberately: the golden
    values below were captured through it on the pre-change commit."""
    g = torch.Generator().manual_seed(3)
    h, w = 12, 14
    a = torch.ones(h, w)
    a[:, 5] = 0.1
    z = (2.0 + torch.rand(h, w, generator=g) * 0.2) * a
    z = z[..., None].expand(h, w, 3).contiguous()
    n = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    n_sum = a[..., None] * n
    gt_d = 2.0 + torch.rand(h, w, generator=g) * 0.2
    gt_n = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    return [x.to(dev) for x in (n_sum, z, a, gt_d, gt_n)]


_K = torch.tensor([[300.0, 0, 7.0], [0, 300.0, 6.0], [0, 0, 1.0]])


# ------------------------------------------------------ observation, not the environment

def test_report_says_torch_ran_when_the_environment_asked_for_fused(monkeypatch):
    """THE MUTANT KILLER. Here the environment and the branch DISAGREE: `MG_TORCH_LOSS`
    is unset, so `_use_fused_loss()` is True and the environment is asking for the fused
    kernel -- but the tensors are on CPU, the fused branch's `alpha.device.type == "mps"`
    precondition fails, and the TORCH chain runs.

    CATCHES the wrong implementation this whole file exists to prevent: recording
    `not _use_fused_loss()` (or `os.environ["MG_TORCH_LOSS"]`) instead of the branch that
    executed. Such an implementation reports a fused call for a run that never touched
    the Metal kernel, which on Task 20's arms is a mislabelled loss chain.
    """
    from metal_gauss.train import (geometry_terms, loss_path_counters,
                                   reset_loss_path_counters, _use_fused_loss)
    monkeypatch.delenv("MG_TORCH_LOSS", raising=False)
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    assert _use_fused_loss() is True, "fixture broken: the environment must ASK for fused"
    reset_loss_path_counters()
    n_sum, z, a, gt_d, gt_n = _tiny_maps("cpu")
    geometry_terms(_args(), [n_sum, z], a, _K, gt_d, gt_n, None)
    c = loss_path_counters()
    assert c["torch_calls"] == 1, f"the torch branch ran once; report says {c}"
    assert c["fused_calls"] == 0, (
        f"reported a fused call for a CPU run the Metal kernel never saw: {c}. This is "
        f"the environment being echoed back, not the branch being observed.")


def test_report_says_dn_was_skipped_when_the_gate_was_requested_but_no_dn_term_ran(monkeypatch):
    """The second disagreement, for the treatment variable itself. `MG_DN_GATE_NEIGHBOURS`
    is set, so `_gate_dn_neighbours()` is True -- but `--depth-normal-weight 0` means no
    depth-normal term is computed at all, gated or otherwise.

    CATCHES recording the gate from the environment rather than from the call. A run with
    the treatment set and the term switched off must NOT be gradeable as a gated arm.
    """
    from metal_gauss.train import (geometry_terms, loss_path_counters,
                                   reset_loss_path_counters, _gate_dn_neighbours)
    monkeypatch.setenv("MG_TORCH_LOSS", "1")
    monkeypatch.setenv("MG_DN_GATE_NEIGHBOURS", "1")
    assert _gate_dn_neighbours() is True, "fixture broken: the environment must ask for the gate"
    reset_loss_path_counters()
    n_sum, z, a, gt_d, gt_n = _tiny_maps("cpu")
    geometry_terms(_args(depth_normal_weight=0.0), [n_sum, z], a, _K, gt_d, gt_n, None)
    c = loss_path_counters()
    assert c["dn_gated_calls"] == 0, f"claimed a gated dn call with the term off: {c}"
    assert c["dn_skipped_calls"] == 1, f"the dn term did not run and nothing says so: {c}"


def test_counters_count_calls_rather_than_recording_a_boolean(monkeypatch):
    """CATCHES a boolean, or a last-write-wins field. Two torch calls and one more must
    read 3, not 1 and not True -- an arm where the treatment reached one view out of 200
    is exactly the failure a boolean cannot see, and it is indistinguishable in a log."""
    from metal_gauss.train import (geometry_terms, loss_path_counters,
                                   reset_loss_path_counters)
    monkeypatch.setenv("MG_TORCH_LOSS", "1")
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    reset_loss_path_counters()
    n_sum, z, a, gt_d, gt_n = _tiny_maps("cpu")
    for _ in range(3):
        geometry_terms(_args(), [n_sum, z], a, _K, gt_d, gt_n, None)
    c = loss_path_counters()
    assert c["torch_calls"] == 3 and c["dn_ungated_calls"] == 3, c
    for k in KEYS:
        assert isinstance(c[k], int) and not isinstance(c[k], bool), f"{k} = {c[k]!r}"


def test_a_mixed_run_reports_both_paths_separately(monkeypatch):
    """CATCHES a counter that reports only the LAST branch taken, and a shared counter
    that adds the two paths together. One CPU call (torch, by precondition) and one MPS
    call (fused) in the same accounting window must appear as one of each."""
    from metal_gauss.train import (geometry_terms, loss_path_counters,
                                   reset_loss_path_counters)
    if not torch.backends.mps.is_available():
        pytest.skip("needs MPS")
    monkeypatch.delenv("MG_TORCH_LOSS", raising=False)
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    reset_loss_path_counters()
    geometry_terms(_args(), *_mapped("cpu"))
    geometry_terms(_args(), *_mapped("mps"))
    c = loss_path_counters()
    assert c["torch_calls"] == 1 and c["fused_calls"] == 1, c
    assert c["dn_ungated_calls"] == 2 and c["dn_gated_calls"] == 0, c


def _mapped(dev):
    n_sum, z, a, gt_d, gt_n = _tiny_maps(dev)
    return [n_sum, z], a, _K, gt_d, gt_n, None


# --------------------------------------------------------------- absence is not a pass

def test_the_block_is_always_present_and_all_zero_means_no_geometry_ran():
    """CATCHES a block that is omitted, or `None`, when nothing was counted. A harness
    asserting `report["observed"]["loss_path"]["torch_calls"] > 0` must be able to
    distinguish "the geometry recipe never ran" -- a failure on every Task 20 arm -- from
    "ran ungated", and it cannot if the key is missing or the value is null."""
    from metal_gauss.train import _run_report, reset_loss_path_counters
    reset_loss_path_counters()
    r = _run_report(argparse.Namespace(steps=10, budget=7, seed=0), [], 1.0, 7)
    lp = r["observed"]["loss_path"]
    assert set(lp) == set(KEYS), lp
    assert all(lp[k] == 0 for k in KEYS), lp
    assert all(isinstance(lp[k], int) for k in KEYS), lp
    assert r["observed"]["schema"] == 1


def test_the_call_total_invariant_holds(monkeypatch):
    """CATCHES a miscount: every call to `geometry_terms` that returns must land in
    exactly one path bucket AND exactly one dn bucket, so
    `fused + torch == gated + ungated + skipped`. A double bump, or a torch bump with no
    dn classification, breaks it. This is the one arithmetic check a harness can run
    without knowing the schedule."""
    from metal_gauss.train import (geometry_terms, loss_path_counters,
                                   reset_loss_path_counters)
    monkeypatch.setenv("MG_TORCH_LOSS", "1")
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    reset_loss_path_counters()
    n_sum, z, a, gt_d, gt_n = _tiny_maps("cpu")
    geometry_terms(_args(), [n_sum, z], a, _K, gt_d, gt_n, None)
    geometry_terms(_args(depth_normal_weight=0.0), [n_sum, z], a, _K, gt_d, gt_n, None)
    monkeypatch.setenv("MG_DN_GATE_NEIGHBOURS", "1")
    geometry_terms(_args(), [n_sum, z], a, _K, gt_d, gt_n, None)
    c = loss_path_counters()
    assert c["fused_calls"] + c["torch_calls"] == 3
    assert c["dn_gated_calls"] + c["dn_ungated_calls"] + c["dn_skipped_calls"] == 3
    assert (c["dn_gated_calls"], c["dn_ungated_calls"], c["dn_skipped_calls"]) == (1, 1, 1), c


@mps
def test_the_refused_fused_plus_gate_combination_counts_nothing(monkeypatch):
    """CATCHES a bump placed BEFORE the refusal. `geometry_terms` raises rather than
    running the fused kernel under a gated label; a counter incremented above that raise
    would leave a phantom fused call behind in a report written by an outer `except`."""
    from metal_gauss.train import (geometry_terms, loss_path_counters,
                                   reset_loss_path_counters)
    monkeypatch.delenv("MG_TORCH_LOSS", raising=False)
    monkeypatch.setenv("MG_DN_GATE_NEIGHBOURS", "1")
    reset_loss_path_counters()
    with pytest.raises(RuntimeError, match="MG_TORCH_LOSS"):
        geometry_terms(_args(), *_mapped("mps"))
    c = loss_path_counters()
    assert all(c[k] == 0 for k in KEYS), f"a refused call was counted as a run: {c}"


# ------------------------------------------------------------------ through train()

def _scene():
    from test_train_recipe import _synthetic_scene
    return _synthetic_scene()


def _train_args(**over):
    from test_train_recipe import _args as recipe_args
    return recipe_args(**over)


@mps
def test_a_gated_torch_arm_reports_every_call_torch_and_gated(monkeypatch):
    """THE ARM THIS EXISTS FOR. Task 20's treatment: `MG_TORCH_LOSS=1` +
    `MG_DN_GATE_NEIGHBOURS=1`. Every step's `geometry_terms` call must be recorded as
    torch and as gated, in the report the harness reads.

    CATCHES the treatment reaching the report but not the loss (or the reverse), and any
    per-step drop-out -- the counts are pinned to the step count, not merely to > 0."""
    from metal_gauss import train as T
    monkeypatch.setenv("MG_TORCH_LOSS", "1")
    monkeypatch.setenv("MG_DN_GATE_NEIGHBOURS", "1")
    out = T.train(_train_args(steps=4, eval_every=4, flatten_loss_weight=1.0,
                              depth_loss_weight=1.0, normal_loss_weight=0.2,
                              depth_normal_weight=0.05), scene=_scene())
    lp = out["observed"]["loss_path"]
    assert lp["torch_calls"] == 4 and lp["fused_calls"] == 0, lp
    assert lp["dn_gated_calls"] == 4 and lp["dn_ungated_calls"] == 0, lp


@mps
def test_an_unforced_arm_reports_the_fused_kernel_and_an_ungated_dn_term(monkeypatch):
    """The floor arm's mirror image, and the reason the pair is gradeable at all: with
    neither variable set the fused kernel runs and the dn term is ungated.

    CATCHES a counter hard-wired to one answer -- an implementation that always says
    "torch, gated" passes the test above and fails here."""
    from metal_gauss import train as T
    monkeypatch.delenv("MG_TORCH_LOSS", raising=False)
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    out = T.train(_train_args(steps=4, eval_every=4, flatten_loss_weight=1.0,
                              depth_loss_weight=1.0, normal_loss_weight=0.2,
                              depth_normal_weight=0.05), scene=_scene())
    lp = out["observed"]["loss_path"]
    assert lp["fused_calls"] == 4 and lp["torch_calls"] == 0, lp
    assert lp["dn_ungated_calls"] == 4 and lp["dn_gated_calls"] == 0, lp


@mps
def test_a_run_with_no_geometry_terms_is_distinguishable_from_an_ungated_run(monkeypatch):
    """ASSERT AGAINST THE IMPOSSIBLE VALUE. A geometry-recipe arm whose geometry never
    ran is a failed arm, and it must not be silently gradeable as an ungated floor.
    Photometric-only training must leave every counter at zero while the ungated arm
    above reads 4 -- absence and "ran ungated" are different reports.

    CATCHES a default that fills the block in from the environment when nothing ran, and
    a counter bumped once per STEP rather than once per geometry call."""
    from metal_gauss import train as T
    monkeypatch.delenv("MG_TORCH_LOSS", raising=False)
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    out = T.train(_train_args(steps=4, eval_every=4, flatten_loss_weight=1.0),
                  scene=_scene())
    lp = out["observed"]["loss_path"]
    assert all(lp[k] == 0 for k in KEYS), f"no geometry term ran, yet: {lp}"


@mps
def test_counters_do_not_leak_from_one_run_into_the_next(monkeypatch):
    """CATCHES accumulation across runs in one process -- which is how a harness that
    sweeps arms in-process, or this very test suite, would read the previous arm's
    chain as this one's. The second run's report must show its own four calls, not
    eight."""
    from metal_gauss import train as T
    monkeypatch.setenv("MG_TORCH_LOSS", "1")
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    a = _train_args(steps=4, eval_every=4, flatten_loss_weight=1.0,
                    depth_loss_weight=1.0, normal_loss_weight=0.2,
                    depth_normal_weight=0.05)
    T.train(a, scene=_scene())
    out = T.train(a, scene=_scene())
    lp = out["observed"]["loss_path"]
    assert lp["torch_calls"] == 4, f"the previous run's calls are still counted: {lp}"


# ---------------------------------------------------- the counters move no number

# `geometry_terms` on the fixture above, captured on e14fa50 -- the commit BEFORE the
# counters existed -- as hex so the comparison is bit-exact rather than approximate.
# Verified deterministic across two separate processes on that commit before it was
# recorded. The trainer as a whole is NOT bit-reproducible (rasterize_backward's atomics;
# `train`'s own docstring says so, and a 40-step run differs from itself in the 5th digit
# of PSNR), so `geometry_terms` -- the function that was actually edited -- is the level
# at which this can be proven rather than asserted.
_GOLDEN_E14FA50 = {
    "cpu_torch_ungated": {"depth": "0x1.fa91d80000000p-7",
                          "depth_normal": "0x1.f874a60000000p-1",
                          "normal": "0x1.4796ba0000000p-1"},
    "cpu_torch_gated": {"depth": "0x1.fa91d80000000p-7",
                        "depth_normal": "0x1.f9435a0000000p-1",
                        "normal": "0x1.4796ba0000000p-1"},
    "mps_fused": {"depth": "0x1.fa91dc0000000p-7",
                  "depth_normal": "0x1.f874a20000000p-1",
                  "normal": "0x1.4796b80000000p-1"},
    "mps_torch_forced": {"depth": "0x1.fa91d80000000p-7",
                         "depth_normal": "0x1.f874a20000000p-1",
                         "normal": "0x1.4796b80000000p-1"},
    "mps_torch_forced_gated": {"depth": "0x1.fa91d80000000p-7",
                               "depth_normal": "0x1.f9435a0000000p-1",
                               "normal": "0x1.4796b80000000p-1"},
    "cpu_dn_only": {"depth_normal": "0x1.f874a60000000p-1"},
}

_CASES = [
    ("cpu_torch_ungated", "cpu", {}, {}),
    ("cpu_torch_gated", "cpu", {"MG_DN_GATE_NEIGHBOURS": "1"}, {}),
    ("mps_fused", "mps", {}, {}),
    ("mps_torch_forced", "mps", {"MG_TORCH_LOSS": "1"}, {}),
    ("mps_torch_forced_gated", "mps",
     {"MG_TORCH_LOSS": "1", "MG_DN_GATE_NEIGHBOURS": "1"}, {}),
    ("cpu_dn_only", "cpu", {}, dict(depth_loss_weight=0.0, normal_loss_weight=0.0)),
]


@mps
@pytest.mark.parametrize("name,dev,env,kw", _CASES)
def test_counters_change_no_loss_value_against_the_pre_change_commit(
        name, dev, env, kw, monkeypatch):
    """PROOF, not assertion, that the accounting is inert. Six configurations spanning
    both branches, both gate settings and the dn-only case, compared bit-exactly against
    values captured on e14fa50.

    CATCHES any edit that moved a number while adding the counters -- most plausibly
    hoisting `_gate_dn_neighbours()` to a different point, or reordering the branch. The
    fixture discriminates: `cpu_torch_ungated` and `cpu_torch_gated` differ in
    `depth_normal`, and `mps_fused` differs from `mps_torch_forced`, so a case that
    silently took the wrong path would fail rather than coincide.
    """
    from metal_gauss.train import geometry_terms
    for k in ("MG_TORCH_LOSS", "MG_DN_GATE_NEIGHBOURS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    n_sum, z, a, gt_d, gt_n = _tiny_maps(dev)
    t = geometry_terms(_args(**kw), [n_sum, z], a, _K, gt_d, gt_n, None)
    got = {k: float(v).hex() for k, v in t.items()}
    assert got == _GOLDEN_E14FA50[name], f"{name}: {got} != {_GOLDEN_E14FA50[name]}"


def test_the_golden_fixture_can_tell_the_paths_apart():
    """A golden that could not separate the configurations would pin nothing. Asserts the
    discriminating power of the fixture above rather than trusting it: the gate must move
    `depth_normal` and only `depth_normal`, and the fused kernel must differ from the
    torch chain, or the parametrised test would pass under a wrong branch."""
    g = _GOLDEN_E14FA50
    assert g["cpu_torch_ungated"]["depth_normal"] != g["cpu_torch_gated"]["depth_normal"]
    assert g["cpu_torch_ungated"]["depth"] == g["cpu_torch_gated"]["depth"]
    assert g["cpu_torch_ungated"]["normal"] == g["cpu_torch_gated"]["normal"]
    assert g["mps_fused"]["depth"] != g["mps_torch_forced"]["depth"]
    assert g["mps_torch_forced"]["depth_normal"] != g["mps_torch_forced_gated"]["depth_normal"]
