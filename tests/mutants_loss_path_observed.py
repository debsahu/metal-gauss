#!/usr/bin/env python3
"""Mutation battery for tests/test_loss_path_observed.py.

Every mutant is (a) PROVEN to change behaviour by a probe that CALLS the code -- never by
reading source text, which is how a previous battery in this project reported two wiring
mutants as behaviour-unchanged -- and then (b) required to kill a NAMED test. Asserting on
a failure count would be satisfied by any failure, including one the mutant caused by
accident.

`PYTHONDONTWRITEBYTECODE=1` is set here rather than left to the caller: with stale .pyc
files a battery fails toward FALSE SURVIVED.

    .venv/bin/python tests/mutants_loss_path_observed.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "metal_gauss" / "train.py"
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

# The probe runs in a fresh process and prints a signature: the observed counters for four
# configurations plus the loss values themselves, so a mutant that moves a NUMBER and one
# that moves a COUNT are both visible.
PROBE = r'''
import argparse, json, os, sys
import torch
sys.path.insert(0, %r)
from metal_gauss.train import (geometry_terms, loss_path_counters,
                               reset_loss_path_counters, _run_report)

def maps(dev):
    g = torch.Generator().manual_seed(3)
    h, w = 12, 14
    a = torch.ones(h, w); a[:, 5] = 0.1
    z = ((2.0 + torch.rand(h, w, generator=g) * 0.2) * a)[..., None].expand(h, w, 3).contiguous()
    n = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    gt_d = 2.0 + torch.rand(h, w, generator=g) * 0.2
    gt_n = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    return [x.to(dev) for x in (a[..., None] * n, z, a, gt_d, gt_n)]

def args(**kw):
    d = dict(depth_loss_weight=1.0, normal_loss_weight=0.2, depth_normal_weight=0.05,
             depth_loss_space="disparity", depth_source="center")
    d.update(kw); return argparse.Namespace(**d)

K = torch.tensor([[300.0, 0, 7.0], [0, 300.0, 6.0], [0, 0, 1.0]])
sig = {}
cases = [("cpu_no_env", "cpu", {}, {}),
         ("cpu_forced_gated", "cpu", {"MG_TORCH_LOSS": "1", "MG_DN_GATE_NEIGHBOURS": "1"}, {}),
         ("cpu_gate_no_term", "cpu", {"MG_DN_GATE_NEIGHBOURS": "1"}, dict(depth_normal_weight=0.0)),
         ("mps_no_env", "mps", {}, {})]
for name, dev, env, kw in cases:
    for k in ("MG_TORCH_LOSS", "MG_DN_GATE_NEIGHBOURS"):
        os.environ.pop(k, None)
    os.environ.update(env)
    reset_loss_path_counters()
    n_sum, z, a, gt_d, gt_n = maps(dev)
    try:
        t = geometry_terms(args(**kw), [n_sum, z], a, K, gt_d, gt_n, None)
        vals = {k2: float(v).hex() for k2, v in sorted(t.items())}
    except Exception as e:
        vals = {"raised": type(e).__name__}
    sig[name] = {"counters": loss_path_counters(), "values": vals}
for k in ("MG_TORCH_LOSS", "MG_DN_GATE_NEIGHBOURS"):
    os.environ.pop(k, None)
# the report block for a run in which NOTHING was counted
reset_loss_path_counters()
sig["empty_report"] = _run_report(argparse.Namespace(steps=1, budget=1, seed=0), [], 1.0, 1)["observed"]
# two accounting windows without a reset between them, which is what `train` must prevent
reset_loss_path_counters()
os.environ["MG_TORCH_LOSS"] = "1"
n_sum, z, a, gt_d, gt_n = maps("cpu")
geometry_terms(args(), [n_sum, z], a, K, gt_d, gt_n, None)
first = loss_path_counters()
geometry_terms(args(), [n_sum, z], a, K, gt_d, gt_n, None)
sig["two_calls"] = {"after_one": first, "after_two": loss_path_counters()}

# TWO `train()` RUNS IN ONE PROCESS. Without this leg the probe cannot see a mutant that
# deletes `train`'s reset -- it would be reported as "no behaviour change", which is a
# defective probe rather than a surviving mutant. Nothing above calls train().
if torch.backends.mps.is_available():
    sys.path.insert(0, os.path.join(%r, "tests"))
    from test_train_recipe import _args as recipe_args, _synthetic_scene
    from metal_gauss import train as T
    os.environ["MG_TORCH_LOSS"] = "1"
    ta = recipe_args(steps=2, eval_every=2, flatten_loss_weight=1.0,
                     depth_loss_weight=1.0, normal_loss_weight=0.2,
                     depth_normal_weight=0.05)
    sig["train_run1"] = T.train(ta, scene=_synthetic_scene())["observed"]
    sig["train_run2"] = T.train(ta, scene=_synthetic_scene())["observed"]
print(json.dumps(sig, sort_keys=True))
''' % (str(ROOT), str(ROOT))


def probe() -> str:
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", PROBE],
                       capture_output=True, text=True, env=ENV, cwd=str(ROOT))
    if r.returncode != 0:
        return "PROBE-CRASH:" + r.stderr.strip()[-400:]
    return r.stdout.strip()


def failing_test_names(node_ids) -> set:
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "--tb=no",
                        "-p", "no:cacheprovider", *node_ids],
                       capture_output=True, text=True, env=ENV, cwd=str(ROOT))
    return set(re.findall(r"^FAILED [^:]+::([\w\[\]\-.,]+)", r.stdout, re.M))


def sub(text, old, new, count=1):
    assert text.count(old) >= 1, f"mutation anchor not found:\n{old}"
    return text.replace(old, new, count)


# (name, mutate(src) -> src, the test whose NAME must fail)
MUTANTS = [
    ("record_the_env_not_the_branch__path",
     lambda s: sub(s, '    _observe("torch_calls")\n',
                   '    _observe("fused_calls" if _use_fused_loss() else "torch_calls")\n'),
     "test_report_says_torch_ran_when_the_environment_asked_for_fused"),

    ("record_the_env_not_the_branch__gate",
     lambda s: sub(s, '        _observe("dn_skipped_calls")\n',
                   '        _observe("dn_gated_calls" if _gate_dn_neighbours() '
                   'else "dn_skipped_calls")\n'),
     "test_report_says_dn_was_skipped_when_the_gate_was_requested_but_no_dn_term_ran"),

    ("boolean_instead_of_counter",
     lambda s: sub(s, "        _loss_path_counts[k] += 1\n",
                   "        _loss_path_counts[k] = 1\n"),
     "test_counters_count_calls_rather_than_recording_a_boolean"),

    ("bump_before_the_refusal",
     lambda s: sub(s, "        if _gate_dn_neighbours():\n",
                   '        _observe("fused_calls", "dn_ungated_calls")\n'
                   "        if _gate_dn_neighbours():\n"),
     "test_the_refused_fused_plus_gate_combination_counts_nothing"),

    ("no_reset_between_runs",
     lambda s: sub(s, "    reset_loss_path_counters()     # likewise", "    #"),
     "test_counters_do_not_leak_from_one_run_into_the_next"),

    ("omit_the_zero_counters_from_the_report",
     lambda s: sub(s, '"loss_path": {k: int(lp.get(k, 0)) for k in LOSS_PATH_KEYS}',
                   '"loss_path": {k: int(lp[k]) for k in LOSS_PATH_KEYS if lp.get(k)}'),
     "test_the_block_is_always_present_and_all_zero_means_no_geometry_ran"),

    ("skipped_counted_as_ungated",
     lambda s: sub(s, '        _observe("dn_skipped_calls")\n',
                   '        _observe("dn_ungated_calls")\n'),
     "test_the_call_total_invariant_holds"),

    ("gate_classification_inverted",
     lambda s: sub(s, '_observe("dn_gated_calls" if gate else "dn_ungated_calls")',
                   '_observe("dn_ungated_calls" if gate else "dn_gated_calls")'),
     "test_a_gated_torch_arm_reports_every_call_torch_and_gated"),

    ("hardwired_to_torch_and_gated",
     lambda s: sub(s, '        _observe("fused_calls", "dn_ungated_calls")\n',
                   '        _observe("torch_calls", "dn_gated_calls")\n'),
     "test_an_unforced_arm_reports_the_fused_kernel_and_an_ungated_dn_term"),

    # Not a counter mutant: it moves a NUMBER. It is here to show the golden fixture
    # would have caught the counters changing the loss, rather than that being asserted.
    ("the_edit_moved_a_number",
     lambda s: sub(s, "            gate_neighbours=gate)", "            gate_neighbours=True)"),
     "test_counters_change_no_loss_value_against_the_pre_change_commit[cpu_torch_ungated-cpu-env0-kw0]"),
]

NODES = ["tests/test_loss_path_observed.py"]


def main() -> int:
    src = TRAIN.read_text()
    base_sig = probe()
    assert not base_sig.startswith("PROBE-CRASH"), base_sig
    base_fail = failing_test_names(NODES)
    assert not base_fail, f"the suite must be green before mutating; failing: {base_fail}"
    print(f"baseline: {len(NODES)} target(s) green, probe signature "
          f"{len(base_sig)} chars\n")

    results = []
    backup = Path(tempfile.mkdtemp()) / "train.py"
    shutil.copy2(TRAIN, backup)
    try:
        for name, mutate, must_fail in MUTANTS:
            TRAIN.write_text(mutate(src))
            sig = probe()
            changed = sig != base_sig
            fails = failing_test_names(NODES) if changed else set()
            killed = must_fail in fails
            results.append((name, changed, killed, must_fail, sorted(fails)[:3]))
            mark = "KILLED" if (changed and killed) else (
                "SURVIVED" if changed else "NO-BEHAVIOUR-CHANGE (not a mutant)")
            print(f"{mark:34s} {name}")
            if changed and not killed:
                print(f"    expected {must_fail} to fail; got {sorted(fails)}")
            if sig.startswith("PROBE-CRASH"):
                print("    probe crashed (still a behaviour change): " + sig[:200])
    finally:
        shutil.copy2(backup, TRAIN)

    assert TRAIN.read_text() == src, "restore failed -- train.py is NOT the original"
    after = failing_test_names(NODES)
    assert not after, f"suite not green after restore: {after}"
    n_ok = sum(1 for _, c, k, _, _ in results if c and k)
    print(f"\n{n_ok}/{len(results)} killed; restore verified by re-running the suite green")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
