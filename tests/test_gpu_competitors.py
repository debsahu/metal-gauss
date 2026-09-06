"""`require_gpu_exclusive()` could not see metal-gauss's own processes.

`gpu_competitors()` matched the executable NAME against
("spirula", "brush_app", "brush", "msplat-train"). Every metal-gauss process --
the trainer, pytest, every bench script -- runs as `python`, so the guard
detected COMPETITOR trainers and was blind to our own. It returned "GPU is
exclusive" while another agent's GPU test suite was saturating this machine, and
every timing this repo has taken beside a sibling metal-gauss process passed it.

The original docstring's reasoning is right and is preserved: a guard that can
never read zero is worse than no guard, because it trains you to ignore it. The
over-correction was matching `comm` ONLY, which makes every python invisible. The
fix keeps the name match AND adds an argv match for python processes, excluding
our own process tree so the old false positive cannot come back.

The parsing is a pure function so the discriminating cases can be tested without
spawning processes outside our own tree -- which is not something a test can do.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from bench.runner import _parse_competitors, gpu_competitors

SELF, PGID = 4242, 4200


def _ps(rows):
    """rows: (pid, ppid, pgid, comm, args)"""
    return "\n".join(f"{p} {pp} {pg} {c} {a}" for p, pp, pg, c, a in rows)


def _base():
    """Our own tree: an ancestor shell, us, and a child we spawned."""
    return [(9000, 1, 9000, "/bin/zsh", "-zsh"),
            (SELF, 9000, PGID, "/usr/bin/python3.12", "python -m bench.foo"),
            (4243, SELF, PGID, "/usr/bin/python3.12", "python -c import metal_gauss")]


def test_another_agents_python_test_suite_IS_a_competitor():
    """The exact process that defeated the old guard: another worktree's venv
    python running pytest over the GPU test files. Fails if the guard still
    matches on `comm` only."""
    rows = _base() + [(5555, 1, 5555, "/other/wt/.venv/bin/python3.12",
                       "/other/wt/.venv/bin/python -m pytest tests/test_metal.py -q")]
    hits = _parse_competitors(_ps(rows), SELF, PGID)
    assert any("5555" in h for h in hits), hits


def test_another_trainer_run_IS_a_competitor():
    rows = _base() + [(5556, 1, 5556, "/other/.venv/bin/python",
                       "python -m metal_gauss.train --colmap x --steps 30000")]
    assert any("5556" in h for h in _parse_competitors(_ps(rows), SELF, PGID))


def test_our_own_process_and_tree_are_NOT_competitors():
    """The failure the original guard was written to fix: a guard that can never
    read zero. Fails if self, the parent shell, or a child we spawned is counted."""
    assert _parse_competitors(_ps(_base()), SELF, PGID) == []


def test_an_unrelated_python_is_NOT_a_competitor():
    """The discrimination that makes the argv match usable at all. A machine runs
    many pythons; firing on all of them would recreate the always-nonzero guard by
    a different route."""
    rows = _base() + [(6000, 1, 6000, "/usr/bin/python3", "python -m http.server 8000"),
                      (6001, 1, 6001, "/usr/bin/python3", "python /Users/x/unrelated.py")]
    assert _parse_competitors(_ps(rows), SELF, PGID) == []


def test_a_shell_whose_ARGV_mentions_brush_is_NOT_a_competitor():
    """THE ORIGINAL FALSE POSITIVE, preserved as a regression test. A naive
    `pgrep -f "spirula|msplat|brush"` matched the wrapper shell whose command line
    contained the pattern, and reported four competitors during a run that had
    one."""
    rows = _base() + [(7000, 1, 7000, "/bin/bash", "bash run_brush_and_spirula.sh"),
                      (7001, 1, 7001, "/bin/bash", "bash -c 'msplat-train --x'")]
    assert _parse_competitors(_ps(rows), SELF, PGID) == []


def test_a_real_competitor_binary_is_STILL_detected_by_name():
    """The old behaviour must survive the fix -- these do not run as python."""
    rows = _base() + [(8000, 1, 8000, "/usr/local/bin/brush", "brush train --x"),
                      (8001, 1, 8001, "/opt/spirula", "spirula --y")]
    hits = _parse_competitors(_ps(rows), SELF, PGID)
    assert len(hits) == 2, hits


def test_the_fixture_can_tell_the_old_rule_from_the_new_one():
    """Discriminating power, asserted. Under the OLD rule -- match `comm` basename
    against the four names -- the pytest and trainer rows below score ZERO hits,
    which is precisely the blindness this file exists to fix. If that stops being
    true the tests above have stopped testing the fix."""
    rows = [(5555, 1, 5555, "/other/.venv/bin/python3.12", "python -m pytest tests/test_metal.py"),
            (5556, 1, 5556, "/other/.venv/bin/python", "python -m metal_gauss.train --steps 3")]
    old_names = ("spirula", "brush_app", "brush", "msplat-train")
    assert not any(c.rsplit("/", 1)[-1] in old_names for _, _, _, c, _ in rows)
    assert len(_parse_competitors(_ps(_base() + rows), SELF, PGID)) == 2


def test_a_SIBLING_process_IS_a_competitor():
    """The case that defeated the first version of the fix, and the realistic one:
    another agent's run, launched from a neighbouring shell under a shared parent.

    Seeding the descendant walk with self AND its ancestors excludes every sibling
    subtree, so a live out-of-tree decoy went undetected. Descendants are expanded
    from SELF only; ancestors are excluded but contribute no descendants."""
    rows = _base() + [
        (9500, 9000, 9500, "/bin/zsh", "-zsh"),                       # sibling shell
        (9501, 9500, 9500, "/other/.venv/bin/python",                 # its GPU job
         "python -m metal_gauss.train --steps 30000")]
    hits = _parse_competitors(_ps(rows), SELF, PGID)
    assert any("9501" in h for h in hits), hits
    # ...and the sibling SHELL itself is still not one, since it is not python
    assert not any("9500" in h for h in hits), hits


def test_comm_is_truncated_by_ps_and_the_parser_survives_it():
    """macOS `ps -o comm=` TRUNCATES TO 16 CHARACTERS. Every venv python therefore
    reads as a path fragment like "/private/tmp/cla", and no basename test on
    `comm` can recognise an interpreter.

    This is the case the hand-written fixtures above could not produce, because I
    wrote their `comm` fields out in full. The suite was green while a live
    out-of-tree decoy went undetected. The parser now takes the executable from
    the first token of argv, which `ps` does not truncate.
    """
    truncated = "/private/tmp/cla"          # exactly what ps emits, 16 chars
    assert len(truncated) == 16
    rows = _base() + [(5557, 1, 5557, truncated,
                       "/private/tmp/x/wt/.venv/bin/python -m pytest tests/test_metal.py -q")]
    hits = _parse_competitors(_ps(rows), SELF, PGID)
    assert any("5557" in h for h in hits), hits
    # ...and a truncated NON-python must still not match
    rows2 = _base() + [(5558, 1, 5558, truncated, "/private/tmp/x/some_helper --brush")]
    assert _parse_competitors(_ps(rows2), SELF, PGID) == []


def test_a_real_detached_process_is_detected_in_REAL_ps_output():
    """End to end on the format `ps` actually emits, not a transcription of it.

    A test cannot spawn a process outside its own tree, so the tree exclusion is
    sidestepped by passing an unrelated `self_pid`. Everything else -- the ps
    invocation, the truncation, the field layout -- is real. Without this, the
    only end-to-end coverage was a NEGATIVE assertion, which cannot catch a
    parser that never matches anything.
    """
    import subprocess as sp
    child = sp.Popen([sys.executable, "-c",
                      "import time; time.sleep(20)  # -m pytest metal_gauss decoy"],
                     start_new_session=True,
                     stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    try:
        time.sleep(0.8)
        ps_text = sp.run(["ps", "-eo", "pid=,ppid=,pgid=,comm=,args="],
                         capture_output=True, text=True).stdout
        # self_pid must be a pid that DOES NOT EXIST. Passing 1 excludes the whole
        # machine, because virtually everything descends from launchd -- the first
        # version of this test did exactly that and read zero hits on a live decoy.
        hits = _parse_competitors(ps_text, self_pid=999_999, self_pgid=-12345)
        assert any(f"pid {child.pid})" in h for h in hits), \
            f"real detached python not detected; {len(hits)} hits"
        # control: with the child inside the excluded tree it must vanish, or the
        # assertion above would be satisfied by a parser that matches everything.
        assert not any(f"pid {child.pid})" in h
                       for h in _parse_competitors(ps_text, self_pid=child.pid,
                                                   self_pgid=-12345))
    finally:
        child.kill(); child.wait()


def test_a_child_we_spawn_for_real_is_not_reported():
    """The pure-function tests use synthetic `ps` text, so one end-to-end case
    pins that the real `ps` parsing agrees with it. A python child of ours, with
    a matching argv, must not make our own guard fire."""
    child = subprocess.Popen(
        [sys.executable, "-c", "import time,sys; sys.argv.append('metal_gauss.train'); time.sleep(6)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.6)
        assert not any(str(child.pid) in h for h in gpu_competitors())
    finally:
        child.kill(); child.wait()
