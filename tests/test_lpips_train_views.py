"""Task 21 step 2: the split selection, which IS the measurement.

If `--split train` returned held-out views, train-view LPIPS would equal
held-out LPIPS by construction, the pre-registered reading "within 0.03 =>
representation, not generalisation" would fire, and no output would look wrong.
That is the single highest-value assertion in this task.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parents[1] / "scripts" / "lpips_train_views.py"
_spec = importlib.util.spec_from_file_location("lpips_train_views", _p)
LTV = importlib.util.module_from_spec(_spec)
sys.modules["lpips_train_views"] = LTV
_spec.loader.exec_module(LTV)


def _scene(n=40, every=8):
    views = [types.SimpleNamespace(name=f"v{i:03d}") for i in range(n)]
    heldout = views[::every]
    hn = {v.name for v in heldout}
    return types.SimpleNamespace(train=[v for v in views if v.name not in hn],
                                 heldout=heldout)


def test_train_selection_is_disjoint_from_heldout():
    s = _scene()
    picked = LTV.select_views(s, "train", 24)
    assert {v.name for v in picked} & {v.name for v in s.heldout} == set()
    assert {v.name for v in picked} <= {v.name for v in s.train}


def test_the_scene_fixture_can_actually_detect_a_leak():
    """Discriminating power, asserted: the two splits must be non-empty and
    disjoint, or the test above passes against any implementation."""
    s = _scene()
    assert len(s.train) == 35 and len(s.heldout) == 5
    assert {v.name for v in s.train} & {v.name for v in s.heldout} == set()
    leaked = LTV.select_views(s, "heldout", 5)
    assert {v.name for v in leaked} & {v.name for v in s.train} == set()
    assert {v.name for v in leaked} == {v.name for v in s.heldout}


def test_selection_count_is_exact_and_evenly_spaced():
    """The first n views of a walkthrough are one corner of one room;
    `evaluate`'s own docstring records that biasing by up to ~0.5 dB."""
    s = _scene(n=80)
    picked = LTV.select_views(s, "train", 24)
    assert len(picked) == 24
    assert picked[0].name == s.train[0].name
    assert picked[-1].name == s.train[-1].name
    idx = [s.train.index(v) for v in picked]
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    assert max(gaps) - min(gaps) <= 1, gaps


def test_selection_returns_all_when_fewer_than_requested():
    s = _scene(n=12)
    assert len(LTV.select_views(s, "heldout", 24)) == len(s.heldout)


def test_selection_refuses_a_nonsense_count():
    with pytest.raises(ValueError, match="n-views"):
        LTV.select_views(_scene(), "train", 0)


def test_selection_refuses_an_empty_split():
    s = types.SimpleNamespace(train=[], heldout=[types.SimpleNamespace(name="a")])
    with pytest.raises(ValueError, match="empty"):
        LTV.select_views(s, "train", 1)
