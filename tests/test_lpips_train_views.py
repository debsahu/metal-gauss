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


def test_every_module_main_needs_actually_imports():
    """THE TEST THAT WAS MISSING. `main()` imported `bench.dn_neighbour_gate`,
    which lives on feat/dn-neighbour-gate and is NOT on main -- a dependency on
    an unmerged branch. Six renders died with ModuleNotFoundError one second
    apart, and nothing in the suite could see it because the tests exercised
    `select_views` and never `main()`. `--help` cannot see it either: the
    imports are deferred inside `main`.
    """
    import importlib
    for mod in ("bench.lpips_attr", "metal_gauss.dataset", "metal_gauss.train"):
        importlib.import_module(mod)
    from bench.lpips_attr import params_from_ply          # noqa: F401
    from metal_gauss.dataset import load_scene            # noqa: F401
    from metal_gauss.train import evaluate                # noqa: F401


def test_params_from_ply_round_trips_the_trainers_own_export(tmp_path):
    """PRE-ACTIVATION SPACE, pinned by a round trip rather than by copying the
    reader. `metal_gauss.io.load_ply` ACTIVATES opacity, scale and rotation, so
    using it here would silently pass a sigmoid'd opacity as a logit -- and a
    value at the rails never comes back."""
    import torch
    from bench.lpips_attr import params_from_ply
    from metal_gauss.train import export_ply
    torch.manual_seed(0)
    n = 17
    p = {"means": torch.randn(n, 3),
         "log_scales": torch.randn(n, 3) * 0.5 - 3.0,
         "quats": torch.randn(n, 4),
         "logit_opac": torch.randn(n) * 4.0,          # spans the sigmoid rails
         "sh_dc": torch.randn(n, 1, 3),
         "sh_rest": torch.randn(n, 15, 3)}
    out = tmp_path / "x.ply"
    export_ply({k: v.clone() for k, v in p.items()}, str(out))
    got, m = params_from_ply(str(out))
    assert m == n
    for k in p:
        assert torch.equal(got[k], p[k]), (k, (got[k] - p[k]).abs().max())


def test_params_from_ply_is_not_the_activating_reader(tmp_path):
    """Discriminating power: the fixture's opacities must be far enough onto the
    rails that an activating reader gives a visibly different answer."""
    import torch
    from bench.lpips_attr import params_from_ply
    from metal_gauss.train import export_ply
    torch.manual_seed(1)
    n = 9
    p = {"means": torch.randn(n, 3), "log_scales": torch.full((n, 3), -3.0),
         "quats": torch.randn(n, 4), "logit_opac": torch.tensor([-8.0, 8.0] * 4 + [0.0]),
         "sh_dc": torch.zeros(n, 1, 3), "sh_rest": torch.zeros(n, 15, 3)}
    out = tmp_path / "y.ply"
    export_ply({k: v.clone() for k, v in p.items()}, str(out))
    got, _ = params_from_ply(str(out))
    assert torch.equal(got["logit_opac"], p["logit_opac"])
    assert (torch.sigmoid(p["logit_opac"]) - p["logit_opac"]).abs().max() > 5.0
