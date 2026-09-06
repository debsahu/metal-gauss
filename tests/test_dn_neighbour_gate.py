"""Task 20: the depth-normal term's NEIGHBOUR gating, behind a flag on the torch path.

`keep` and `alpha > 0.5` gate the BASE pixel only. `n_d(v,u)` is differentiated from
`depth_img` over the stencil {(v,u), (v,u+1), (v+1,u)} (`normals_from_depth`), so a
pixel's normal can be built from a dropped or uncovered neighbour's depth. That is what
both this trainer and Brush compute today; `gate_neighbours=True` is the candidate
alternative, measured in bench/dn_neighbour_gate.py and NOT the default.

Every test here says what it would catch, and each was confirmed to FAIL against a wrong
implementation -- 11 mutants, 11 killed, each asserted on the failing test's NAME rather
than on a failure count. The battery and its results are in the Task 20 step 2 commit
message and in research/metal-gauss.md section 13; it needs PYTHONDONTWRITEBYTECODE=1 to
be trustworthy (section 12.5 -- without it the battery fails toward FALSE SURVIVED).
"""
import argparse

import pytest
import torch

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


# ----------------------------------------------------------------- the stencil helper

def _single_hole(h, w, vh, uh):
    c = torch.ones(h, w, dtype=torch.bool)
    c[vh, uh] = False
    return c


def test_dn_neighbours_covered_uses_the_normals_from_depth_stencil():
    """CATCHES: the wrong stencil. `n_d(v,u)` reads (v,u+1) and (v+1,u); an implementation
    that reads (v,u-1)/(v-1,u), the diagonal (v+1,u+1), or folds the base pixel back in
    gates a DIFFERENT set of base pixels for the same hole. Each hole below is placed so
    the three candidate stencils disagree.
    """
    from metal_gauss.geometry_loss import dn_neighbours_covered

    # hole at (0,1): the RIGHT neighbour of (0,0) only.
    g = ~dn_neighbours_covered(_single_hole(3, 3, 0, 1))
    assert g[0, 0] and not g[0, 1] and not g[1, 0] and not g[1, 1]
    # hole at (1,0): the DOWN neighbour of (0,0) only.
    g = ~dn_neighbours_covered(_single_hole(3, 3, 1, 0))
    assert g[0, 0] and not g[0, 1] and not g[1, 0] and not g[1, 1]
    # hole at (1,1): down-neighbour of (1,0)... no: right-neighbour of (1,0) and
    # down-neighbour of (0,1). NOT a neighbour of (0,0) under this stencil, which is
    # exactly what a diagonal-reading implementation would get wrong.
    g = ~dn_neighbours_covered(_single_hole(3, 3, 1, 1))
    assert g[0, 1] and g[1, 0] and not g[0, 0]


def _wrong_stencils():
    """Plausible wrong neighbour rules, written out so the fixture can be shown to
    separate them. Each CHANGES BEHAVIOUR -- a candidate that does not is not a mutant and
    is not listed. Notably absent: "AND the base pixel back in", which is behaviourally
    identical inside `depth_normal_loss` because `valid` already carries `covered` at the
    base, so it would prove nothing.
    """
    def backward_stencil(c):        # reads (v,u-1) and (v-1,u)
        o = torch.zeros_like(c)
        o[1:, 1:] = c[1:, :-1] & c[:-1, 1:]
        return o

    def diagonal(c):                # reads (v+1,u+1) only
        o = torch.zeros_like(c)
        o[:-1, :-1] = c[1:, 1:]
        return o

    def either_not_both(c):         # OR where AND belongs
        o = torch.zeros_like(c)
        o[:-1, :-1] = c[:-1, 1:] | c[1:, :-1]
        return o

    def right_only(c):              # forgets the row below
        o = torch.zeros_like(c)
        o[:-1, :-1] = c[:-1, 1:]
        return o

    return {"backward_stencil": backward_stencil, "diagonal": diagonal,
            "either_not_both": either_not_both, "right_only": right_only}


def test_dn_neighbours_covered_fixture_separates_every_wrong_stencil():
    """THE FIXTURE'S OWN DISCRIMINATING POWER, asserted so a future wrong stencil cannot
    quietly re-pin these numbers. For each wrong rule there must exist a hole position in
    the fixture where it disagrees with the shipped one -- otherwise the cases above are
    passing for free."""
    from metal_gauss.geometry_loss import dn_neighbours_covered
    holes = [(v, u) for v in range(3) for u in range(3)]
    for name, wrong in _wrong_stencils().items():
        assert any(not torch.equal(dn_neighbours_covered(_single_hole(3, 3, v, u)),
                                   wrong(_single_hole(3, 3, v, u)))
                   for v, u in holes), name


def test_dn_neighbours_covered_marks_the_last_row_and_column_ungated():
    """CATCHES a wrap-around or an off-by-one that lets the last row/column pass. Those
    base pixels have no (v,u+1)/(v+1,u), `normals_from_depth` emits exactly zero there,
    and the gate must not resurrect them."""
    from metal_gauss.geometry_loss import dn_neighbours_covered
    ok = dn_neighbours_covered(torch.ones(4, 5, dtype=torch.bool))
    assert not ok[-1].any() and not ok[:, -1].any()
    assert ok[:-1, :-1].all()


# --------------------------------------------------------------- the loss, gate binding

def _binding_case():
    """One base pixel that IS in the loss under the shipped rule and whose normal is built
    from an UNCOVERED neighbour's depth. A 2x3 frame: base (0,0) is covered, its right
    neighbour (0,1) is not.

    Contributions, ungated: (0,0) err 1.0, (0,2)... (0,2) is the last column, and (1,*) is
    the last row, so only (0,0) and (0,1) can ever be base pixels, and (0,1) is uncovered.
    So ungated = 1 pixel at err 1.0 -> 1.0; gated = 0 pixels -> 0.0 (the `clamp_min(1)`
    denominator, not NaN).
    """
    nd = torch.zeros(2, 3, 3)
    nr = torch.zeros(2, 3, 3)
    nd[0, 0] = torch.tensor([0.0, 0.0, -1.0])
    nr[0, 0] = torch.tensor([1.0, 0.0, 0.0])          # 90 deg -> 1 - 0 = 1.0
    alpha = torch.tensor([[1.0, 0.1, 1.0], [1.0, 1.0, 1.0]])
    return nd, nr, alpha


def test_gate_drops_a_base_pixel_whose_neighbour_is_uncovered():
    """CATCHES the gate being a no-op, or gating on the base pixel again (which would
    change nothing, since the base pixel is already gated). The fixture's discriminating
    power is asserted first: ungated and gated must DIFFER here, or the case is vacuous."""
    from metal_gauss.geometry_loss import depth_normal_loss
    nd, nr, alpha = _binding_case()
    ungated = depth_normal_loss(nd, nr, alpha).item()
    gated = depth_normal_loss(nd, nr, alpha, gate_neighbours=True).item()
    assert ungated != gated, "fixture does not separate the two rules"
    assert ungated == pytest.approx(1.0)
    assert gated == pytest.approx(0.0)


def test_gate_default_is_off_and_matches_the_shipped_signature():
    """CATCHES flipping the default. The shipped semantic is UNGATED; this task measures
    the alternative, it does not adopt it."""
    from metal_gauss.geometry_loss import depth_normal_loss
    nd, nr, alpha = _binding_case()
    assert (depth_normal_loss(nd, nr, alpha).item()
            == depth_normal_loss(nd, nr, alpha, gate_neighbours=False).item())
    assert depth_normal_loss(nd, nr, alpha).item() != depth_normal_loss(
        nd, nr, alpha, gate_neighbours=True).item()


def test_gate_is_exactly_a_noop_when_every_pixel_is_covered():
    """CATCHES a gate that drops the interior too (e.g. an inverted predicate, or an
    `any` where an `all` belongs). Built on a REAL `normals_from_depth` output so the
    last row/column are genuinely zero and cannot mask an error at the border."""
    from metal_gauss.geometry_loss import depth_normal_loss, normals_from_depth
    g = torch.Generator().manual_seed(7)
    depth = 2.0 + torch.rand(9, 11, generator=g)
    nd = normals_from_depth(depth, 300.0, 300.0, 5.0, 4.0)
    nr = torch.nn.functional.normalize(torch.randn(9, 11, 3, generator=g), dim=-1)
    alpha = torch.ones(9, 11)
    a = depth_normal_loss(nd, nr, alpha)
    b = depth_normal_loss(nd, nr, alpha, gate_neighbours=True)
    assert a.item() > 0.0                                  # not vacuous
    assert torch.equal(a, b)


def test_gate_covers_keep_as_well_as_alpha():
    """CATCHES a gate that reads alpha only. The caller passes `alpha * keep`, so a
    keep==0 neighbour is exactly an alpha==0 neighbour; this pins that the mask reaches
    the neighbour test through the same argument, since P-MASK is the only dataset where
    `keep` can fire at all."""
    from metal_gauss.geometry_loss import depth_normal_loss
    nd, nr, alpha = _binding_case()
    alpha = torch.ones(2, 3)
    keep = torch.ones(2, 3)
    keep[0, 1] = 0.0                                       # dropped, not uncovered
    ungated = depth_normal_loss(nd, nr, alpha * keep).item()
    gated = depth_normal_loss(nd, nr, alpha * keep, gate_neighbours=True).item()
    assert ungated == pytest.approx(1.0) and gated == pytest.approx(0.0)


# ------------------------------------------------------------------- the trainer wiring

def _args(**kw):
    d = dict(depth_loss_weight=1.0, normal_loss_weight=0.2, depth_normal_weight=0.05,
             depth_loss_space="disparity", depth_source="center")
    d.update(kw)
    return argparse.Namespace(**d)


def _tiny_maps(dev="cpu"):
    """(n_sum, z_img, alpha, gt_d, gt_n) with one uncovered column, on a real depth ramp
    so `normals_from_depth` produces a non-degenerate field."""
    g = torch.Generator().manual_seed(3)
    h, w = 12, 14
    a = torch.ones(h, w)
    a[:, 5] = 0.1                                          # an uncovered column
    z = (2.0 + torch.rand(h, w, generator=g) * 0.2) * a
    z = z[..., None].expand(h, w, 3).contiguous()
    n = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    n_sum = a[..., None] * n
    gt_d = 2.0 + torch.rand(h, w, generator=g) * 0.2
    gt_n = torch.nn.functional.normalize(torch.randn(h, w, 3, generator=g), dim=-1)
    return [x.to(dev) for x in (n_sum, z, a, gt_d, gt_n)]


def test_env_flag_reaches_the_torch_path_and_changes_only_the_dn_term(monkeypatch):
    """CATCHES the flag being read but not applied (the failure this repo keeps finding).
    Also pins the blast radius: depth and normal are untouched by a dn-only gate."""
    from metal_gauss.train import geometry_terms
    monkeypatch.setenv("MG_TORCH_LOSS", "1")
    n_sum, z, a, gt_d, gt_n = _tiny_maps()
    K = torch.tensor([[300.0, 0, 7.0], [0, 300.0, 6.0], [0, 0, 1.0]])
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    off = geometry_terms(_args(), [n_sum, z], a, K, gt_d, gt_n, None)
    monkeypatch.setenv("MG_DN_GATE_NEIGHBOURS", "1")
    on = geometry_terms(_args(), [n_sum, z], a, K, gt_d, gt_n, None)
    assert off["depth_normal"].item() != on["depth_normal"].item()
    assert off["depth"].item() == on["depth"].item()
    assert off["normal"].item() == on["normal"].item()


def test_gate_flag_is_off_by_default_end_to_end(monkeypatch):
    """CATCHES a default flip at the env-var layer: an unset variable, "0" and "false"
    must all mean the shipped semantic."""
    from metal_gauss.train import _gate_dn_neighbours
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    assert _gate_dn_neighbours() is False
    for v in ("0", "", "false", "False"):
        monkeypatch.setenv("MG_DN_GATE_NEIGHBOURS", v)
        assert _gate_dn_neighbours() is False
    for v in ("1", "true", "yes"):
        monkeypatch.setenv("MG_DN_GATE_NEIGHBOURS", v)
        assert _gate_dn_neighbours() is True


@mps
def test_fused_path_refuses_the_gate_rather_than_ignoring_it(monkeypatch):
    """THE ONE THAT MATTERS. The gate exists on the torch reference path only -- the
    fused Metal kernel does not implement it. A run that sets the flag and silently takes
    the fused path would report an UNGATED number under a gated label, which is exactly
    the class of result this project keeps having to retract. It must raise."""
    from metal_gauss.train import geometry_terms
    n_sum, z, a, gt_d, gt_n = _tiny_maps(dev="mps")
    K = torch.tensor([[300.0, 0, 7.0], [0, 300.0, 6.0], [0, 0, 1.0]])
    monkeypatch.delenv("MG_TORCH_LOSS", raising=False)          # fused path selected
    monkeypatch.setenv("MG_DN_GATE_NEIGHBOURS", "1")
    with pytest.raises(RuntimeError, match="MG_TORCH_LOSS"):
        geometry_terms(_args(), [n_sum, z], a, K, gt_d, gt_n, None)


@mps
def test_fused_path_is_untouched_when_the_gate_is_off(monkeypatch):
    """CATCHES the refusal above firing on every fused run. Without this the guard could
    be an unconditional raise and the suite would still be green on the test above."""
    from metal_gauss.train import geometry_terms
    n_sum, z, a, gt_d, gt_n = _tiny_maps(dev="mps")
    K = torch.tensor([[300.0, 0, 7.0], [0, 300.0, 6.0], [0, 0, 1.0]])
    monkeypatch.delenv("MG_TORCH_LOSS", raising=False)
    monkeypatch.delenv("MG_DN_GATE_NEIGHBOURS", raising=False)
    t = geometry_terms(_args(), [n_sum, z], a, K, gt_d, gt_n, None)
    assert torch.isfinite(t["depth_normal"])
