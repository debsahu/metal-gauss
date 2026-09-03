# Architecture

How the Metal backend is built and what it is checked against. Results live
in [BENCHMARKS.md](BENCHMARKS.md); rejected approaches, with numbers, live in
[NEGATIVE_RESULTS.md](../bench/results/NEGATIVE_RESULTS.md).

## 🔬 Correctness

| check | result |
|---|---|
| forward vs float64-gradcheck oracle | max abs ≤ 2×10⁻³ |
| gradients vs torch autograd of the oracle | rel ≤ 3.5×10⁻⁶, cosine 1.000000 (all five tensors) |
| Metal binning vs torch binning | **bit-identical** image and alpha |
| fused SSIM vs the `F.conv2d` expression | value bit-identical, gradient cosine 1.00000012 |
| fused Adam vs `torch.optim.Adam` | rel 4.7×10⁻⁷ after 30 steps |
| **fused aux channels** vs separate passes, forward | **bit-identical** (`torch.equal`) |
| **fused aux channels**, gradients vs the multipass oracle | rel ≤ 1×10⁻⁵, cosine ≥ 1 − 10⁻⁶ |
| **aux lanes vs the alpha VJP** | a 100× aux cotangent moves `d_uv`/`d_conic`/`d_opacity` by rel < 10⁻⁶ |
| **`normals_from_depth`** kernel vs `torch.autograd` | rel ≤ 1×10⁻⁵, cosine ≥ 1 − 10⁻⁶ (gather adjoint, no atomics → deterministic) |
| **fused geometry losses** vs an **f64** reference | 7.0×10⁻⁶ — closer to truth than torch's own f32 chain at 8.1×10⁻⁶ |

**Auxiliary channels carry a different gradient contract from RGB.** The RGB lanes fold
into the alpha VJP; the aux lanes deliberately do not, so a depth or normal loss cannot
reduce its error by changing opacity or footprint instead of moving the gaussian. Omitting
that separation collapses splats into needles (in-plane aspect 0.296 → 0.066, needle
fraction 17% → 57%). `render()` therefore requires `aux_detach_weights` explicitly and
refuses live-weight requests on the fused path rather than silently serving the wrong
contract.

Tests: `pytest tests`. Thirteen of them are in `tests/test_runner.py` and guard the
benchmark harness rather than the kernels — every wrong number this project has published came
from a harness, and the two that did the most damage are now regression tests.

## ⚙️ How it works

- **Projection + SH + activations** fused into one kernel, thread-per-gaussian, analytic backward.
- **Tile binning in Metal**, including an exact ellipse-vs-tile test that drops 38.7 % of
  tile-gaussian pairs while leaving the image bit-identical (gsplat's AccuTile, in-kernel).
- **Rasterization** with threadgroup-cooperative attribute staging and simdgroup-aggregated
  gradient atomics (32 lanes → 1 atomic).
- **SSIM entirely in Metal** — stack construction, both separable blur passes and the tail. The
  symmetric gaussian is self-adjoint, so one kernel serves the forward and both adjoint passes.
- **Adam in one pass** instead of torch's five, at the measured memory-bandwidth floor.
- **MCMC densification** with gradient-targeted relocation, exact multiplicity correction and
  Adam-moment resets.
- Metal compiles at **runtime** via `newLibraryWithSource` — hence no Xcode.
