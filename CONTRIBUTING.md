# Contributing

Issues and pull requests are welcome. Two things about this repo are unusual
enough to be worth knowing before you start.

## You need an Apple Silicon Mac

The kernels are Metal and compile at runtime. There is no CPU fallback and no
CUDA path, so the rasteriser tests cannot run anywhere else.

CI is deliberately CPU-only: it runs the harness tests and checks that the
generated tables still match their JSONs. A job that needs a GPU it does not
have would fail forever, which is worse than not running. So **CI passing does
not mean the kernels are fine** — run the full suite locally.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[bench,train]" pytest lpips scikit-image imageio tqdm torchvision
uv run --frozen python scripts/fix_openmp.py   # REQUIRED -- see below
.venv/bin/python -m pytest -q          # 362 tests, needs an Apple GPU
```

**The dedup step is not optional, and it is not one-time.** Four of those wheels
(torch, pycolmap, sklearn, open3d) each vendor their own `libomp.dylib`, and
importing more than one in a process aborts with `OMP: Error #15`. Without the
step, the `pytest` line above hard-crashes rather than failing a test.
`scripts/fix_openmp.py` points every copy at a single real library; we do not
use `KMP_DUPLICATE_LIB_OK=TRUE`, which LLVM documents as able to "silently
produce incorrect results" — exactly the failure class this repo exists to catch.

Re-run it after **any** dependency change. `uv run --frozen` reconciles the venv
against the lockfile and restores the vendored copy over the symlink, so an
out-of-lock install (including the one above) leaves the next run broken.

`scripts/fix_openmp.py` defaults to `.venv`. A second environment needs `--venv <dir>`
explicitly — without it the script happily reports "already deduplicated — nothing to
do" about the *wrong* venv and the next run aborts with `OMP: Error #15` (exit 134).

### That `uv pip install` line installs an OUT-OF-LOCK torch (the test it flipped is fixed)

`pyproject.toml` asks only for `torch>=2.5`, so the line above resolves to the newest
wheel (**2.14.0** as of 2026-09-04) while `uv.lock` pins **2.13.0**. Prefer the lockfile:
the two are not interchangeable for benchmarking, and only one of them is what CI-equivalent
numbers were measured on.

**The test this used to flip has been replaced, and both torches are green as of
2026-09-04** — `366 passed` on 2.13.0 (plus the allocator-dependent skip below) and
`366 passed` on 2.14.0, full `pytest -q`, one machine, private `TORCH_EXTENSIONS_DIR`.

What it used to be, kept because the failure shape is worth knowing.
`test_flatten_flag_actually_reaches_the_training_loss` asserted a **≥ 10%** drop in the
median smallest axis over two 40-step training arms, and measured **10.51–10.67%** on the
pinned torch against **9.38–9.48%** on 2.14.0 — about 0.5 pp of margin, which the wheel
bump spent. Nothing was ever wrong with the flatten flag.

**The bar was measuring the optimiser's step budget, not flatten's strength, and the
mechanism is now measured.** Adam's first update is exactly `-lr·sign(g)`, so *any* weight
large enough to flip the min-axis gradient sign produces the identical step. Measured at
step 1: `w = 50` and `w = 100` export min-axis p50 **0.278286368** and **0.278286368**,
bit-identical, and the log-space shift against `w = 0` is exactly `2·lr = 1.0e-2` on the
median splat. That is why raising the weight could never restore the margin.

The replacement asserts the mechanism the name claims, at three named tests off one
module-scoped fixture of three **one-step** arms (`w = 0, W, 2W`), where the splat state is
still the seed so the only admissible difference between arms is `w × flatten`:

- `test_flatten_flag_actually_reaches_the_training_loss` — the total loss is linear in `w`.
  Measured rel residual **1.9e-8 / 6.4e-8 / 1.9e-8**, i.e. f32 rounding of a single
  addition. **NOT** the `4.4e-16 / 1.8e-15` recorded here previously: the loss is f32, so
  `fl32(loss + w·flatten) − loss` cannot agree with `w·flatten` to f64 roundoff. That
  earlier figure does not reproduce.
- `test_flatten_gradient_reaches_log_scales` — `d(loss)/d(log_scales)` is linear in `w`,
  scored **per element** on the argmin lanes (max **9.9e-8–1.5e-7**) with leakage onto the
  other lanes ≤ **1.2e-8** of the on-lane magnitude.
- `test_flatten_moves_the_min_axis_through_the_optimiser` — direction only, no magnitude
  bar, so it cannot saturate.

Seven mutants were run against them, each kill recorded by test **name**: inert term,
detached `log_scales`, half weight, sign flip, double add, `min`→`max`, `min`→`mean`.
Two are worth knowing. A **detached** term and a term spread over **all three axes** both
leave the loss value exactly right, so the loss probe is blind to both and only the
gradient probe kills them. And `min`→`max` is **behaviourally identical** at step 1,
because the seed is exactly isotropic (per-splat log-scale spread max `0.000e+00` on 400
of 400) — lane *selection* is owned by the two hand-built unit tests above it, which do
kill it.

**Neither documented route gives a green suite.** All four rows measured on `main`
`86a9e03` on 2026-09-04, full `pytest -q`, one machine, private `TORCH_EXTENSIONS_DIR`:

| environment | torch | result |
|---|---|---|
| `uv pip install` line above | 2.14.0 | **361 passed, 1 failed** — `test_flatten_flag_actually_reaches_the_training_loss`. **Green as of 2026-09-04**: that test is replaced, and this route now measures `366 passed`. |
| `uv sync --frozen --extra bench --extra train` | 2.13.0 | **361 passed, 1 failed** — `test_ssim_matches_skimage` (`ModuleNotFoundError`) |
| ... `--extra test` as well | 2.13.0 | **361 passed, 1 failed** — same; `scikit-image` is in no extra |
| **lockfile + `scikit-image` installed explicitly** | 2.13.0 | **362 passed** ✅ |

So the green recipe is the lockfile sync **plus** `scikit-image`, which the extras do not
carry:

```bash
uv venv --python 3.12 .venv
uv sync --frozen --extra bench --extra train
uv pip install --python .venv/bin/python scikit-image pytest plyfile
.venv/bin/python scripts/fix_openmp.py --venv .venv    # AFTER the out-of-lock install
.venv/bin/python -m pytest -q
```

`fix_openmp.py` must come last: the `uv pip install` restores torch's vendored `libomp`
over the symlink, exactly as the paragraph above warns.

### `max|Δ| / max|ref|` goes BLIND on a fixture with a degenerate lane

Audited 2026-09-04 across the whole suite, by hooking `Tensor.__sub__` so the form is
caught however it is spelled. Read this before adding a gradient comparison.

`tests/test_fused_geom_loss.py::_bars` asserts `rel = max|got − ref| / max|ref| ≤ 1e-5`.
On the `exposing()` fixture `max|ref|` for `d(n_sum)` is **3.63e11** — contributed
entirely by the 15% of pixels the fixture sets to `alpha = 0`, where `alpha` clamps to
`1e-10` and the gradient explodes — while the median `|ref|` is **5.4e-5**. A dynamic
range of **6.7e15** turns the stated bar into `max|Δ| ≤ 3.6e6` in absolute terms, and
**85.7% of the nonzero reference sits below that**. Cosine is pinned to 1.0 by the same
few components.

The symptom is unmistakable once you look for it. Across clean code and three different
kernel mutants, that assertion returns

    2.2241192858826983e-08

**bit-identical to 17 significant figures** — the max is attained at the same degenerate
element every time, so the statistic is a constant of the fixture and not a function of
the kernel at all. `test_metric_space_also_matches` is blind on both of its assertions the
same way (`metric dL/dz` has an effective tolerance of ±54.8).

This is how mutant **M9** (dropping the normalise Jacobian) survived the full suite while
being wrong on 4,891 of 6,240 elements at a median 19% error. **It is a property of the
statistic's FORM, not of its threshold — do not respond by tightening 1e-5.**

What to do instead, as `test_normalise_jacobian_is_present_in_the_nsum_gradient` does:
**mask out the degenerate support first, then score.** A high-quantile denominator alone
does **not** rescue it — measured, `p99|ref|` over the unmasked tensor is 3.63e11,
identical to the max, because the degenerate population is far larger than 1%. The mask is
what makes the scale mean anything.

Scope, so this is not read as wider than it is. Every other comparison in the suite was
measured and is fine: the next-worst dynamic range is **1.2e4**
(`test_sh_split_layout.py`, which asserts against an absolute floor anyway), the Tier 2
aux equivalence gate sits at **1.8e2** with 0.25% blind, and the `normals_from_depth`
kernel tests at **≤ 4.0e2** with ≤ 0.41% blind. The two adjoint derivation scripts behind
`research/normals-from-depth-adjoint.md` (31/31) and `research/depth-normal-loss-adjoint.md`
(44/44) were re-run instrumented and are sound: max dynamic range **89** for the former,
and for the latter 40 of 42 checks under 2.7e3 — the two `normalise VJP` checks do carry
the structure (97.4% blind) but their per-element errors are **1.7e-16** (f64) and
**1.2e-7** (f32), so those claims survive a sound statistic.

### The Metal extension cache is SHARED across every checkout — set `TORCH_EXTENSIONS_DIR`

`torch.utils.cpp_extension.load` keys its build directory on the extension **name**, not
on the source path, so every checkout on the machine compiles into the same
`~/Library/Caches/torch_extensions/py312_cpu/metal_gauss_metal/`. Two `git worktree`s —
or one worktree and the primary checkout — share one `.so` and one `FileBaton` lock.

Both consequences have been observed:

- A build in your worktree **replaces the `.so` another checkout is currently
  executing**. When the two venvs are on different torch versions the replacement is
  ABI-mismatched, and the other run fails somewhere with no visible relation to what it
  was testing.
- A killed build leaves the shared `lock` behind and hangs *every* checkout, which is
  the hang documented below.

**Set a private cache for any work outside the primary checkout**, for the whole
session rather than for the one command you expect to rebuild:

```bash
export TORCH_EXTENSIONS_DIR="$PWD/.torch_ext"
```

The `.metal` sources are compiled at runtime by `newLibraryWithSource`
(`metal_gauss/metal_backend.py`), so editing a shader needs no C++ rebuild — but the
Objective-C++ bridge does, and the bridge is what this cache holds.

### If a run hangs at 0% CPU with an empty log

Delete `~/Library/Caches/torch_extensions/py312_cpu/metal_gauss_metal/lock`
— but only if `lsof` on it shows nothing, since a held lock means a real
concurrent build.

`metal_gauss/metal_backend.py` builds the Metal extension through
`torch.utils.cpp_extension.load`, which serialises builds with a `FileBaton`:
create the lock `O_EXCL`, build, release. A process that is killed or `abort()`s
in between never releases it, and `FileBaton.wait()` is `while
os.path.exists(lock): time.sleep(0.1)` with no timeout — so one dead run makes
every later run spin forever, even though the `.so` is already built. The
symptom is distinctive: sleeping, 0% CPU, ~150 MB RSS, and no output at all.

`sample <pid>` confirms it in one command — the main thread sits in `time_sleep`
under `THPFunction_apply`. The parked `__kmp_*` worker threads in that output are
the normal idle state, not the problem.

## Performance claims need a warm machine and an idle one

Apple Silicon runs short bursts at boost clock. The same kernel times ~12 ms or
~19 ms depending on how warm the laptop is, so the first measurement of anything
is optimistic. `bench/quick.py` ramps for 2 seconds before timing; if you write
a new benchmark, do the same.

Wall-clock numbers also assume the GPU is yours alone. `bench/runner.py` exposes
`require_gpu_exclusive()` — call it at the top of anything that reports timings.

Before quoting a difference, check it against the run-to-run spread. Ours is
0.19 dB worst case; a competitor's can be far wider. A difference smaller than
the spread is not a result. `bench/compare/variance.py` repeats a configuration
and reports mean, stdev and range.

## Tables and figures are generated

Every table in `README.md` and `docs/` is produced from a JSON in
`bench/results/`. Do not hand-edit them between the `<!-- BEGIN:x -->` markers.

```bash
.venv/bin/python bench/readme_tables.py          # regenerate
.venv/bin/python bench/readme_tables.py --check  # fails if a table drifted
```

The figures come from `bench/compare/plot_pareto_scenes.py` and
`plot_margin.py`. `docs/BENCHMARKS.md` lists every script and what it is for.

## Where things live

| | |
|---|---|
| `metal_gauss/` | the trainer and the Metal kernels (`csrc/*.metal`) |
| `bench/` | benchmarks, table generation, provenance checks |
| `bench/compare/` | competitor harnesses and the figures |
| `tests/` | correctness, mostly against a torch reference |
| `bench/results/NEGATIVE_RESULTS.md` | levers measured and rejected — read this before optimising |

## Scope

Competitor benchmarks run their shipped configuration, scored by one evaluator
on the official split, strictly sequentially. If you add an implementation,
follow that and say which configuration you used — a comparison is only worth
publishing if someone else can reproduce the protocol.
