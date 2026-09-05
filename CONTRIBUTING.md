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

### That `uv pip install` line installs an OUT-OF-LOCK torch, and it flips a test

`pyproject.toml` asks only for `torch>=2.5`, so the line above resolves to the newest
wheel (**2.14.0** as of 2026-09-04) while `uv.lock` pins **2.13.0**. Measured on `main`
`86a9e03` in a clean worktree with everything else held fixed:

    tests/test_geometry_loss.py::test_flatten_flag_actually_reaches_the_training_loss
        torch 2.13.0 (lockfile)   PASS
        torch 2.14.0 (line above) FAIL

Nothing is wrong with the flatten flag. The term reaches the loss exactly — at step 1,
where the splat state is still the seed in both arms, `Δloss = w × flatten` to machine
precision at w = 10 and w = 50. The test asserts a **≥ 10%** drop in the median smallest
axis over 40 steps and measures **10.51–10.67%** on the pinned torch against
**9.38–9.48%** on 2.14.0. The bar has about 0.5 pp of margin and the wheel bump spends
it.

The assertion is also weaker than it looks: the effect **saturates**. At w = 1, 50 and
200 the reduction is the same to within 0.1 pp, because Adam's update is scale-invariant
and 40 steps at the scales group's `lr = 5e-3` caps how far `log_scales` can travel. So
raising the weight cannot restore the margin — only more steps, a larger scale lr, or a
bar set from a measured floor rather than a round number.

**Neither documented route gives a green suite.** All four rows measured on `main`
`86a9e03` on 2026-09-04, full `pytest -q`, one machine, private `TORCH_EXTENSIONS_DIR`:

| environment | torch | result |
|---|---|---|
| `uv pip install` line above | 2.14.0 | **361 passed, 1 failed** — `test_flatten_flag_actually_reaches_the_training_loss` |
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
