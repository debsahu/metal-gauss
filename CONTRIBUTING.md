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
