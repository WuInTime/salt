# MLIR contraction cache experiment

This directory contains the eight original and tile-size-8 MLIR contraction
benchmarks used to compare SALT miss-count predictions with Cachegrind
simulations of fully associative, 8-way, and 12-way L1 data caches.
For the complete evaluator workflow and paper-result map, start with the
repository-root `README.md`.

## Dependencies

- Rust and Cargo (use the toolchain pinned by `rust-toolchain`)
- LLVM/MLIR 21, including `mlir-opt`
- `clang++` with static-linking support
- Valgrind with Cachegrind
- Python 3.12 or newer
- Python packages: `matplotlib`, `numpy`, and `pandas`

Install the pinned Python dependencies from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

The experiment starts from MLIR, so Polygeist is not required. The older C
benchmark scripts under `scripts/` contain a machine-specific Polygeist path
and are not part of this workflow.

## Quick validation

From the repository root, run:

```bash
RESULTS_DIR=/tmp/mlir-contraction-smoke \
  bash scripts/run-mlir-contraction-evaluation.sh --smoke-test
```

This tests the original and tiled 3D tensor-vector kernel with a reduced 1 KiB
cache-size sweep. It exercises MLIR input staging, SALT, executable emission,
all three Cachegrind organizations, SQLite output, and both plot variants.

## Full evaluation

```bash
bash scripts/run-mlir-contraction-evaluation.sh
```

The full sweep uses the original parameters:

- tile size: 8 for every affine loop;
- SALT block size: 8 double elements, corresponding to a 64-byte line;
- L1 upper limit: 65,536 bytes;
- L1 line size: 64 bytes;
- LL cache: 65,536 bytes, 16-way, with 64-byte lines.

The fully associative simulation tests every 64-byte increment through 65,536
bytes. The 8-way and 12-way simulations test geometrically increasing cache
sizes beginning at one set: 512 and 768 bytes, respectively. The complete run
launches 16,624 Cachegrind processes and can take substantial time.

Results are written to `results/mlir-contractions/`:

- `data-fully-associative.db`
- `data-8way-associative.db`
- `data-12way-associative.db`
- `miss_count_comparison_all_programs_log.svg`
- `miss_count_comparison_all_programs_log.pdf`
- `miss_count_comparison_all_programs_linear.svg`
- `miss_count_comparison_all_programs_linear.pdf`
- `work/constant/` (staged MLIR inputs and generated SALT JSON)

The script copies the original and checked-in tiled MLIR into
`RESULTS_DIR/work`, generates SALT JSON there, and points
`scripts/graph_contractions_salt_vs_cg.py` at that staged tree.
Checked-in benchmark artifacts are therefore not modified. Use an empty
`RESULTS_DIR`: the script refuses to add records to existing databases, which
prevents accidental mixing of reduced and full sweeps.

## Individual stages

The artifact uses the checked-in tile-size-8 inputs. For reference, an
equivalent tiling pipeline is:

```bash
mlir-opt constant_input.mlir \
  --affine-loop-tile="tile-size=8" \
  --affine-loop-normalize="promote-single-iter" \
  --affine-simplify-structures \
  --canonicalize \
  -o tiled_output.mlir
```

Generate a SALT curve with the already built analyzer:

<!-- cargo run --locked --release -p analyzer --no-default-features --bin analyzer -- \ -->
```bash
target/release/analyzer \
  -i input.mlir --json -o output.json salt --block-size=8
```

Regenerate plots from existing databases and SALT JSON:

```bash
MPLCONFIGDIR=/tmp/salt-matplotlib \
  python3 scripts/graph_contractions_salt_vs_cg.py
```

All paths accepted by `graph_contractions_salt_vs_cg.py` can be overridden; run
`python3 scripts/graph_contractions_salt_vs_cg.py --help` for details.

## Selected timing figure

Build the Barvinok-enabled analyzer and Cachegrind runner first:

```bash
cargo build --locked --release -p analyzer --bin analyzer
cargo build --locked --release -p cachegrind-runner --bin cachegrind-runner
```

The full evaluation command already records wall time around each tiled
fully-associative sweep, measures symbolic Barvinok and SALT after the cache
experiment, and writes:

- `results/timing-simulation-wall.tsv`
- `results/timing-selected.json`
- `results/timing-selected.svg`
- `results/timing-selected.pdf`

Thus the normal artifact workflow is simply:

```bash
bash scripts/run-mlir-contraction-evaluation.sh
```

To regenerate only the selected figure from those measured results:

```bash
MPLCONFIGDIR=/tmp/salt-matplotlib \
python3 scripts/timing_selected.py \
  --data results/mlir-contractions/timing-selected.json \
  --out results/mlir-contractions/timing-selected.svg
```

If the cache experiment completed but the later symbolic timing or plotting
stage failed, reuse the completed simulation timing without rerunning
Cachegrind:

```bash
python3 scripts/measure_timing_selected.py \
  --simulation-times results/mlir-contractions/timing-simulation-wall.tsv \
  --simulation-db results/mlir-contractions/data-fully-associative.db \
  --out results/mlir-contractions/timing-selected.json
```

The timing manifest contains the wall-clock samples used for the graph,
commands, tool versions, cache parameters, and the cache sizes actually
observed in the simulation database. SQLite's per-record `process_time` is not
used for the blue bars: the cache runner evaluates cache sizes in parallel, so
summing those subprocess times would not represent elapsed experiment time.

The simulation series is the complete fully-associative tiled sweep and is the
slow part. The timing manifest records the cache-size list observed in the
simulation database so reviewers can verify the measured configuration.

## Methodology notes and known limitations

1. The constant MLIR inputs and their `simulation.prologue` array declarations
   were created manually from the symbolic inputs. No symbolic-to-constant MLIR
   generator is currently preserved.
2. A C padding utility exists under `benchmarks/einsum/constant_global/`, but
   it does not operate on these MLIR files. The contraction dimensions are
   tile-compatible, but no separate MLIR physical-padding stage is preserved.
3. `cachegrind-runner` emits a freestanding executable containing only the loop
   structure and memory references. Every modeled access is emitted as a write
   followed by a compiler memory barrier.
4. The simulation prologues declare `double` arrays, including for MLIR kernels
   whose element type is `f32`. This intentionally makes the modeled element
   size eight bytes but should be stated when interpreting the results.
5. Cachegrind rejects the smallest 64-byte fully associative cache. The current
   runner does not propagate that failure and stores a zero-valued record; the
   recovered plot script clips this leading nonpositive point.
6. Deep tiled nests may introduce a small number of compiler-generated data
   references. In the smoke test, Cachegrind reported 884,790 references for
   the tiled 3D kernel versus SALT's 884,736 logical accesses.
