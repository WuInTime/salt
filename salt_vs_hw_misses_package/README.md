# SALT vs. hardware L1D misses package

This package generates SALT miss-count JSON files from the repository's MLIR,
compares them with measured PMC L1D load misses, and creates
`salt_vs_hw_misses.svg`.
It includes the exact C sources and PMC collector for the 16 orig/tiled
benchmarks plus original stencil, for 17 benchmarks total.

## One-script analysis workflow

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 salt_vs_hw_misses.py
```

That single command performs all three analysis steps:

1. Runs the SALT analyzer on each MLIR input, then derives its miss count for
   the selected cache size.
2. Matches those estimates with the PMC measurements by benchmark name and
   writes `data/salt_vs_hw_misses_results.csv`.
3. Calculates MARE and Pearson correlation and writes
   `output/salt_vs_hw_misses.svg`.

By default it compares against the pre-measured Intel i7-7700 data in
`data/pmu_i7-7700_result.csv`.

To compare against a newly collected PMC file instead:

```bash
python3 salt_vs_hw_misses.py --pmc data/pmu_results_17_new.csv
```

Custom output paths and cache geometry are also supported:

```bash
python3 salt_vs_hw_misses.py \
  --pmc /path/to/pmc.csv \
  --results-output data/my_results.csv \
  --plot-output output/my_graph.svg \
  --cache-size-bytes 32768 \
  --cache-line-bytes 64
```

The default cache model is 32 KiB with 64-byte lines. Therefore, each SALT
miss-ratio curve is evaluated at 512 cache lines. The derived miss count is:

```text
salt_estimated_miss_count = selected_miss_ratio × total_count
```

## SALT generation

The script runs `cargo run --locked --release --bin analyzer` with the SALT
block size set to 8. The eight original and eight tiled contraction inputs come
from `benchmarks/mlir-contractions/constant/`; the stencil input comes from
`benchmarks/examples/const_stencil5pt.mlir`.

Generated JSON defaults to `results/mlir-contractions/work/constant/` at the
repository root, with
tiled results in its `tiled/` subdirectory. Pass `--salt-json-dir` only when a
different output location is wanted. Each generated file provides
`total_count`, `miss_ratio`, and `turning_points`, for example:

```json
{
  "total_count": "884736",
  "miss_ratio_curve": {
    "miss_ratio": [1.0, 0.0842, 0.0425],
    "turning_points": [0.0, 3.0, 13.1]
  }
}
```

Expected original filenames use the analyzer's `constant_*` convention:

```text
constant_3d_tensor_vector-salt.json
constant_4d_tensor-salt.json
constant_attention_score-salt.json
constant_batched_gemm-salt.json
constant_context_lookup-salt.json
constant_matrix_matrix-salt.json
constant_matrix_vector-salt.json
constant_rowwise_softmax_max-salt.json
constant_stencil5pt-salt.json
```

Expected tiled filenames are:

```text
tiled_3d_tensor_vector-salt.json
tiled_4d_tensor-salt.json
tiled_attention_score-salt.json
tiled_batched_gemm-salt.json
tiled_context_lookup-salt.json
tiled_matrix_matrix-salt.json
tiled_matrix_vector-salt.json
tiled_rowwise_softmax_max-salt.json
```

The script requires all 17 MLIR inputs and all 17 PMC rows. It reports missing
or duplicate results rather than silently omitting benchmarks.

## PMC input formats

Both packaged PMC formats are accepted:

```text
program,csv_l1d_load_miss
orig_matrix_matrix,992723
```

```text
kernel,L1D.load_miss
constant_matrix_matrix,992723
```

Names beginning with `constant_` are automatically normalized to the graph's
`orig_` naming convention.

## Collect new PMC measurements

The PMC collector remains under `benchmarks/` because it is part of benchmark
execution, not the root analysis workflow:

```bash
cd benchmarks
make
python3 collect_pmc.py --cpu 1 --repeats 3
cd ..
```

This writes `data/pmu_results_17_new.csv`. Then pass it to the one root script
with `--pmc data/pmu_results_17_new.csv`.

See `benchmarks/README.md` for the source manifest and
`pmc_measurement/README.md` for Linux counter permissions and semantics.

## Package layout

```text
salt_vs_hw_misses_package/
├── README.md
├── requirements.txt
├── salt_vs_hw_misses.py       # only root Python workflow
├── data/
│   ├── pmu_i7-7700_result.csv
│   └── pmu_results_17_new.csv
├── output/
│   └── salt_vs_hw_misses.svg
├── benchmarks/
│   ├── Makefile
│   ├── benchmark_runner.c
│   ├── collect_pmc.py
│   ├── include/kernel_interface.h
│   └── kernels/mlir/{orig,tiled}/*.c
└── pmc_measurement/
    ├── README.md
    ├── example.c
    └── pmc_l1d_misses.h
```
