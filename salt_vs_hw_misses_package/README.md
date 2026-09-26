# SALT vs. hardware L1D misses package

This package generates SALT miss-count JSON files from the repository's MLIR,
compares them with packaged measurements of Linux's generic L1D/read/miss event,
and creates SVG and PDF plots under `results/salt-vs-hardware/` at the repository
root. On the Intel systems validated for this artifact, that generic selector
maps to `L1D.REPLACEMENT`; the historical load-miss labels are retained for
compatibility with the paper data and plotting workflow.

The package includes the exact C sources and PMC collector for the 16
orig/tiled benchmarks plus original stencil, for 17 benchmarks total.

## Reproduce Figure 4 with the reference data

The normal, portable artifact path uses the checked-in Intel i7-7700 PMC data.
It does not read hardware counters on the current machine. From the repository
root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

./scripts/run-salt-vs-hardware-evaluation.sh
```

The repository-level driver builds the analyzer once and then performs all
three analysis steps:

1. Runs the SALT analyzer on each MLIR input, then derives its miss count for
   the selected cache size.
2. Matches those estimates with the PMC measurements by benchmark name and
   writes `results/salt-vs-hardware/salt_vs_hw_misses_results.csv` at the
   repository root.
3. Calculates MAPE (the percentage form of MARE) and Pearson correlation and
   writes `results/salt-vs-hardware/salt_vs_hw_misses.{svg,pdf}`.

By default it compares against the pre-measured Intel i7-7700 data in
`salt_vs_hw_misses_package/data/pmu_i7-7700_result.csv`.

To compare against another PMC file while retaining the reference cache
geometry, use the wrapper's `--pmc` option:

```bash
RESULTS_DIR=results/salt-vs-hardware-local \
./scripts/run-salt-vs-hardware-evaluation.sh \
  --pmc /path/to/pmc.csv
```

For lower-level use with an already built analyzer, the Python script exposes
custom output paths and cache geometry:

```bash
python3 salt_vs_hw_misses_package/salt_vs_hw_misses.py \
  --analyzer target/release/analyzer \
  --pmc /path/to/pmc.csv \
  --results-output results/salt-vs-hardware/my_results.csv \
  --plot-output results/salt-vs-hardware/my_graph.svg \
  --cache-size-bytes 32768 \
  --cache-line-bytes 64 \
  --elements-per-cache-line 8
```

The default cache model is 32 KiB with 64-byte lines. Therefore, each SALT
miss-ratio curve is evaluated at 512 cache lines. The derived miss count is:

```text
salt_estimated_miss_count = selected_miss_ratio × total_count
```

## SALT generation

The repository-level driver builds the release analyzer once and passes
`target/release/analyzer` to this script with `--analyzer`. The script then
invokes that binary directly for each input, avoiding repeated Cargo build
checks. The SALT block size is 8. The eight original and eight tiled
contraction inputs come from `benchmarks/mlir-contractions/constant/`; the
stencil input comes from `benchmarks/examples/const_stencil5pt.mlir`.

Here, `--block-size=8` means eight target elements per cache line, not eight
bytes. These benchmarks use 8-byte `double` values, so eight elements model the
Intel i7-7700's 64-byte L1D cache line.

For a short end-to-end check, run:

```bash
./scripts/run-salt-vs-hardware-evaluation.sh --smoke-test
```

This evaluates an original contraction, its tiled form, and the stencil. The
full workflow regenerates all 17 SALT predictions. SALT is fast enough that
this simpler, self-contained behavior is preferred over detecting and copying
JSON files from another workflow.

Generated JSON defaults to `results/salt-vs-hardware/salt-json/` at the
repository root, with tiled results in its `tiled/` subdirectory. Pass
`--salt-json-dir` only when a different output location is wanted. Each
generated file provides
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

## Optional: collect PMCs on another machine

This step is optional for artifact evaluation. The normal Figure 4 workflow
uses the checked-in `data/pmu_i7-7700_result.csv`, measured on an Intel Core
i7-7700 with simultaneous multithreading (hyperthreading) disabled.

The CSV's historical load-miss fields contain Linux's generic
`L1-dcache-load-misses` event. On the validated i7-7700, i7-6700, and Xeon Gold
6126 systems, Linux maps this selector to `L1D.REPLACEMENT`, not to the raw
retired-load event `MEM_LOAD_RETIRED.L1_MISS`. Do not mix those two event types
in one comparison.

For a meaningful fresh measurement, pass `--cpu` to pin the benchmark to one
logical CPU. The collector discovers every other logical CPU sharing that
physical core, temporarily offlines any online siblings, and restores only the
CPUs it changed when collection finishes or raises an ordinary error. Pinning
with `taskset` alone is not sufficient: a running sibling can perturb the count.

Changing CPU online state normally requires administrator privileges. The
collector writes sysfs directly when allowed; otherwise it requests `sudo` only
for `tee` on the sibling CPU's `online` file. If authorization fails, no
measurement is taken. Follow the evaluator site's host-management policy. After
an unrecoverable interruption such as `SIGKILL` or power loss, inspect the
sibling set and online state and restore it if necessary (replace `1` with the
selected logical CPU):

```bash
cat /sys/devices/system/cpu/cpu1/topology/thread_siblings_list
lscpu -e=CPU,CORE,ONLINE
```

Record the CPU model, selected logical CPU, sibling state, compiler, kernel
event mapping, and repeat count with any new result.

From the repository root, collect the median of three runs into `results/`:

```bash
mkdir -p results/salt-vs-hardware-local
python3 salt_vs_hw_misses_package/benchmarks/collect_pmc.py \
  --cpu 1 --repeats 3 \
  --output results/salt-vs-hardware-local/pmu.csv

RESULTS_DIR=results/salt-vs-hardware-local \
./scripts/run-salt-vs-hardware-evaluation.sh \
  --pmc results/salt-vs-hardware-local/pmu.csv
```

The wrapper defaults to a 32 KiB L1D, 64-byte cache lines, and eight `double`
elements per line. If the measured processor differs, also pass
`--cache-size-bytes`, `--cache-line-bytes`, and
`--elements-per-cache-line`. The collector's source manifest and measurement
behavior are documented in `benchmarks/README.md`; counter semantics and Linux
permissions are documented in `pmc_measurement/README.md`.

## Package layout

```text
salt_vs_hw_misses_package/
├── README.md
├── requirements.txt
├── salt_vs_hw_misses.py       # only root Python workflow
├── data/
│   └── pmu_i7-7700_result.csv # checked-in reference measurement
├── output/
│   └── salt_vs_hw_misses.svg  # checked-in reference figure
├── benchmarks/
│   ├── Makefile
│   ├── benchmark_runner.c
│   ├── collect_pmc.py
│   ├── include/kernel_interface.h
│   └── kernels/{orig,tiled}/*.c
└── pmc_measurement/
    ├── README.md
    ├── example.c
    └── pmc_l1d_misses.h
```
