# SALT

SALT is an automatic asymptotic locality analyzer for affine loop programs.
This repository contains the Rust implementation, benchmark inputs, and three
drivers that reproduce the evaluation artifacts.

SALT is distributed under the MIT License; see `LICENSE`.

## Artifact overview

The artifact has three evaluation workflows. These scripts build the required
binaries, run the analyses and Cachegrind simulations, and generate the plots
and intermediate data used by the evaluation:

| Workflow | Command | Output directory |
| --- | --- | --- |
| Matrix multiplication T0-T2 | `./scripts/run-matmul-t0-t2-evaluation.sh` | `results/matmul-t0-t2/` |
| MLIR contractions | `./scripts/run-mlir-contraction-evaluation.sh` | `results/mlir-contractions/` |
| SALT vs. hardware L1D misses | `./scripts/run-salt-vs-hardware-evaluation.sh` | `results/salt-vs-hardware/` |

Generated results and Cargo build products are intentionally excluded from
version control. Each workflow can write somewhere else by setting
`RESULTS_DIR`.

Documentation is organized by audience:

| Document | Purpose |
| --- | --- |
| This README | Evaluator setup, smoke/full commands, paper-result mapping, and the optional local-PMC path |
| `benchmarks/matmul-t0-t2/README.md` | Figure 1 inputs and cache-sweep configuration |
| `benchmarks/mlir-contractions/README.md` | Figures 2–3 methodology, outputs, and limitations |
| `salt_vs_hw_misses_package/README.md` | Figure 4 data processing and detailed fresh-measurement procedure |
| `salt_vs_hw_misses_package/benchmarks/README.md` | The 17 native kernels and PMC collector behavior |
| `salt_vs_hw_misses_package/pmc_measurement/README.md` | Low-level Linux PMU interface and permissions |

## Recommended Docker workflow

Docker is the recommended evaluator path. The image uses Ubuntu 24.04, the
official LLVM/MLIR 21 packages, the pinned Rust toolchain and the Python
versions in `requirements.txt`.

Build the image from the repository root:

```bash
docker build --progress=plain \
  --build-arg ARTIFACT_REVISION="$(git rev-parse HEAD)" \
  -t salt-artifact:ae .
```

Use `--no-cache` once before submission to verify a completely clean build.

Check the toolchain and run the Rust test suite:

```bash
docker run --rm salt-artifact:ae test
```

The `test` command isolates analyzer unit tests into separate processes so the
suite also works with Symbolica's free restricted mode.

Run all three reduced evaluation workflows and retain their results on the host:

```bash
mkdir artifact-results
docker run --rm --init \
  -v "$PWD/artifact-results:/artifact/results" \
  salt-artifact:ae smoke
```

The mounted directory must not already contain `contraction-smoke/`,
`matmul-smoke/`, or `salt-vs-hardware-smoke/`; this prevents results from
different runs from being mixed.

After validating the smoke run, start the complete evaluation with an empty
directory:

```bash
mkdir full-results
docker run --rm --init \
  -v "$PWD/full-results:/artifact/results" \
  salt-artifact:ae reproduce
```

If a separately obtained Symbolica license is required, pass the existing
shell variable at runtime rather than storing it in the image:

```bash
docker run --rm --init \
  -e SYMBOLICA_LICENSE \
  -v "$PWD/artifact-results:/artifact/results" \
  salt-artifact:ae smoke
```

The container records its tool versions in `environment-smoke.txt` or
`environment-full.txt` alongside the results. Run `docker run --rm
salt-artifact:ae help` for all container commands.

## Requirements

The direct, non-containerized workflow requires:

- Linux on an x86-64 machine;
- Rustup and Cargo (the pinned toolchain is in `rust-toolchain`);
- LLVM/MLIR 21 development libraries and tools, including Polly;
- Clang/Clang++ 21, LLD, GCC/G++, and a C/C++ build toolchain;
- CMake, Autoconf, Automake, Libtool, and `pkg-config`;
- GMP and NTL development libraries;
- Valgrind with Cachegrind;
- Python 3 with the packages pinned in `requirements.txt`.

Configure LLVM for the Rust bindings. The exact library directory can vary by
distribution; `.envrc.example` contains the tested Fedora layout. A typical
LLVM installation under `/usr/lib/llvm-21` uses:

```bash
export PATH="/usr/lib/llvm-21/bin:$PATH"
export MLIR_SYS_210_PREFIX=/usr/lib/llvm-21
export TABLEGEN_210_PREFIX=/usr/lib/llvm-21
export LIBCLANG_PATH=/usr/lib/llvm-21/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I/usr/lib/llvm-21/lib/clang/21/include"
```

Install the Python dependencies in an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Cargo commands use the committed `Cargo.lock` and pass `--locked`. The first
build downloads Rust dependencies and can take several minutes.

The Docker image also installs the pinned toolchain's `rustfmt` component.
This is required by the pinned Barvinok binding generator, not only for source
formatting.

## Symbolica

SALT uses Symbolica for symbolic polynomial manipulation. No Symbolica
license key is included in this artifact. The evaluation drivers set
`SYMBOLICA_HIDE_BANNER=1` and use Symbolica's free restricted mode, which is
limited to one instance and one core per device. Make sure another unlicensed
Symbolica process is not running on the evaluation machine.

To use a separately obtained license, set `SYMBOLICA_LICENSE` only in the local
environment. Never add a key to this repository or an artifact archive.

## Quick validation

Run reduced experiments first. The commands below use temporary output
directories and do not modify the checked-in benchmark inputs.

```bash
RESULTS_DIR=/tmp/salt-matmul-smoke \
MAX_CACHE_BLOCKS=8 \
./scripts/run-matmul-t0-t2-evaluation.sh
```

```bash
RESULTS_DIR=/tmp/salt-contraction-smoke \
./scripts/run-mlir-contraction-evaluation.sh --smoke-test
```

```bash
RESULTS_DIR=/tmp/salt-hardware-smoke \
./scripts/run-salt-vs-hardware-evaluation.sh --smoke-test
```

The matrix-multiplication check runs all three loop organizations over a small
cache range. The contraction check runs the original and tiled 3D
tensor-vector kernels, all three cache organizations, and both plot variants.
The SALT-vs-hardware check analyzes an original kernel, its tiled form, and the
stencil; it then checks the frozen PMC input and generates the derived CSV and
plot. It should report three benchmarks, MARE approximately `0.0072`, and
Pearson correlation approximately `1.0000`. Fresh PMU collection is
deliberately not part of the portable smoke test.

## Figure 4: reference data or local PMCs

The normal artifact command uses the checked-in measurements from an Intel
Core i7-7700 with hyperthreading disabled. It does not access the evaluator's
hardware counters:

```bash
./scripts/run-salt-vs-hardware-evaluation.sh
```

To perform the optional experiment on the current machine, first choose a
logical CPU whose sibling hyperthread is offline, then collect three repeats:

```bash
mkdir -p results/salt-vs-hardware-local
python3 salt_vs_hw_misses_package/benchmarks/collect_pmc.py \
  --cpu 1 \
  --repeats 3 \
  --output results/salt-vs-hardware-local/pmu.csv
```

Compare SALT with the new measurements through the same wrapper:

```bash
RESULTS_DIR=results/salt-vs-hardware-local \
./scripts/run-salt-vs-hardware-evaluation.sh \
  --pmc results/salt-vs-hardware-local/pmu.csv
```

The defaults model a 32 KiB cache, 64-byte lines, and eight 8-byte elements per
line. If the measured L1D geometry differs, pass the corresponding
`--cache-size-bytes`, `--cache-line-bytes`, and
`--elements-per-cache-line` values. Fresh PMC results are machine-dependent and
are not expected to match the packaged i7-7700 values. PMU permissions, CPU
topology checks, and hyperthread-isolation requirements are detailed in
`salt_vs_hw_misses_package/README.md`.

## Full evaluation

Start with empty output directories. The contraction driver refuses to append
to existing SQLite databases so that results from different configurations
cannot be mixed.

```bash
./scripts/run-mlir-contraction-evaluation.sh
./scripts/run-matmul-t0-t2-evaluation.sh
./scripts/run-salt-vs-hardware-evaluation.sh
```

The full contraction experiment runs the original and tiled forms of eight
kernels. Runtime depends strongly on CPU count and machine load; the
Cachegrind sweeps are the dominant cost.

The matrix-multiplication workflow generates:

- SALT and Cachegrind JSON data for T0, T1, and T2;
- `matmul_t0-t2_salt_cachegrind.svg`;
- `matmul_t0-t2_salt_cachegrind.png`.

The MLIR-contraction workflow generates:

- fully associative, 8-way, and 12-way SQLite databases;
- staged MLIR inputs and SALT JSON under `work/constant/`;
- `miss_count_comparison_all_programs_log.svg`;
- `miss_count_comparison_all_programs_linear.svg`;
- `timing-simulation-wall.tsv`;
- `timing-selected.json` and `timing-selected.svg`.

The SALT-vs-hardware workflow generates:

- SALT JSON for the 17 evaluated kernels under `salt-json/`;
- `salt_vs_hw_misses_results.csv`;
- `salt_vs_hw_misses.svg`.

## Paper claims and generated artifacts

The following table is the evaluator-facing map from the paper to the files
produced by the full workflows. Figure numbering refers to the submitted
paper.

| Paper result | Reproduction command | Generated artifact | Supported claim |
| --- | --- | --- | --- |
| Figure 1 | `./scripts/run-matmul-t0-t2-evaluation.sh` | `results/matmul-t0-t2/matmul_t0-t2_salt_cachegrind.svg` | SALT's predicted cache behavior agrees with Cachegrind for the T0, T1, and T2 matrix-multiplication loop organizations. |
| Figure 2 | `./scripts/run-mlir-contraction-evaluation.sh` | `results/mlir-contractions/miss_count_comparison_all_programs_log.svg` and `miss_count_comparison_all_programs_linear.svg` | SALT miss-count predictions are compared with fully associative, 8-way, and 12-way Cachegrind simulations for the original and tiled MLIR contractions. |
| Figure 3 | `./scripts/run-mlir-contraction-evaluation.sh` | `results/mlir-contractions/timing-selected.svg` with numeric data in `timing-selected.json` and `timing-simulation-wall.tsv` | SALT analysis time is compared with the measured simulation time for the selected contraction kernels. |
| Figure 4 | `./scripts/run-salt-vs-hardware-evaluation.sh` | `results/salt-vs-hardware/salt_vs_hw_misses.svg` with numeric data in `salt_vs_hw_misses_results.csv` | SALT estimates are compared with measured L1D load misses for 17 kernels. The default workflow uses the checked-in Intel Core i7-7700 measurements. |

For Figure 4, the reference data should produce 17 points, MARE approximately
`0.0168`, and Pearson correlation approximately `0.9996` (displayed as `1.000`
in the plot). Small rendering differences do not change these numeric checks.

The artifact supports regeneration of all four figures and their intermediate
numeric data. Fresh collection of Figure 4's hardware counters is not claimed
to be machine-independent: the values depend on the processor, PMU event,
compiler, and system configuration. The checked-in i7-7700 CSV is therefore
the reference input for reproduction. Instructions for an optional fresh
collection, including the CPU-affinity and SMT requirements, are in
`salt_vs_hw_misses_package/README.md`.

Detailed configurations, methodology notes, and individual-stage commands are
in:

- `benchmarks/matmul-t0-t2/README.md`;
- `benchmarks/mlir-contractions/README.md`;
- `salt_vs_hw_misses_package/README.md`.

## Repository layout

```text
analyzer/            Main Barvinok and SALT analysis executable
raffine/             MLIR affine-program extraction library
denning/             Miss-ratio curve construction
cachegrind-runner/   MLIR-to-Cachegrind simulation runner
benchmarks/          Evaluation inputs and experiment-specific documentation
scripts/             End-to-end drivers, measurement tools, and plotters
results/             Generated artifacts (ignored by Git)
```

## Using the analyzer directly

Build the analyzer with the pinned dependency graph:

```bash
cargo build --locked --release -p analyzer --bin analyzer
```

Run SALT on an MLIR input:

```bash
./target/release/analyzer \
  --input benchmarks/examples/sym_heat_center_only.mlir \
  --json \
  --output /tmp/salt-result.json \
  salt \
  --block-size=8
```

Omit `--block-size` to keep the cache-line capacity symbolic. Use
`./target/release/analyzer --help` to list the complete command-line interface.

### Meaning of `--block-size`

Despite its historical name, `--block-size` is measured in target array
elements, not bytes. It specifies how many elements fit in one modeled cache
line:

```text
block size in elements = cache-line bytes / element bytes
```

For the double-precision contraction and hardware experiments, a 64-byte line
contains eight 8-byte `f64` elements, so the scripts use `--block-size=8`. For
the single-precision matrix-multiplication experiment, a 32-byte line contains
eight 4-byte `f32` elements, so that workflow also uses `--block-size=8`.
Cache-size axes and turning points in SALT output are consequently expressed
in numbers of cache lines. When applying SALT to a different element type or
line size, recompute this value rather than passing a byte count.

## Development checks

```bash
cargo build --locked --release
cargo test --locked --release
```
