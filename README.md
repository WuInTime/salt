# SALT

SALT is an automatic asymptotic locality analyzer for affine loop programs. This repository contains the Rust implementation, benchmark inputs, and three workflows that reproduce the paper's results.

SALT is distributed under the MIT License; see `LICENSE`.

## Release

- Version: `1.1`
- Zenodo DOI: [10.5281/zenodo.21884197](https://doi.org/10.5281/zenodo.21884197)
- PACT 2026 artifact evaluation: Available, Functional, and Reproduced
- [PACT 2026 Trovi record](https://trovi.chameleoncloud.org/dashboard/artifacts/fa2befc9-24de-442a-9fde-2141e0502eab)

## Artifact overview

The artifact has three evaluation workflows. These scripts build the required binaries, run the analyses and Cachegrind simulations, and generate the plots and intermediate data:

| Workflow | Command | Output directory |
| --- | --- | --- |
| Matrix multiplication T0-T2 | `./scripts/run-matmul-t0-t2-evaluation.sh` | `results/matmul-t0-t2/` |
| MLIR contractions | `./scripts/run-mlir-contraction-evaluation.sh` | `results/mlir-contractions/` |
| SALT vs. hardware L1D misses | `./scripts/run-salt-vs-hardware-evaluation.sh` | `results/salt-vs-hardware/` |

Generated results and Cargo build products are intentionally excluded from version control. Each workflow can write somewhere else by setting `RESULTS_DIR`.

Documentation is organized by audience:

| Document | Purpose |
| --- | --- |
| This README | Setup, smoke/full commands, paper-result mapping, and the optional local-PMC path |
| `benchmarks/matmul-t0-t2/README.md` | Figure 1 inputs and cache-sweep configuration |
| `benchmarks/mlir-contractions/README.md` | Figures 2–3 methodology, outputs, and limitations |
| `salt_vs_hw_misses_package/README.md` | Figure 4 data processing and detailed fresh-measurement procedure |
| `salt_vs_hw_misses_package/benchmarks/README.md` | The 17 native kernels and PMC collector behavior |
| `salt_vs_hw_misses_package/pmc_measurement/README.md` | Low-level Linux PMU interface and permissions |

## Hardware and resource requirements

For portable reproduction using the checked-in reference measurements:

- Architecture: x86-64
- CPU: no particular model is required
- Memory: 8 GiB minimum; 16 GiB recommended
- Free disk space: approximately 25 GiB for the Docker build and 1 GiB for the complete results
- GPU: not required
- Network: required only during the initial Docker build to download dependencies

The reference Figure 4 hardware-counter data were collected on an Intel Core i7-7700. Reproducing those exact measurements does not require an i7-7700 because the reference measurements are included in the artifact. Optional collection of fresh hardware-counter data has additional machine-specific requirements described under [Figure 4: reference data or local PMCs](#figure-4-reference-data-or-local-pmcs).

## Recommended Docker workflow

Docker provides the recommended reproducibility environment. The image uses Ubuntu 24.04, the official LLVM/MLIR 21 packages, the pinned Rust toolchain, and the Python versions in `requirements.txt`.

Build the image from the repository root:

```bash
docker build --progress=plain -t salt-artifact .
```

Use `--no-cache` to verify a completely clean build.

Check the toolchain and run the Rust test suite:

```bash
docker run --rm salt-artifact test
```

The `test` command isolates analyzer unit tests into separate processes so the suite also works with Symbolica's free restricted mode.

Run all three reduced evaluation workflows and retain their results on the host:

```bash
mkdir artifact-results
docker run --rm --init \
  -v "$PWD/artifact-results:/artifact/results" \
  salt-artifact smoke
```

The mounted directory must not already contain `contraction-smoke/`, `matmul-smoke/`, or `salt-vs-hardware-smoke/`; this prevents results from different runs from being mixed.

After validating the smoke run, start the complete evaluation with an empty directory:

```bash
mkdir full-results
docker run --rm --init \
  -v "$PWD/full-results:/artifact/results" \
  salt-artifact reproduce
```

If a separately obtained Symbolica license is required, pass the existing shell variable at runtime rather than storing it in the image:

```bash
docker run --rm --init \
  -e SYMBOLICA_LICENSE \
  -v "$PWD/artifact-results:/artifact/results" \
  salt-artifact smoke
```

The container records its tool versions in `environment-smoke.txt` or `environment-full.txt` alongside the results. Run `docker run --rm salt-artifact help` for all container commands.

## Typical runtimes

The following approximate wall-clock times are based on warm runs on an Intel Core i7-7700 with four cores, eight hardware threads, and 32 GiB of memory:

| Phase | Approximate time |
| --- | ---: |
| Initial `docker build` | Budget 1–2 hours; not measured in this run |
| `docker run --rm salt-artifact test` | 1.5 minutes |
| `docker run ... salt-artifact smoke` | 30 seconds |
| Full matrix-multiplication workflow | 1.5 minutes |
| Full contraction workflow | Approximately 8 hours (rough estimate from a recorded run) |
| Figure 4 processing with the included measurements | Less than 5 seconds |
| Complete `reproduce` workflow, excluding image build | Approximately 8–9 hours |

These times vary substantially with CPU performance, available hardware threads, system load, network speed, and Docker cache state. The initial image build downloads the complete toolchain and compiles the Rust workspace. During `reproduce`, the contraction Cachegrind sweeps are the only multi-hour phase and dominate the overall runtime; the other two workflows should finish within a few minutes. Fresh Figure 4 hardware-counter collection is optional and is not included in these estimates.

## Requirements

The direct, non-containerized workflow requires:

- Linux on an x86-64 machine;
- Rustup and Cargo (the pinned toolchain is in `rust-toolchain`);
- LLVM/MLIR 21 development libraries and tools, including Polly;
- Clang/Clang++ 21, LLD, GCC/G++, and a C/C++ build toolchain;
- CMake, Autoconf, Automake, Libtool, and `pkg-config`;
- GMP and NTL development libraries;
- Valgrind with Cachegrind;
- Python 3.12 or newer with the packages pinned in `requirements.txt`.

Configure LLVM for the Rust bindings. The exact library directory can vary by distribution; `.envrc.example` contains the tested Fedora layout. A typical LLVM installation under `/usr/lib/llvm-21` uses:

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

Cargo commands use the committed `Cargo.lock` and pass `--locked`. The first build downloads Rust dependencies and can take several minutes.

The Docker image also installs the pinned toolchain's `rustfmt` component. This is required by the pinned Barvinok binding generator, not only for source formatting.

## Symbolica

SALT uses Symbolica for symbolic polynomial manipulation. No Symbolica license key is included in this artifact. The workflow scripts set `SYMBOLICA_HIDE_BANNER=1` and use Symbolica's free restricted mode, which is limited to one instance and one core per device. Make sure another unlicensed Symbolica process is not running on the host.

To use a separately obtained license, set `SYMBOLICA_LICENSE` only in the local environment. Never add a key to this repository or an artifact archive.

## Quick validation

Run reduced experiments first. The commands below use temporary output directories and do not modify the checked-in benchmark inputs.

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

The matrix-multiplication check runs all three loop organizations over a small cache range. The contraction check runs the original and tiled 3D tensor-vector kernels, all three cache organizations, and both plot variants. The SALT-vs-hardware check analyzes an original kernel, its tiled form, and the stencil; it then checks the frozen PMC input and generates the derived CSV and plot. It should report three benchmarks, MAPE approximately `0.72%` (equivalently, MARE `0.0072`), and Pearson correlation approximately `1.0000`. Fresh PMU collection is deliberately not part of the portable smoke test.

The smoke workflow evaluates only three representative hardware-comparison benchmarks, so its MAPE (approximately `0.72%`) is not expected to match the full 17-benchmark Figure 4 MAPE (approximately `1.68%`).

## Reproduction criteria

A reproduction is considered successful when:

1. `docker run --rm salt-artifact test` completes successfully.
2. The `smoke` workflow completes without error and produces the three reduced result directories.
3. The complete workflows generate the artifacts listed under [Paper claims and generated artifacts](#paper-claims-and-generated-artifacts).
4. For the packaged Figure 4 reference data, the full workflow reports 17 benchmarks, MAPE approximately `1.68%` (equivalently, MARE `0.0168`), and Pearson correlation approximately `0.9996`.
5. Small rendering differences in generated plots are acceptable if the underlying numeric results match the checks above.
6. Absolute timing values are machine- and load-dependent and are not expected to match the reference measurements exactly. Figure 3 reproduction should preserve the reported qualitative comparison rather than identical wall-clock times.
7. Fresh PMU measurements are machine-dependent and are not expected to match the packaged Intel Core i7-7700 measurements. The checked-in measurements are the reference input for portable reproduction.

## Figure 4: reference data or local PMCs

The normal artifact command uses the checked-in measurements from an Intel Core i7-7700 with hyperthreading disabled. It does not access the current machine's hardware counters:

```bash
./scripts/run-salt-vs-hardware-evaluation.sh
```

To perform the optional experiment on the current machine, first choose a logical CPU whose sibling hyperthread is offline, then collect three repeats:

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

The defaults model a 32 KiB cache, 64-byte lines, and eight 8-byte elements per line. If the measured L1D geometry differs, pass the corresponding `--cache-size-bytes`, `--cache-line-bytes`, and `--elements-per-cache-line` values. Fresh PMC results are machine-dependent and are not expected to match the packaged i7-7700 values. PMU permissions, CPU topology checks, and hyperthread-isolation requirements are detailed in `salt_vs_hw_misses_package/README.md`.

## Full evaluation

Start with empty output directories. The contraction driver refuses to append to existing SQLite databases so that results from different configurations cannot be mixed.

```bash
./scripts/run-mlir-contraction-evaluation.sh
./scripts/run-matmul-t0-t2-evaluation.sh
./scripts/run-salt-vs-hardware-evaluation.sh
```

The full contraction experiment runs the original and tiled forms of eight kernels. Runtime depends strongly on CPU count and machine load; the Cachegrind sweeps are the dominant cost.

The matrix-multiplication workflow generates:

- SALT and Cachegrind JSON data for T0, T1, and T2;
- `matmul_t0-t2_salt_cachegrind.svg`;
- `matmul_t0-t2_salt_cachegrind.pdf`;
- `matmul_t0-t2_salt_cachegrind.png`.

The MLIR-contraction workflow generates:

- fully associative, 8-way, and 12-way SQLite databases;
- staged MLIR inputs and SALT JSON under `work/constant/`;
- `miss_count_comparison_all_programs_log.svg`;
- `miss_count_comparison_all_programs_log.pdf`;
- `miss_count_comparison_all_programs_linear.svg`;
- `miss_count_comparison_all_programs_linear.pdf`;
- `timing-simulation-wall.tsv`;
- `timing-selected.json`, `timing-selected.svg`, and `timing-selected.pdf`.

The SALT-vs-hardware workflow generates:

- SALT JSON for the 17 evaluated kernels under `salt-json/`;
- `salt_vs_hw_misses_results.csv`;
- `salt_vs_hw_misses.svg`;
- `salt_vs_hw_misses.pdf`.

## Paper claims and generated artifacts

The following table maps the paper's figures to the files produced by the full workflows.

| Paper result | Reproduction command | Generated artifact | Supported claim |
| --- | --- | --- | --- |
| Figure 1 | `./scripts/run-matmul-t0-t2-evaluation.sh` | `results/matmul-t0-t2/matmul_t0-t2_salt_cachegrind.{svg,pdf,png}` | SALT's predicted cache behavior agrees with Cachegrind for the T0, T1, and T2 matrix-multiplication loop organizations. |
| Figure 2 | `./scripts/run-mlir-contraction-evaluation.sh` | `results/mlir-contractions/miss_count_comparison_all_programs_{log,linear}.{svg,pdf}` | SALT miss-count predictions are compared with fully associative, 8-way, and 12-way Cachegrind simulations for the original and tiled MLIR contractions. |
| Figure 3 | `./scripts/run-mlir-contraction-evaluation.sh` | `results/mlir-contractions/timing-selected.{svg,pdf}` with numeric data in `timing-selected.json` and `timing-simulation-wall.tsv` | SALT analysis time is compared with the measured simulation time for the selected contraction kernels. |
| Figure 4 | `./scripts/run-salt-vs-hardware-evaluation.sh` | `results/salt-vs-hardware/salt_vs_hw_misses.{svg,pdf}` with numeric data in `salt_vs_hw_misses_results.csv` | SALT estimates are compared with measured L1D load misses for 17 kernels. The default workflow uses the checked-in Intel Core i7-7700 measurements. |

For Figure 4, the reference data should produce 17 points, MAPE approximately `1.68%` (equivalently, MARE `0.0168`), and Pearson correlation approximately `0.9996`. These values are calculated from the 17-row reference dataset shipped in this release. The unrounded MARE is `0.016824`, or MAPE `1.6824%`, consistent with the `0.0170` MARE shown in Figure 4. Small rendering differences do not change these numeric checks.

The artifact supports regeneration of all four figures and their intermediate numeric data. Fresh collection of Figure 4's hardware counters is not claimed to be machine-independent: the values depend on the processor, PMU event, compiler, and system configuration. The checked-in i7-7700 CSV is therefore the reference input for reproduction. Instructions for an optional fresh collection, including the CPU-affinity and SMT requirements, are in `salt_vs_hw_misses_package/README.md`.

Detailed configurations, methodology notes, and individual-stage commands are in:

- `benchmarks/matmul-t0-t2/README.md`;
- `benchmarks/mlir-contractions/README.md`;
- `salt_vs_hw_misses_package/README.md`.

## Troubleshooting

- **Docker build fails while downloading dependencies.** Verify network access and retry the build. Use `--no-cache` when validating a clean build; cached rebuilds are appropriate during ordinary use.
- **Symbolica reports another unlicensed instance or cannot start in restricted mode.** Stop other unlicensed Symbolica processes on the machine. The free restricted mode permits only one instance and one core per device.
- **A workflow reports that an output directory or SQLite database already exists.** Use a new empty output directory. Existing outputs are rejected intentionally to prevent data from different runs from being mixed.
- **Fresh PMC collection fails because of permissions or CPU-topology checks.** The portable/reference evaluation does not require access to hardware counters. Use the checked-in measurements, or consult `salt_vs_hw_misses_package/README.md` for the additional requirements of optional fresh collection.
- **Results differ only in absolute runtime.** Timing is sensitive to CPU performance, system load, and available hardware threads. Compare the generated numeric outputs and qualitative timing relationship rather than requiring identical wall-clock values.

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

Omit `--block-size` to keep the cache-line capacity symbolic. Use `./target/release/analyzer --help` to list the complete command-line interface.

### Meaning of `--block-size`

Despite its historical name, `--block-size` is measured in target array elements, not bytes. It specifies how many elements fit in one modeled cache line:

```text
block size in elements = cache-line bytes / element bytes
```

For the double-precision contraction and hardware experiments, a 64-byte line contains eight 8-byte `f64` elements, so the scripts use `--block-size=8`. For the single-precision matrix-multiplication experiment, a 32-byte line contains eight 4-byte `f32` elements, so that workflow also uses `--block-size=8`. Cache-size axes and turning points in SALT output are consequently expressed in numbers of cache lines. When applying SALT to a different element type or line size, recompute this value rather than passing a byte count.

## Development checks

```bash
cargo build --locked --release
cargo test --locked --release
```
