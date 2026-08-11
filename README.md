# AutoLALA

AutoLALA is an automatic asymptotic locality analyzer for affine loop programs.
This repository contains the Rust implementation, benchmark inputs, and two
drivers that reproduce the evaluation artifacts.

## Artifact overview

The artifact has two evaluation workflows. These scripts build the required
binaries, run the analyses and Cachegrind simulations, and generate the plots
and intermediate data used by the evaluation:

| Workflow | Command | Output directory |
| --- | --- | --- |
| Matrix multiplication T0-T2 | `./scripts/run-matmul-t0-t2-evaluation.sh` | `results/matmul-t0-t2/` |
| MLIR contractions | `./scripts/run-mlir-contraction-evaluation.sh` | `results/mlir-contractions/` |

Generated results and Cargo build products are intentionally excluded from
version control. Each workflow can write somewhere else by setting
`RESULTS_DIR`.

## Requirements

The direct, non-containerized workflow requires:

- Linux on an x86-64 machine;
- Rustup and Cargo (the pinned toolchain is in `rust-toolchain`);
- LLVM/MLIR 21 development libraries and tools;
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

## Symbolica

AutoLALA uses Symbolica for symbolic polynomial manipulation. No Symbolica
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
RESULTS_DIR=/tmp/autolala-matmul-smoke \
MAX_CACHE_BLOCKS=8 \
./scripts/run-matmul-t0-t2-evaluation.sh
```

```bash
RESULTS_DIR=/tmp/autolala-contraction-smoke \
./scripts/run-mlir-contraction-evaluation.sh --smoke-test
```

The matrix-multiplication check runs all three loop organizations over a small
cache range. The contraction check runs the original and tiled 3D
tensor-vector kernels, all three cache organizations, and both plot variants.

## Full evaluation

Start with empty output directories. The contraction driver refuses to append
to existing SQLite databases so that results from different configurations
cannot be mixed.

```bash
./scripts/run-matmul-t0-t2-evaluation.sh
./scripts/run-mlir-contraction-evaluation.sh
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

Detailed configurations, methodology notes, and individual-stage commands are
in:

- `benchmarks/matmul-t0-t2/README.md`;
- `benchmarks/mlir-contractions/README.md`.

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

Omit `--block-size` to keep the cache block size symbolic. Use
`./target/release/analyzer --help` to list the complete command-line interface.

## Development checks

```bash
cargo build --locked --release
cargo test --locked --release
```
