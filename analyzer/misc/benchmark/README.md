# T0–T2 matrix-multiplication cache experiment

This experiment compares Cachegrind and SALT miss-ratio curves for an untiled
matrix multiplication, one level of tiling, and two levels of tiling.

From the repository root, run:

```bash
scripts/run-matmul-t0-t2-evaluation.sh
```

Results are written to `analyzer/misc/benchmark/results/matmul-t0-t2/`.
The driver generates the six JSON inputs expected by
`scripts/graph_mm_salt_vs_cg.py`, followed by SVG and PNG figures.

For a quick pipeline check:

```bash
RESULTS_DIR=/tmp/matmul-t0-t2-smoke \
MAX_CACHE_BLOCKS=8 \
scripts/run-matmul-t0-t2-evaluation.sh
```

By default, the driver generates a sparse geometric sweep through 32,768 cache
blocks, augmented with powers of two. No cache-size list needs to be maintained
manually. To run every cache size through 1,024 blocks instead, use:

```bash
CACHE_SAMPLING=dense MAX_CACHE_BLOCKS=1024 \
scripts/run-matmul-t0-t2-evaluation.sh
```

`CACHE_STEP` controls the dense sweep. `CACHE_GROWTH_FACTOR` controls the sparse
geometric sweep and defaults to 1.5.

## Input mapping

| Variant | Cachegrind C | SALT MLIR |
| --- | --- | --- |
| T0 | `matmul.c` | `../const_matmul_3acc.mlir` |
| T1 | `matmul-t1.c` | `../const_matmul_once_tiled.mlir` |
| T2 | `matmul-t2.c` | `../const_matmul_twice_tiled.mlir` |

Both the checked-in C kernels and the constant SALT MLIR use 256×256 matrices.
The T1 tile size is 32 and the T2 hierarchy is 128×32; both divide 256 exactly.

The original six JSON files and optional `tmp/mr_t*.json` files are missing.
`parallel_runner.py` supports both the original dense sampling pattern and a
sparse geometric pattern. The driver uses GCC and 32-byte lines. The later
`og.py` appears to be the supplemental generator for `tmp/mr_t*.json`: GCC,
64-byte lines, and geometrically increasing block counts. Its preserved
command examples stop at 8192 blocks.

`og.py` cannot currently execute because a LaTeX paragraph was pasted directly
after its Python code. More importantly, merging results expressed in numbers
of blocks while changing the line size from 32 to 64 bytes mixes different
byte capacities. This needs clarification before presenting combined accuracy
metrics.

The default artifact driver performs a sparse sweep intended for plotting. It
must not be described as the original dense 2--1024 accuracy sweep; select
`CACHE_SAMPLING=dense MAX_CACHE_BLOCKS=1024` when reproducing that experiment.

Cachegrind uses 32-byte lines because the C arrays contain `float`. SALT uses a
block size of eight `f32` elements, also 32 bytes.
