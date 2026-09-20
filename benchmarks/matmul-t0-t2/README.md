# T0–T2 matrix-multiplication cache experiment

This experiment compares Cachegrind and SALT miss-ratio curves for an untiled
matrix multiplication, one level of tiling, and two levels of tiling.
For the complete evaluator workflow and paper-result map, start with the
repository-root `README.md`.

From the repository root, run:

```bash
scripts/run-matmul-t0-t2-evaluation.sh
```

Results are written to `results/matmul-t0-t2/`.
The driver generates the six JSON inputs expected by
`scripts/graph_mm_salt_vs_cg.py`, followed by SVG, PDF, and PNG figures.

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
| T0 | `matmul.c` | `../examples/const_matmul_3acc.mlir` |
| T1 | `matmul-t1.c` | `../examples/const_matmul_once_tiled.mlir` |
| T2 | `matmul-t2.c` | `../examples/const_matmul_twice_tiled.mlir` |

Both the checked-in C kernels and the constant SALT MLIR use 256×256 matrices.
The T1 tile size is 32 and the T2 hierarchy is 128×32; both divide 256 exactly.

## Cache model and sampling

The artifact driver uses GCC and 32-byte Cachegrind lines. Because the arrays
contain 4-byte `float` values, SALT's `--block-size=8` means eight target
elements per line and models the same 32-byte line.

`parallel_runner.py` supports the sparse geometric default and the optional
dense configuration shown above. Cache capacities in the generated JSON are
expressed in numbers of 32-byte blocks. Do not combine these results with data
generated using a different line size without first converting the x-axis to
bytes.

Evaluators should use
`scripts/run-matmul-t0-t2-evaluation.sh`.
