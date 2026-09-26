# The 17 benchmark sources and PMC collector

The portable Figure 4 reproduction and the optional local-measurement procedure
are documented in the parent `README.md`. This file describes the native
benchmark runner and collector themselves.

This directory ships the exact project C implementations for the 16 benchmarks
shown in the supplied miss-count figure, plus the original stencil benchmark:

```text
orig_3d_tensor_vector          tiled_3d_tensor_vector
orig_4d_tensor                 tiled_4d_tensor
orig_attention_score           tiled_attention_score
orig_batched_gemm              tiled_batched_gemm
orig_context_lookup            tiled_context_lookup
orig_matrix_matrix             tiled_matrix_matrix
orig_matrix_vector             tiled_matrix_vector
orig_rowwise_softmax_max       tiled_rowwise_softmax_max
orig_stencil5pt
```

Build and confirm the manifest:

```bash
make
./bin/benchmark_runner --list
```

Collect one measurement per program, pinned to logical CPU 1:

```bash
python3 collect_pmc.py --cpu 1
```

For a less noisy result, collect three runs and retain their median:

```bash
python3 collect_pmc.py --cpu 1 --repeats 3
```

The default output is `../data/pmu_results_17_new.csv`. Initialization and
cleanup are outside the measured interval; only `kernel->execute()` is counted.
Counter semantics and permissions are documented in
`../pmc_measurement/README.md`.

PMC values are machine- and run-dependent. When `--cpu` is supplied, the
collector reads the selected CPU's sibling set from Linux sysfs, temporarily
offlines any online siblings, and restores only the CPUs it changed after the
collection. It writes sysfs directly when permitted and otherwise requests the
narrow `sudo tee` operation. If neither succeeds, it exits before measuring.
Pinning with `taskset` alone is not sufficient: during validation, an
online sibling changed some replacement counts by approximately 3x. After an
unrecoverable interruption such as `SIGKILL` or power loss, verify the sibling
state manually.

Use the same CPU, affinity, compiler flags, frequency/prefetch configuration,
and operating-system conditions when comparing a new collection with the
frozen paper data.
