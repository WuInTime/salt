# Standalone L1D read-miss counter

The evaluator-facing Figure 4 workflow is documented in the package's parent
`README.md`; this file covers the low-level counter interface.

`pmc_l1d_misses.h` extracts the hardware event used for the plot's PMC values
from the larger PMU framework. It counts the calling thread only and excludes
kernel and hypervisor activity.

Build and run the example on Linux:

```bash
gcc -O3 -Wall -Wextra -std=c11 example.c -o example
taskset -c 1 ./example
```

The output is a CSV-style line such as:

```text
L1D.load_miss,123456
```

To instrument another workload, include `pmc_l1d_misses.h`, open the counter,
call `pmc_l1d_misses_start()` immediately before the workload, and call
`pmc_l1d_misses_stop()` immediately afterward. Initialization and allocation
should remain outside the measured region, as in `example.c`.

Linux may deny access with `Permission denied` when `perf_event_paranoid` is
restrictive. Use the site's approved perf/PMU access policy (for example, an
administrator may grant `CAP_PERFMON` or adjust `kernel.perf_event_paranoid`).
The generic L1D cache event is also CPU-dependent; an unsupported event will
fail at `perf_event_open` rather than returning a fabricated value.
