# Standalone generic L1D event counter

The evaluator-facing Figure 4 workflow is documented in the package's parent
`README.md`; this file covers the low-level counter interface.

`pmc_l1d_misses.h` requests Linux's generic `L1-dcache-load-misses` event, using
`PERF_TYPE_HW_CACHE` with the L1D/read/miss selector. On the Intel Core i7-7700,
Core i7-6700, and Xeon Gold 6126 systems validated for this artifact, Linux maps
that selector to `L1D.REPLACEMENT` (raw event `0x51`, umask `0x01`). It is not
the same event as `MEM_LOAD_RETIRED.L1_MISS` (raw event `0xd1`, umask `0x08`).
The historical `L1D.load_miss` and `csv_l1d_load_miss` labels are retained for
compatibility with the packaged data and plotting workflow.

The counter counts the calling thread only and excludes kernel and hypervisor
activity.

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
The generic event's mapping is CPU- and kernel-dependent; an unsupported event
will fail at `perf_event_open` rather than returning a fabricated value. Check
the local mapping with `perf list --details` before interpreting a fresh
collection. Do not mix values from the generic replacement event with values
from the raw retired-load event in one comparison.
