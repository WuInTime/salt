#ifndef PMC_L1D_MISSES_H
#define PMC_L1D_MISSES_H

#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef struct {
    int fd;
} pmc_l1d_misses_t;

typedef struct {
    uint64_t value;
    uint64_t time_enabled;
    uint64_t time_running;
} pmc_l1d_read_t;

static inline uint64_t pmc_l1d_miss_config(void)
{
    return (uint64_t)PERF_COUNT_HW_CACHE_L1D |
           ((uint64_t)PERF_COUNT_HW_CACHE_OP_READ << 8) |
           ((uint64_t)PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
}

static inline int pmc_l1d_misses_open(pmc_l1d_misses_t *counter)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_HW_CACHE;
    attr.size = sizeof(attr);
    attr.config = pmc_l1d_miss_config();
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                       PERF_FORMAT_TOTAL_TIME_RUNNING;

    counter->fd = (int)syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
    if (counter->fd < 0) {
        fprintf(stderr, "perf_event_open(L1D read misses): %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static inline int pmc_l1d_misses_start(pmc_l1d_misses_t *counter)
{
    if (ioctl(counter->fd, PERF_EVENT_IOC_RESET, 0) == -1 ||
        ioctl(counter->fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {
        fprintf(stderr, "starting L1D miss counter: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

static inline int pmc_l1d_misses_stop(pmc_l1d_misses_t *counter,
                                      uint64_t *scaled_value)
{
    pmc_l1d_read_t result;
    if (ioctl(counter->fd, PERF_EVENT_IOC_DISABLE, 0) == -1) {
        fprintf(stderr, "stopping L1D miss counter: %s\n", strerror(errno));
        return -1;
    }
    if (read(counter->fd, &result, sizeof(result)) != (ssize_t)sizeof(result)) {
        fprintf(stderr, "reading L1D miss counter: %s\n", strerror(errno));
        return -1;
    }
    if (result.time_running == 0) {
        fprintf(stderr, "L1D miss counter never ran\n");
        return -1;
    }

    long double scaled = (long double)result.value;
    if (result.time_running != result.time_enabled) {
        scaled *= (long double)result.time_enabled / result.time_running;
    }
    *scaled_value = (uint64_t)(scaled + 0.5L);
    return 0;
}

static inline void pmc_l1d_misses_close(pmc_l1d_misses_t *counter)
{
    if (counter->fd >= 0) {
        close(counter->fd);
        counter->fd = -1;
    }
}

#endif
