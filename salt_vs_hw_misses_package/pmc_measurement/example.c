#include "pmc_l1d_misses.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ELEMENTS (16U * 1024U * 1024U)

int main(void)
{
    uint64_t *data = malloc(ELEMENTS * sizeof(*data));
    if (data == NULL) {
        perror("malloc");
        return 1;
    }
    for (size_t i = 0; i < ELEMENTS; ++i) {
        data[i] = i;
    }

    pmc_l1d_misses_t counter = {.fd = -1};
    if (pmc_l1d_misses_open(&counter) != 0 ||
        pmc_l1d_misses_start(&counter) != 0) {
        free(data);
        return 1;
    }

    volatile uint64_t sum = 0;
    for (size_t i = 0; i < ELEMENTS; i += 8) {
        sum += data[i];
    }

    uint64_t misses = 0;
    if (pmc_l1d_misses_stop(&counter, &misses) != 0) {
        pmc_l1d_misses_close(&counter);
        free(data);
        return 1;
    }
    pmc_l1d_misses_close(&counter);

    printf("L1D.REPLACEMENT,%" PRIu64 "\n", misses);
    fprintf(stderr, "checksum=%" PRIu64 "\n", sum);
    free(data);
    return 0;
}
