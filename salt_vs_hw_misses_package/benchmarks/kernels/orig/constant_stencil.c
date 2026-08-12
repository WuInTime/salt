// Baseline 5-point stencil kernel
// B[i][j] = A[i][j] + A[i - 1][j] + A[i + 1][j] + A[i][j - 1] + A[i][j + 1]

#include "../../include/kernel_interface.h"
#include <stdio.h>

#define DIM_N 512
#define DIM_M 512

// Padding to keep the 512-wide rows from mapping to the same L1d sets.
#define ARRAY_0_DIM0 541
#define ARRAY_0_DIM1 536
#define ARRAY_1_DIM0 541
#define ARRAY_1_DIM1 536
#define padding DIM_N+8 // 512 + padding to break 4 KiB strides that map to the same L1d sets

static volatile double A[DIM_N][DIM_M] __attribute__((aligned(64)));
static volatile double B[DIM_N][padding] __attribute__((aligned(64)));

// static volatile double B[DIM_N][DIM_M] __attribute__((aligned(64)));
// static volatile double A[DIM_N][padding] __attribute__((aligned(64)));

static void constant_stencil_init(void)
{
    for (int i = 0; i < DIM_N; i++) {
        for (int j = 0; j < DIM_M; j++) {
            A[i][j] = (double)(i * DIM_M + j);
            B[i][j] = 0.0;
        }
    }
}

static void constant_stencil_execute(void)
{
    for (int i = 1; i < DIM_N - 1; i++) {
        for (int j = 1; j < DIM_M - 1; j++) {
            B[i][j] +=
            A[i + 1][j] +
            A[i][j - 1] +
            A[i][j] +
            A[i][j + 1] +
            A[i - 1][j]
            ;
        }
    }
}

static void constant_stencil_cleanup(void)
{
    volatile double sink = B[DIM_N - 2][DIM_M - 2];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void constant_stencil_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t constant_stencil_kernel = {
    .name = "constant_stencil",
    .description = "Baseline 5-point stencil (512x512)",
    .init = constant_stencil_init,
    .execute = constant_stencil_execute,
    .cleanup = constant_stencil_cleanup,
    .verify = constant_stencil_verify
};
