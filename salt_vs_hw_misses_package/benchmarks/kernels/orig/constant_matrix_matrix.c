// Matrix-Matrix Multiplication Kernel
// C[i][k] = A[i][j] * B[j][k]
// Converted from MLIR constant_matrix_matrix.mlir

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: A: 256x160, B: 160x192, C: 256x192
// simulation.prologue: "volatile double ARRAY_0[257][184], ARRAY_1[167][248], ARRAY_2[257][248];"
#define DIM_I 256
#define DIM_J 160
#define DIM_K 192

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 257  // 256 + padding
#define ARRAY_0_DIM1 184  // 160 + padding
#define ARRAY_1_DIM0 167  // 160 + padding
#define ARRAY_1_DIM1 248  // 192 + padding
#define ARRAY_2_DIM0 257  // 256 + padding
#define ARRAY_2_DIM1 248  // 192 + padding

static volatile double A[ARRAY_0_DIM0][ARRAY_0_DIM1] __attribute__((aligned(64)));
static volatile double B[ARRAY_1_DIM0][ARRAY_1_DIM1] __attribute__((aligned(64)));
static volatile double C[ARRAY_2_DIM0][ARRAY_2_DIM1] __attribute__((aligned(64)));

static void matrix_matrix_init(void)
{
    for (int i = 0; i < DIM_I; i++) {
        for (int j = 0; j < DIM_J; j++) {
            A[i][j] = (double)(i * DIM_J + j);
        }
    }
    
    for (int j = 0; j < DIM_J; j++) {
        for (int k = 0; k < DIM_K; k++) {
            B[j][k] = (double)(j * DIM_K + k);
        }
    }
    
    for (int i = 0; i < DIM_I; i++) {
        for (int k = 0; k < DIM_K; k++) {
            C[i][k] = 0.0;
        }
    }
}

static void matrix_matrix_execute(void)
{
    // C[i][k] = A[i][j] * B[j][k]
    for (int i = 0; i < DIM_I; i++) {
        for (int k = 0; k < DIM_K; k++) {
            for (int j = 0; j < DIM_J; j++) {
                // double A_ij = A[i][j];
                // double B_jk = B[j][k];
                // double prod = A_ij * B_jk;
                // C[i][k] = prod;
                C[i][k] += A[i][j] * B[j][k];
            }
        }
    }
}

static void matrix_matrix_cleanup(void)
{
    volatile double sink = C[DIM_I - 1][DIM_K - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void matrix_matrix_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t constant_matrix_matrix_kernel = {
    .name = "constant_matrix_matrix",
    .description = "Matrix-Matrix multiplication (256x160 * 160x192 -> 256x192)",
    .init = matrix_matrix_init,
    .execute = matrix_matrix_execute,
    .cleanup = matrix_matrix_cleanup,
    .verify = matrix_matrix_verify
};
