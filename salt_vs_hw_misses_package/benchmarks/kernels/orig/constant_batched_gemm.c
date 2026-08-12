// Batched GEMM (General Matrix Multiply) Kernel
// C[p][i][j] = A[p][i][k] * B[p][k][j]
// Converted from MLIR constant_batched_gemm.mlir

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: A: 32x64x128, B: 32x128x96, C: 32x64x96
// simulation.prologue: "volatile double ARRAY_0[37][67][136], ARRAY_1[37][131][104], ARRAY_2[37][67][104];"
#define DIM_P 32   // Batch size
#define DIM_I 64
#define DIM_J 96
#define DIM_K 128

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 37   // 32 + padding
#define ARRAY_0_DIM1 67   // 64 + padding
#define ARRAY_0_DIM2 136  // 128 + padding
#define ARRAY_1_DIM0 37   // 32 + padding
#define ARRAY_1_DIM1 131  // 128 + padding
#define ARRAY_1_DIM2 104  // 96 + padding
#define ARRAY_2_DIM0 37   // 32 + padding
#define ARRAY_2_DIM1 67   // 64 + padding
#define ARRAY_2_DIM2 104  // 96 + padding

static volatile double A[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2] __attribute__((aligned(64)));
static volatile double B[ARRAY_1_DIM0][ARRAY_1_DIM1][ARRAY_1_DIM2] __attribute__((aligned(64)));
static volatile double C[ARRAY_2_DIM0][ARRAY_2_DIM1][ARRAY_2_DIM2] __attribute__((aligned(64)));

static void batched_gemm_init(void)
{
    for (int p = 0; p < DIM_P; p++) {
        for (int i = 0; i < DIM_I; i++) {
            for (int k = 0; k < DIM_K; k++) {
                A[p][i][k] = (double)(p * DIM_I * DIM_K + i * DIM_K + k);
            }
        }
    }
    
    for (int p = 0; p < DIM_P; p++) {
        for (int k = 0; k < DIM_K; k++) {
            for (int j = 0; j < DIM_J; j++) {
                B[p][k][j] = (double)(p * DIM_K * DIM_J + k * DIM_J + j);
            }
        }
    }
    
    for (int p = 0; p < DIM_P; p++) {
        for (int i = 0; i < DIM_I; i++) {
            for (int j = 0; j < DIM_J; j++) {
                C[p][i][j] = 0.0;
            }
        }
    }
}

static void batched_gemm_execute(void)
{
    // C[p][i][j] = A[p][i][k] * B[p][k][j]
    for (int p = 0; p < DIM_P; p++) {
        for (int i = 0; i < DIM_I; i++) {
            for (int j = 0; j < DIM_J; j++) {
                for (int k = 0; k < DIM_K; k++) {
                    // double A_pik = A[p][i][k];
                    // double B_pkj = B[p][k][j];
                    // double prod = A_pik * B_pkj;
                    // C[p][i][j] = prod;
                    C[p][i][j] += A[p][i][k] * B[p][k][j];
                }
            }
        }
    }
}

static void batched_gemm_cleanup(void)
{
    volatile double sink = C[DIM_P - 1][DIM_I - 1][DIM_J - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void batched_gemm_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t constant_batched_gemm_kernel = {
    .name = "constant_batched_gemm",
    .description = "Batched GEMM (32x64x128 * 32x128x96 -> 32x64x96)",
    .init = batched_gemm_init,
    .execute = batched_gemm_execute,
    .cleanup = batched_gemm_cleanup,
    .verify = batched_gemm_verify
};
