// 4D Tensor Contraction Kernel
// C[i][j] = A[i][k][l][j] * B[k][l]
// Converted from MLIR constant_4d_tensor.mlir

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: A: 80x48x32x64, B: 48x32, C: 80x64
// simulation.prologue: "volatile double ARRAY_0[83][53][37][88], ARRAY_1[53][40], ARRAY_2[83][88];"
#define DIM_I 80
#define DIM_J 64
#define DIM_K 48
#define DIM_L 32

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 83  // 80 + padding
#define ARRAY_0_DIM1 53  // 48 + padding
#define ARRAY_0_DIM2 37  // 32 + padding
#define ARRAY_0_DIM3 88  // 64 + padding
#define ARRAY_1_DIM0 53  // 48 + padding
#define ARRAY_1_DIM1 40  // 32 + padding
#define ARRAY_2_DIM0 83  // 80 + padding
#define ARRAY_2_DIM1 88  // 64 + padding

static volatile double A[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2][ARRAY_0_DIM3] __attribute__((aligned(64)));
static volatile double B[ARRAY_1_DIM0][ARRAY_1_DIM1] __attribute__((aligned(64)));
static volatile double C[ARRAY_2_DIM0][ARRAY_2_DIM1] __attribute__((aligned(64)));

static void tensor4d_init(void)
{
    for (int i = 0; i < DIM_I; i++) {
        for (int k = 0; k < DIM_K; k++) {
            for (int l = 0; l < DIM_L; l++) {
                for (int j = 0; j < DIM_J; j++) {
                    A[i][k][l][j] = (double)(i * DIM_K * DIM_L * DIM_J + k * DIM_L * DIM_J + l * DIM_J + j);
                }
            }
        }
    }
    
    for (int k = 0; k < DIM_K; k++) {
        for (int l = 0; l < DIM_L; l++) {
            B[k][l] = (double)(k * DIM_L + l);
        }
    }
    
    for (int i = 0; i < DIM_I; i++) {
        for (int j = 0; j < DIM_J; j++) {
            C[i][j] = 0.0;
        }
    }
}

static void tensor4d_execute(void)
{
    // C[i][j] = A[i][k][l][j] * B[k][l]
    for (int i = 0; i < DIM_I; i++) {
        for (int j = 0; j < DIM_J; j++) {
            for (int k = 0; k < DIM_K; k++) {
                for (int l = 0; l < DIM_L; l++) {
                    // double A_iklj = A[i][k][l][j];
                    // double B_kl = B[k][l];
                    // double prod = A_iklj * B_kl;
                    // C[i][j] = prod;
                    C[i][j] += A[i][k][l][j] * B[k][l];
                }
            }
        }
    }
}

static void tensor4d_cleanup(void)
{
    volatile double sink = C[DIM_I - 1][DIM_J - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void tensor4d_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t constant_4d_tensor_kernel = {
    .name = "constant_4d_tensor",
    .description = "4D Tensor contraction (80x48x32x64 * 48x32 -> 80x64)",
    .init = tensor4d_init,
    .execute = tensor4d_execute,
    .cleanup = tensor4d_cleanup,
    .verify = tensor4d_verify
};
