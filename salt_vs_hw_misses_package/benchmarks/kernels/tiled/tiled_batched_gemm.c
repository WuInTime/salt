// Tiled Batched GEMM (General Matrix Multiply) Kernel
// C[p][i][j] = A[p][i][k] * B[p][k][j] with tiling
// Converted from MLIR tiled_batched_gemm.mlir

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: A: 32x64x128, B: 32x128x96, C: 32x64x96
// simulation.prologue: "volatile double ARRAY_0[37][67][136], ARRAY_1[37][131][104], ARRAY_2[37][67][104];"
#define DIM_P 32   // Batch size
#define DIM_I 64
#define DIM_J 96
#define DIM_K 128

// Tiling parameters from MLIR
// affine.for %arg3 = 0 to 4, %arg4 = 0 to 8, %arg5 = 0 to 12, %arg6 = 0 to 16
// inner: %arg7, %arg8, %arg9, %arg10 = 0 to 8
// access: A[%arg7 + %arg3 * 8, %arg8 + %arg4 * 8, %arg10 + %arg6 * 8]
#define TILE_SIZE 8

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

static void tiled_batched_gemm_init(void)
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

static void tiled_batched_gemm_execute(void)
{
    // Tiled loops from MLIR
    // for %arg3 = 0 to 4 (32/8), %arg4 = 0 to 8 (64/8), %arg5 = 0 to 12 (96/8), %arg6 = 0 to 16 (128/8)
    // inner: for %arg7, %arg8, %arg9, %arg10 = 0 to 8
    for (int arg3 = 0; arg3 < 4; arg3++) {
        for (int arg4 = 0; arg4 < 8; arg4++) {
            for (int arg5 = 0; arg5 < 12; arg5++) {
                for (int arg6 = 0; arg6 < 16; arg6++) {
                    for (int arg7 = 0; arg7 < 8; arg7++) {
                        for (int arg8 = 0; arg8 < 8; arg8++) {
                            for (int arg9 = 0; arg9 < 8; arg9++) {
                                for (int arg10 = 0; arg10 < 8; arg10++) {
                                    // int p = arg7 + arg3 * 8;
                                    // int i = arg8 + arg4 * 8;
                                    // int j = arg9 + arg5 * 8;
                                    // int k = arg10 + arg6 * 8;
                                    // double A_pik = A[p][i][k];
                                    // double B_pkj = B[p][k][j];
                                    // double prod = A_pik * B_pkj;
                                    // C[p][i][j] = prod;
                                    C[arg7 + arg3 * 8][arg8 + arg4 * 8][arg9 + arg5 * 8] += A[arg7 + arg3 * 8][arg8 + arg4 * 8][arg10 + arg6 * 8] * B[arg7 + arg3 * 8][arg10 + arg6 * 8][arg9 + arg5 * 8];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void tiled_batched_gemm_cleanup(void)
{
    volatile double sink = C[DIM_P - 1][DIM_I - 1][DIM_J - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void tiled_batched_gemm_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t tiled_batched_gemm_kernel = {
    .name = "tiled_batched_gemm",
    .description = "Tiled Batched GEMM (32x64x128 * 32x128x96 -> 32x64x96)",
    .init = tiled_batched_gemm_init,
    .execute = tiled_batched_gemm_execute,
    .cleanup = tiled_batched_gemm_cleanup,
    .verify = tiled_batched_gemm_verify
};
