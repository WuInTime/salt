// Tiled Row-wise Softmax Max Computation Kernel
// M[b][h][q] = fmaxf(M[b][h][q], S[b][h][q][k]) with tiling
// Converted from MLIR tiled_rowwise_softmax_max.mlir
// First step of softmax: finding max values for numerical stability

#include "../../include/kernel_interface.h"
#include <stdio.h>
#include <math.h>

// Array dimensions from MLIR: S: 4x16x512x512, M: 4x16x512
// simulation.prologue: "volatile double ARRAY_0[5][17][541][536], ARRAY_1[5][17][536];"
#define DIM_B 4    // Batch size
#define DIM_H 16   // Number of attention heads
#define DIM_S_Q 512 // Query sequence length
#define DIM_S_K 512 // Key sequence length

// Tiling parameters from MLIR
// affine.for %arg2 = 0 to 2, %arg3 = 0 to 64, %arg4 = 0 to 64
// inner: %arg5, %arg6, %arg7, %arg8 = 0 to 4, 8, 8, 8
// access: S[%arg5, %arg6 + %arg2 * 8, %arg7 + %arg3 * 8, %arg8 + %arg4 * 8]
#define TILE_SIZE 8

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 5    // 4 + padding (S)
#define ARRAY_0_DIM1 17   // 16 + padding
#define ARRAY_0_DIM2 541  // 512 + padding
#define ARRAY_0_DIM3 536  // 512 + padding
#define ARRAY_1_DIM0 5    // 4 + padding (M)
#define ARRAY_1_DIM1 17   // 16 + padding
#define ARRAY_1_DIM2 536  // 512 + padding

// static volatile double S[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2][ARRAY_0_DIM3] __attribute__((aligned(64)));
// static volatile double M[ARRAY_1_DIM0][ARRAY_1_DIM1][ARRAY_1_DIM2] __attribute__((aligned(64)));

static volatile double S[DIM_B][DIM_H][DIM_S_Q][DIM_S_K+8] __attribute__((aligned(64)));
static volatile double M[DIM_B][DIM_H][DIM_S_Q+8] __attribute__((aligned(64)));

static void tiled_rowwise_softmax_max_init(void)
{
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int q = 0; q < DIM_S_Q; q++) {
                for (int k = 0; k < DIM_S_K; k++) {
                    S[b][h][q][k] = (double)(b * DIM_H * DIM_S_Q * DIM_S_K + h * DIM_S_Q * DIM_S_K + q * DIM_S_K + k);
                }
            }
        }
    }
    
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int q = 0; q < DIM_S_Q; q++) {
                M[b][h][q] = 0.0f;
            }
        }
    }
}

static void tiled_rowwise_softmax_max_execute(void)
{
    // Tiled loops from MLIR
    // for %arg2 = 0 to 2 (16/8), %arg3 = 0 to 64 (512/8), %arg4 = 0 to 64 (512/8)
    // inner: for %arg5, %arg6, %arg7, %arg8 = 0 to 4, 8, 8, 8
    // double cst = 0.0f;
    for (int arg2 = 0; arg2 < 2; arg2++) {
        for (int arg3 = 0; arg3 < 64; arg3++) {
            for (int arg4 = 0; arg4 < 64; arg4++) {
                for (int arg5 = 0; arg5 < 4; arg5++) {
                    for (int arg6 = 0; arg6 < 8; arg6++) {
                        for (int arg7 = 0; arg7 < 8; arg7++) {
                            for (int arg8 = 0; arg8 < 8; arg8++) {
                                // int b = arg5;
                                // int h = arg6 + arg2 * 8;
                                // int q = arg7 + arg3 * 8;
                                // int k = arg8 + arg4 * 8;
                                // double score_val = S[b][h][q][k];
                                // double new_max = fmaxf(score_val, cst);
                                // M[b][h][q] = new_max;
                                M[arg5][arg6 + arg2 * 8][arg7 + arg3 * 8] = fmaxf(S[arg5][arg6 + arg2 * 8][arg7 + arg3 * 8][arg8 + arg4 * 8], M[arg5][arg6 + arg2 * 8][arg7 + arg3 * 8]);
                            }
                        }
                    }
                }
            }
        }
    }
}

static void tiled_rowwise_softmax_max_cleanup(void)
{
    volatile double sink = M[DIM_B - 1][DIM_H - 1][DIM_S_Q - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void tiled_rowwise_softmax_max_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t tiled_rowwise_softmax_max_kernel = {
    .name = "tiled_rowwise_softmax_max",
    .description = "Tiled Row-wise Softmax Max (4x16x512x512 -> 4x16x512)",
    .init = tiled_rowwise_softmax_max_init,
    .execute = tiled_rowwise_softmax_max_execute,
    .cleanup = tiled_rowwise_softmax_max_cleanup,
    .verify = tiled_rowwise_softmax_max_verify
};
