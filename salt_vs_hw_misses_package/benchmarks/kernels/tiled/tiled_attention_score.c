// Tiled Attention Score Computation Kernel
// S[b][h][q][k] += Q[b][h][q][d] * K[b][h][k][d] with tiling
// Converted from MLIR tiled_attention_score.mlir
// Transformer attention mechanism: Query-Key dot product

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: Q: 4x16x256x64, K: 4x16x256x64, S: 4x16x256x256
// simulation.prologue: "volatile double ARRAY_0[5][17][257][88], ARRAY_1[5][17][257][88], ARRAY_2[5][17][257][296];"
#define DIM_B 4    // Batch size
#define DIM_H 16   // Number of attention heads
#define DIM_S_Q 256 // Query sequence length
#define DIM_S_K 256 // Key sequence length
#define DIM_D 64   // Head dimension

// Tiling parameters from MLIR
// affine.for %arg3 = 0 to 2, %arg4 = 0 to 32, %arg5 = 0 to 32, %arg6 = 0 to 8
// inner: %arg7, %arg8, %arg9, %arg10, %arg11 = 0 to 8, 8, 8, 8, 8
// access: Q[%arg7, %arg8 + %arg3 * 8, %arg9 + %arg4 * 8, %arg11 + %arg6 * 8]
#define TILE_SIZE 8

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 5    // 4 + padding (Q)
#define ARRAY_0_DIM1 17   // 16 + padding
#define ARRAY_0_DIM2 257  // 256 + padding
#define ARRAY_0_DIM3 88   // 64 + padding
#define ARRAY_1_DIM0 5    // 4 + padding (K)
#define ARRAY_1_DIM1 17   // 16 + padding
#define ARRAY_1_DIM2 257  // 256 + padding
#define ARRAY_1_DIM3 88   // 64 + padding
#define ARRAY_2_DIM0 5    // 4 + padding (S)
#define ARRAY_2_DIM1 17   // 16 + padding
#define ARRAY_2_DIM2 257  // 256 + padding
#define ARRAY_2_DIM3 296  // 256 + padding
#define padding 8 // Minimal padding to prevent mapping to the same L1d sets

// static volatile double Q[DIM_B][DIM_H][DIM_S_Q+padding][DIM_D+padding] __attribute__((aligned(64)));
// static volatile double K[DIM_B][DIM_H][DIM_S_K+padding][DIM_D+padding] __attribute__((aligned(64)));
// static volatile double S[DIM_B][DIM_H][DIM_S_Q+padding][DIM_S_K+padding] __attribute__((aligned(64)));

static volatile double Q[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2][ARRAY_0_DIM3] __attribute__((aligned(64)));
static volatile double K[ARRAY_1_DIM0][ARRAY_1_DIM1][ARRAY_1_DIM2][ARRAY_1_DIM3] __attribute__((aligned(64)));
static volatile double S[ARRAY_2_DIM0][ARRAY_2_DIM1][ARRAY_2_DIM2][ARRAY_2_DIM3] __attribute__((aligned(64)));

static void tiled_attention_score_init(void)
{
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int q = 0; q < DIM_S_Q; q++) {
                for (int d = 0; d < DIM_D; d++) {
                    Q[b][h][q][d] = (double)(b * DIM_H * DIM_S_Q * DIM_D + h * DIM_S_Q * DIM_D + q * DIM_D + d);
                }
            }
        }
    }
    
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int k = 0; k < DIM_S_K; k++) {
                for (int d = 0; d < DIM_D; d++) {
                    K[b][h][k][d] = (double)(b * DIM_H * DIM_S_K * DIM_D + h * DIM_S_K * DIM_D + k * DIM_D + d);
                }
            }
        }
    }
    
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int q = 0; q < DIM_S_Q; q++) {
                for (int k = 0; k < DIM_S_K; k++) {
                    S[b][h][q][k] = 0.0f;
                }
            }
        }
    }
}

static void tiled_attention_score_execute(void)
{
    // Tiled loops from MLIR
    // for %arg3 = 0 to 2 (16/8), %arg4 = 0 to 32 (256/8), %arg5 = 0 to 32 (256/8), %arg6 = 0 to 8 (64/8)
    // inner: for %arg7, %arg8, %arg9, %arg10, %arg11 = 0 to 4, 8, 8, 8, 8
    // double cst = 0.0f;
    for (int arg3 = 0; arg3 < 2; arg3++) {
        for (int arg4 = 0; arg4 < 32; arg4++) {
            for (int arg5 = 0; arg5 < 32; arg5++) {
                for (int arg6 = 0; arg6 < 8; arg6++) {
                    for (int arg7 = 0; arg7 < 4; arg7++) {
                        for (int arg8 = 0; arg8 < 8; arg8++) {
                            for (int arg9 = 0; arg9 < 8; arg9++) {
                                for (int arg10 = 0; arg10 < 8; arg10++) {
                                    for (int arg11 = 0; arg11 < 8; arg11++) {
                                        // int b = arg7;
                                        // int h = arg8 + arg3 * 8;
                                        // int q = arg9 + arg4 * 8;
                                        // int k = arg10 + arg5 * 8;
                                        // int d = arg11 + arg6 * 8;
                                        // double query_val = Q[b][h][q][d];
                                        // double key_val = K[b][h][k][d];
                                        // double mul = query_val * key_val;
                                        // double add = mul + cst;
                                        // S[b][h][q][k] = add;
                                        S[arg7][arg8 + arg3 * 8][arg9 + arg4 * 8][arg10 + arg5 * 8] += Q[arg7][arg8 + arg3 * 8][arg9 + arg4 * 8][arg11 + arg6 * 8] * K[arg7][arg8 + arg3 * 8][arg10 + arg5 * 8][arg11 + arg6 * 8];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

static void tiled_attention_score_cleanup(void)
{
    volatile double sink = S[DIM_B - 1][DIM_H - 1][DIM_S_Q - 1][DIM_S_K - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void tiled_attention_score_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t tiled_attention_score_kernel = {
    .name = "tiled_attention_score",
    .description = "Tiled Attention Score computation (4x16x256x64 Q*K -> 4x16x256x256)",
    .init = tiled_attention_score_init,
    .execute = tiled_attention_score_execute,
    .cleanup = tiled_attention_score_cleanup,
    .verify = tiled_attention_score_verify
};
