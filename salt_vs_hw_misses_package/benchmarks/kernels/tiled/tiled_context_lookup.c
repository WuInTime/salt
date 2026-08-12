// Tiled Context Lookup Computation Kernel
// O[b][h][q][d] += P[b][h][q][k] * V[b][h][k][d] with tiling
// Converted from MLIR tiled_context_lookup.mlir
// Transformer attention mechanism: weighted sum of value vectors

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: P: 4x16x256x256, V: 4x16x256x64, O: 4x16x256x64
// simulation.prologue: "volatile double ARRAY_0[5][17][257][296], ARRAY_1[5][17][257][88], ARRAY_2[5][17][257][88];"
#define DIM_B 4    // Batch size
#define DIM_H 16   // Number of attention heads
#define DIM_S_Q 256 // Query sequence length
#define DIM_S_K 256 // Key sequence length
#define DIM_D 64   // Head dimension

// Tiling parameters from MLIR
// affine.for %arg3 = 0 to 2, %arg4 = 0 to 32, %arg5 = 0 to 8, %arg6 = 0 to 32
// inner: %arg7, %arg8, %arg9, %arg10, %arg11 = 0 to 4, 8, 8, 8, 8
// access: P[%arg7, %arg8 + %arg3 * 8, %arg9 + %arg4 * 8, %arg11 + %arg6 * 8]
#define TILE_SIZE 8

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 5    // 4 + padding (P)
#define ARRAY_0_DIM1 17   // 16 + padding
#define ARRAY_0_DIM2 257  // 256 + padding
#define ARRAY_0_DIM3 296  // 256 + padding
#define ARRAY_1_DIM0 5    // 4 + padding (V)
#define ARRAY_1_DIM1 17   // 16 + padding
#define ARRAY_1_DIM2 257  // 256 + padding
#define ARRAY_1_DIM3 88   // 64 + padding
#define ARRAY_2_DIM0 5    // 4 + padding (O)
#define ARRAY_2_DIM1 17   // 16 + padding
#define ARRAY_2_DIM2 257  // 256 + padding
#define ARRAY_2_DIM3 88   // 64 + padding

// static volatile double P[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2][ARRAY_0_DIM3] __attribute__((aligned(64)));
// static volatile double V[ARRAY_1_DIM0][ARRAY_1_DIM1][ARRAY_1_DIM2][ARRAY_1_DIM3] __attribute__((aligned(64)));
// static volatile double O[ARRAY_2_DIM0][ARRAY_2_DIM1][ARRAY_2_DIM2][ARRAY_2_DIM3] __attribute__((aligned(64)));

static volatile double P[DIM_B][DIM_H][DIM_S_Q][DIM_S_K+8] __attribute__((aligned(64)));
static volatile double V[DIM_B][DIM_H][DIM_S_K][DIM_D+8] __attribute__((aligned(64)));
static volatile double O[DIM_B][DIM_H][DIM_S_Q][DIM_D+8] __attribute__((aligned(64)));

static void tiled_context_lookup_init(void)
{
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int q = 0; q < DIM_S_Q; q++) {
                for (int k = 0; k < DIM_S_K; k++) {
                    P[b][h][q][k] = (double)(b * DIM_H * DIM_S_Q * DIM_S_K + h * DIM_S_Q * DIM_S_K + q * DIM_S_K + k);
                }
            }
        }
    }
    
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int k = 0; k < DIM_S_K; k++) {
                for (int d = 0; d < DIM_D; d++) {
                    V[b][h][k][d] = (double)(b * DIM_H * DIM_S_K * DIM_D + h * DIM_S_K * DIM_D + k * DIM_D + d);
                }
            }
        }
    }
    
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int q = 0; q < DIM_S_Q; q++) {
                for (int d = 0; d < DIM_D; d++) {
                    O[b][h][q][d] = 0.0f;
                }
            }
        }
    }
}

static void tiled_context_lookup_execute(void)
{
    // Tiled loops from MLIR
    // for %arg3 = 0 to 2 (16/8), %arg4 = 0 to 32 (256/8), %arg5 = 0 to 8 (64/8), %arg6 = 0 to 32 (256/8)
    // inner: for %arg7, %arg8, %arg9, %arg10, %arg11 = 0 to 4, 8, 8, 8, 8
    // double cst = 0.0f;
    for (int arg3 = 0; arg3 < 2; arg3++) {
        for (int arg4 = 0; arg4 < 32; arg4++) {
            for (int arg5 = 0; arg5 < 8; arg5++) {
                for (int arg6 = 0; arg6 < 32; arg6++) {
                    for (int arg7 = 0; arg7 < 4; arg7++) {
                        for (int arg8 = 0; arg8 < 8; arg8++) {
                            for (int arg9 = 0; arg9 < 8; arg9++) {
                                for (int arg10 = 0; arg10 < 8; arg10++) {
                                    for (int arg11 = 0; arg11 < 8; arg11++) {
                                        // int b = arg7;
                                        // int h = arg8 + arg3 * 8;
                                        // int q = arg9 + arg4 * 8;
                                        // int d = arg10 + arg5 * 8;
                                        // int k = arg11 + arg6 * 8;
                                        // double prob_val = P[b][h][q][k];
                                        // double value_val = V[b][h][k][d];
                                        // double mul = prob_val * value_val;
                                        // double add = mul + cst;
                                        // O[b][h][q][d] = add;
                                        O[arg7][arg8 + arg3 * 8][arg9 + arg4 * 8][arg10 + arg5 * 8] += P[arg7][arg8 + arg3 * 8][arg9 + arg4 * 8][arg11 + arg6 * 8] * V[arg7][arg8 + arg3 * 8][arg11 + arg6 * 8][arg10 + arg5 * 8];
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

static void tiled_context_lookup_cleanup(void)
{
    volatile double sink = O[DIM_B - 1][DIM_H - 1][DIM_S_Q - 1][DIM_D - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void tiled_context_lookup_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t tiled_context_lookup_kernel = {
    .name = "tiled_context_lookup",
    .description = "Tiled Context Lookup (4x16x256x256 P*V -> 4x16x256x64)",
    .init = tiled_context_lookup_init,
    .execute = tiled_context_lookup_execute,
    .cleanup = tiled_context_lookup_cleanup,
    .verify = tiled_context_lookup_verify
};
