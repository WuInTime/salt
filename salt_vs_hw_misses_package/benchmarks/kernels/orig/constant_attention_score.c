// Attention Score Computation Kernel
// S[b][h][q][k] += Q[b][h][q][d] * K[b][h][k][d]
// Converted from MLIR constant_attention_score.mlir
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

static volatile double Q[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2][ARRAY_0_DIM3] __attribute__((aligned(64)));
static volatile double K[ARRAY_1_DIM0][ARRAY_1_DIM1][ARRAY_1_DIM2][ARRAY_1_DIM3] __attribute__((aligned(64)));
static volatile double S[ARRAY_2_DIM0][ARRAY_2_DIM1][ARRAY_2_DIM2][ARRAY_2_DIM3] __attribute__((aligned(64)));

static void attention_score_init(void)
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

static void attention_score_execute(void)
{
    // S[b][h][q][k] += Q[b][h][q][d] * K[b][h][k][d]
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int q = 0; q < DIM_S_Q; q++) {
                for (int k = 0; k < DIM_S_K; k++) {
                    for (int d = 0; d < DIM_D; d++) {
                        // double query_val = Q[b][h][q][d];
                        // double key_val = K[b][h][k][d];
                        // double score_val = 0.0;
                        // double mul = query_val * key_val;
                        // double add = score_val + mul;
                        // S[b][h][q][k] = add;
                        S[b][h][q][k] += Q[b][h][q][d] * K[b][h][k][d];
                    }
                }
            }
        }
    }
}

static void attention_score_cleanup(void)
{
    volatile double sink = S[DIM_B - 1][DIM_H - 1][DIM_S_Q - 1][DIM_S_K - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void attention_score_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t constant_attention_score_kernel = {
    .name = "constant_attention_score",
    .description = "Attention Score computation (4x16x256x64 Q*K -> 4x16x256x256)",
    .init = attention_score_init,
    .execute = attention_score_execute,
    .cleanup = attention_score_cleanup,
    .verify = attention_score_verify
};
