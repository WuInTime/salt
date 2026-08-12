// Context Lookup Computation Kernel
// O[b][h][q][d] += P[b][h][q][k] * V[b][h][k][d]
// Converted from MLIR constant_context_lookup.mlir
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

static volatile double P[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2][ARRAY_0_DIM3] __attribute__((aligned(64)));
static volatile double V[ARRAY_1_DIM0][ARRAY_1_DIM1][ARRAY_1_DIM2][ARRAY_1_DIM3] __attribute__((aligned(64)));
static volatile double O[ARRAY_2_DIM0][ARRAY_2_DIM1][ARRAY_2_DIM2][ARRAY_2_DIM3] __attribute__((aligned(64)));

static void context_lookup_init(void)
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

static void context_lookup_execute(void)
{
    // O[b][h][q][d] += P[b][h][q][k] * V[b][h][k][d]
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int q = 0; q < DIM_S_Q; q++) {
                for (int d = 0; d < DIM_D; d++) {
                    for (int k = 0; k < DIM_S_K; k++) {
                        // double prob_val = P[b][h][q][k];
                        // double value_val = V[b][h][k][d];
                        // double output_val = 0.0;
                        // double mul = prob_val * value_val;
                        // double add = output_val + mul;
                        // O[b][h][q][d] = add;
                        O[b][h][q][d] += P[b][h][q][k] * V[b][h][k][d];
                    }
                }
            }
        }
    }
}

static void context_lookup_cleanup(void)
{
    volatile double sink = O[DIM_B - 1][DIM_H - 1][DIM_S_Q - 1][DIM_D - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void context_lookup_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t constant_context_lookup_kernel = {
    .name = "constant_context_lookup",
    .description = "Context Lookup (4x16x256x256 P*V -> 4x16x256x64)",
    .init = context_lookup_init,
    .execute = context_lookup_execute,
    .cleanup = context_lookup_cleanup,
    .verify = context_lookup_verify
};
