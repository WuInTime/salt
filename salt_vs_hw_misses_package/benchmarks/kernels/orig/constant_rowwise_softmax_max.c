// Row-wise Softmax Max Computation Kernel
// M[b][h][q] = fmaxf(M[b][h][q], S[b][h][q][k])
// Converted from MLIR constant_rowwise_softmax_max.mlir
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

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 5    // 4 + padding (S)
#define ARRAY_0_DIM1 17   // 16 + padding
#define ARRAY_0_DIM2 541  // 512 + padding
#define ARRAY_0_DIM3 536  // 512 + padding
#define ARRAY_1_DIM0 5    // 4 + padding (M)
#define ARRAY_1_DIM1 17   // 16 + padding
#define ARRAY_1_DIM2 536  // 512 + padding

static volatile double S[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2][ARRAY_0_DIM3] __attribute__((aligned(64)));
static volatile double M[ARRAY_1_DIM0][ARRAY_1_DIM1][ARRAY_1_DIM2] __attribute__((aligned(64)));

static void rowwise_softmax_max_init(void)
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
                M[b][h][q] = 0.0;
            }
        }
    }
}

static void rowwise_softmax_max_execute(void)
{
    // M[b][h][q] = fmaxf(M[b][h][q], S[b][h][q][k])
    for (int b = 0; b < DIM_B; b++) {
        for (int h = 0; h < DIM_H; h++) {
            for (int q = 0; q < DIM_S_Q; q++) {
                for (int k = 0; k < DIM_S_K; k++) {
                    // double score_val = S[b][h][q][k];
                    // double max_val = 0.0;
                    // double new_max = fmaxf(max_val, score_val);
                    // M[b][h][q] = new_max;
                    M[b][h][q] = fmaxf(M[b][h][q], S[b][h][q][k]);
                }
            }
        }
    }
}

static void rowwise_softmax_max_cleanup(void)
{
    volatile double sink = M[DIM_B - 1][DIM_H - 1][DIM_S_Q - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void rowwise_softmax_max_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t constant_rowwise_softmax_max_kernel = {
    .name = "constant_rowwise_softmax_max",
    .description = "Row-wise Softmax Max (4x16x512x512 -> 4x16x512)",
    .init = rowwise_softmax_max_init,
    .execute = rowwise_softmax_max_execute,
    .cleanup = rowwise_softmax_max_cleanup,
    .verify = rowwise_softmax_max_verify
};
