// Tiled Matrix-Vector Multiplication Kernel
// C[i] = A[i][j] * B[j] with tiling
// Converted from MLIR tiled_matrix_vector.mlir

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: A: 512x1024, B: 1024, C: 512
// simulation.prologue: "volatile double ARRAY_0[541][1048], ARRAY_1[1048], ARRAY_2[536];"
#define DIM_I 512
#define DIM_J 1024

// Tiling parameters from MLIR
// affine.for %arg3 = 0 to 64 { affine.for %arg4 = 0 to 128 { ... %arg5 + %arg3 * 8, %arg6 + %arg4 * 8
#define TILE_I 8
#define TILE_J 8

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 541  // 512 + padding
#define ARRAY_0_DIM1 1048 // 1024 + padding
#define ARRAY_1_DIM 1048  // 1024 + padding
#define ARRAY_2_DIM 536   // 512 + padding

static volatile double A[ARRAY_0_DIM0][ARRAY_0_DIM1] __attribute__((aligned(64)));
static volatile double B[ARRAY_1_DIM] __attribute__((aligned(64)));
static volatile double C[ARRAY_2_DIM] __attribute__((aligned(64)));

static void tiled_matrix_vector_init(void)
{
    for (int i = 0; i < DIM_I; i++) {
        for (int j = 0; j < DIM_J; j++) {
            A[i][j] = (double)(i * DIM_J + j);
        }
    }
    
    for (int j = 0; j < DIM_J; j++) {
        B[j] = (double)j;
    }
    
    for (int i = 0; i < DIM_I; i++) {
        C[i] = 0.0;
    }
}

static void tiled_matrix_vector_execute(void)
{
    // Tiled loops: outer tiles, inner tile elements
    // for %arg3 = 0 to 64 (512/8), for %arg4 = 0 to 128 (1024/8)
    // inner: for %arg5 = 0 to 8, for %arg6 = 0 to 8
    // access: A[%arg5 + %arg3 * 8, %arg6 + %arg4 * 8]
    for (int arg3 = 0; arg3 < 64; arg3++) {
        for (int arg4 = 0; arg4 < 128; arg4++) {
            for (int arg5 = 0; arg5 < 8; arg5++) {
                for (int arg6 = 0; arg6 < 8; arg6++) {
                    // int i = arg5 + arg3 * 8;
                    // int j = arg6 + arg4 * 8;
                    // double A_ij = A[i][j];
                    // double B_j = B[j];
                    // double prod = A_ij * B_j;
                    // C[i] = prod;
                    C[arg5 + arg3 * 8] += A[arg5 + arg3 * 8][arg6 + arg4 * 8] * B[arg6 + arg4 * 8];
                }
            }
        }
    }
}

static void tiled_matrix_vector_cleanup(void)
{
    volatile double sink = C[DIM_I - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void tiled_matrix_vector_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t tiled_matrix_vector_kernel = {
    .name = "tiled_matrix_vector",
    .description = "Tiled Matrix-Vector multiplication (512x1024 * 1024 -> 512)",
    .init = tiled_matrix_vector_init,
    .execute = tiled_matrix_vector_execute,
    .cleanup = tiled_matrix_vector_cleanup,
    .verify = tiled_matrix_vector_verify
};
