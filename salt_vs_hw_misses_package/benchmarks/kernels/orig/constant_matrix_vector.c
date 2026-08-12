// Matrix-Vector Multiplication Kernel
// C[i] = A[i][j] * B[j]
// Converted from MLIR constant_matrix_vector.mlir

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: A: 512x1024, B: 1024, C: 512
// simulation.prologue: "volatile double ARRAY_0[541][1048], ARRAY_1[1048], ARRAY_2[536];"
#define DIM_I 512
#define DIM_J 1024

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 541  // 512 + padding
#define ARRAY_0_DIM1 1048 // 1024 + padding
#define ARRAY_1_DIM 1048  // 1024 + padding
#define ARRAY_2_DIM 536   // 512 + padding

static volatile double A[ARRAY_0_DIM0][ARRAY_0_DIM1] __attribute__((aligned(64)));
static volatile double B[ARRAY_1_DIM] __attribute__((aligned(64)));
static volatile double C[ARRAY_2_DIM] __attribute__((aligned(64)));

static void matrix_vector_init(void)
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

static void matrix_vector_execute(void)
{
    // C[i] = A[i][j] * B[j]
    for (int i = 0; i < DIM_I; i++) {
        for (int j = 0; j < DIM_J; j++) {
            // double A_ij = A[i][j];
            // double B_j = B[j];
            // double prod = A_ij * B_j;
            // C[i] = prod;
            C[i] += A[i][j] * B[j];
        }
    }
}

static void matrix_vector_cleanup(void)
{
    volatile double sink = C[DIM_I - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void matrix_vector_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t constant_matrix_vector_kernel = {
    .name = "constant_matrix_vector",
    .description = "Matrix-Vector multiplication (512x1024 * 1024 -> 512)",
    .init = matrix_vector_init,
    .execute = matrix_vector_execute,
    .cleanup = matrix_vector_cleanup,
    .verify = matrix_vector_verify
};
