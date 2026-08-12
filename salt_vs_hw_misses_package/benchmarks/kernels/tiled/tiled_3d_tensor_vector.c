// Tiled 3D Tensor-Vector Contraction Kernel
// C[i][j] = A[i][j][k] * B[k] with tiling
// Converted from MLIR tiled_3d_tensor_vector.mlir

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: A: 96x64x48, B: 48, C: 96x64
// simulation.prologue: "volatile double ARRAY_0[97][67][56], ARRAY_1[56], ARRAY_2[97][88];"
#define DIM_I 96
#define DIM_J 64
#define DIM_K 48

// Tiling parameters from MLIR
// affine.for %arg3 = 0 to 12, %arg4 = 0 to 8, %arg5 = 0 to 6
// inner: %arg6, %arg7, %arg8 = 0 to 8
// access: A[%arg6 + %arg3 * 8, %arg7 + %arg4 * 8, %arg8 + %arg5 * 8]
#define TILE_SIZE 8

// Padding based on simulation.prologue
#define ARRAY_0_DIM0 97  // 96 + padding
#define ARRAY_0_DIM1 67  // 64 + padding
#define ARRAY_0_DIM2 56  // 48 + padding
#define ARRAY_1_DIM 56   // 48 + padding
#define ARRAY_2_DIM0 97  // 96 + padding
#define ARRAY_2_DIM1 88  // 64 + padding

static volatile double A[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2] __attribute__((aligned(64)));
static volatile double B[ARRAY_1_DIM] __attribute__((aligned(64)));
static volatile double C[ARRAY_2_DIM0][ARRAY_2_DIM1] __attribute__((aligned(64)));

static void tiled_tensor3d_vector_init(void)
{
    for (int i = 0; i < DIM_I; i++) {
        for (int j = 0; j < DIM_J; j++) {
            for (int k = 0; k < DIM_K; k++) {
                A[i][j][k] = (double)(i * DIM_J * DIM_K + j * DIM_K + k);
            }
        }
    }
    
    for (int k = 0; k < DIM_K; k++) {
        B[k] = (double)k;
    }
    
    for (int i = 0; i < DIM_I; i++) {
        for (int j = 0; j < DIM_J; j++) {
            C[i][j] = 0.0;
        }
    }
}

static void tiled_tensor3d_vector_execute(void)
{
    // Tiled loops from MLIR
    // for %arg3 = 0 to 12 (96/8), for %arg4 = 0 to 8 (64/8), for %arg5 = 0 to 6 (48/8)
    // inner: for %arg6, %arg7, %arg8 = 0 to 8
    for (int arg3 = 0; arg3 < 12; arg3++) {
        for (int arg4 = 0; arg4 < 8; arg4++) {
            for (int arg5 = 0; arg5 < 6; arg5++) {
                for (int arg6 = 0; arg6 < 8; arg6++) {
                    for (int arg7 = 0; arg7 < 8; arg7++) {
                        for (int arg8 = 0; arg8 < 8; arg8++) {
                            // int i = arg6 + arg3 * 8;
                            // int j = arg7 + arg4 * 8;
                            // int k = arg8 + arg5 * 8;
                            // double A_ijk = A[i][j][k];
                            // double B_k = B[k];
                            // double prod = A_ijk * B_k;
                            // C[i][j] = prod;
                            C[arg6 + arg3 * 8][arg7 + arg4 * 8] += A[arg6 + arg3 * 8][arg7 + arg4 * 8][arg8 + arg5 * 8] * B[arg8 + arg5 * 8];
                        }
                    }
                }
            }
        }
    }
}

static void tiled_tensor3d_vector_cleanup(void)
{
    volatile double sink = C[DIM_I - 1][DIM_J - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void tiled_tensor3d_vector_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t tiled_3d_tensor_vector_kernel = {
    .name = "tiled_3d_tensor_vector",
    .description = "Tiled 3D Tensor-Vector contraction (96x64x48 * 48 -> 96x64)",
    .init = tiled_tensor3d_vector_init,
    .execute = tiled_tensor3d_vector_execute,
    .cleanup = tiled_tensor3d_vector_cleanup,
    .verify = tiled_tensor3d_vector_verify
};
