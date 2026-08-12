// Tiled Matrix-Matrix Multiplication Kernel
// C[i][k] = A[i][j] * B[j][k] with tiling
// Converted from MLIR tiled_matrix_matrix.mlir

#include "../../include/kernel_interface.h"
#include <stdio.h>

/* stringify helpers for macro values in .description */
#define STR_HELPER(x) #x
#define XSTR(x) STR_HELPER(x)

// Array dimensions from MLIR: A: 256x160, B: 160x192, C: 256x192
// simulation.prologue: "volatile double ARRAY_0[257][184], ARRAY_1[167][248], ARRAY_2[257][248];"
#ifndef DIM_I
#define DIM_I 256
#endif
#ifndef DIM_J
#define DIM_J 160
#endif
#ifndef DIM_K
#define DIM_K 192
#endif

// Tiling parameters from MLIR
// affine.for %arg3 = 0 to 32, %arg4 = 0 to 24, %arg5 = 0 to 20
// inner: %arg6, %arg7, %arg8 = 0 to 8
// access: A[%arg6 + %arg3 * 8, %arg8 + %arg5 * 8]
#ifndef TILE_SIZE
#define TILE_SIZE 8
#endif

// Padding based on simulation.prologue
#ifndef A_PAD_I
#define A_PAD_I 1
#endif
#ifndef A_PAD_J
#define A_PAD_J 24
#endif
#ifndef B_PAD_J
#define B_PAD_J 7
#endif
#ifndef B_PAD_K
#define B_PAD_K 56
#endif
#ifndef C_PAD_I
#define C_PAD_I 1
#endif
#ifndef C_PAD_K
#define C_PAD_K 56
#endif

#define ARRAY_0_DIM0 (DIM_I + A_PAD_I)
#define ARRAY_0_DIM1 (DIM_J + A_PAD_J)
#define ARRAY_1_DIM0 (DIM_J + B_PAD_J)
#define ARRAY_1_DIM1 (DIM_K + B_PAD_K)
#define ARRAY_2_DIM0 (DIM_I + C_PAD_I)
#define ARRAY_2_DIM1 (DIM_K + C_PAD_K)

static volatile double A[ARRAY_0_DIM0][ARRAY_0_DIM1] __attribute__((aligned(64)));
static volatile double B[ARRAY_1_DIM0][ARRAY_1_DIM1] __attribute__((aligned(64)));
static volatile double C[ARRAY_2_DIM0][ARRAY_2_DIM1] __attribute__((aligned(64)));

static void tiled_matrix_matrix_init(void)
{
    for (int i = 0; i < DIM_I; i++) {
        for (int j = 0; j < DIM_J; j++) {
            A[i][j] = (double)(i * DIM_J + j);
        }
    }
    
    for (int j = 0; j < DIM_J; j++) {
        for (int k = 0; k < DIM_K; k++) {
            B[j][k] = (double)(j * DIM_K + k);
        }
    }
    
    for (int i = 0; i < DIM_I; i++) {
        for (int k = 0; k < DIM_K; k++) {
            C[i][k] = 0.0;
        }
    }
}

// static void tiled_matrix_matrix_execute(void)
// {
//     for (int i0 = 0; i0 < DIM_I; i0 += TILE_SIZE) {
//         for (int k0 = 0; k0 < DIM_K; k0 += TILE_SIZE) {
//             for (int j0 = 0; j0 < DIM_J; j0 += TILE_SIZE) {
//                 for (int i = i0; i < i0 + TILE_SIZE && i < DIM_I; i++) {
//                     for (int k = k0; k < k0 + TILE_SIZE && k < DIM_K; k++) {
//                         for (int j = j0; j < j0 + TILE_SIZE && j < DIM_J; j++) {
//                             C[i][k] += A[i][j] * B[j][k];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

static void tiled_matrix_matrix_execute(void)
{
    for (int i0 = 0; i0 < DIM_I; i0 += TILE_SIZE) {
        for (int k0 = 0; k0 < DIM_K; k0 += TILE_SIZE) {
            for (int j0 = 0; j0 < DIM_J; j0 += TILE_SIZE) {
                for (int i = i0; i < i0 + TILE_SIZE; i++) {
                    for (int k = k0; k < k0 + TILE_SIZE; k++) {
                        for (int j = j0; j < j0 + TILE_SIZE; j++) {
                            C[i][k] += A[i][j] * B[j][k];
                        }
                    }
                }
            }
        }
    }
}

static void tiled_matrix_matrix_cleanup(void)
{
    volatile double sink = C[DIM_I - 1][DIM_K - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void tiled_matrix_matrix_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t tiled_matrix_matrix_kernel = {
    .name = "tiled_matrix_matrix",
    .description = "Tiled Matrix-Matrix multiplication ("
                   XSTR(DIM_I) "x" XSTR(DIM_J) " * "
                   XSTR(DIM_J) "x" XSTR(DIM_K) " -> "
                   XSTR(DIM_I) "x" XSTR(DIM_K) ", tile "
                   XSTR(TILE_SIZE) ")",
    .init = tiled_matrix_matrix_init,
    .execute = tiled_matrix_matrix_execute,
    .cleanup = tiled_matrix_matrix_cleanup,
    .verify = tiled_matrix_matrix_verify
};
