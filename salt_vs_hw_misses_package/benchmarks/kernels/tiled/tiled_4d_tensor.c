// Tiled 4D Tensor Contraction Kernel
// C[i][j] = A[i][k][l][j] * B[k][l] with tiling
// Converted from MLIR tiled_4d_tensor.mlir

#include "../../include/kernel_interface.h"
#include <stdio.h>

// Array dimensions from MLIR: A: 80x48x32x64, B: 48x32, C: 80x64
// simulation.prologue: "volatile double ARRAY_0[83][53][37][88], ARRAY_1[53][40], ARRAY_2[83][88];"
#define DIM_I 80
#define DIM_J 64
#define DIM_K 48
#define DIM_L 32

// Tiling parameters from MLIR
// affine.for %arg3 = 0 to 10, %arg4 = 0 to 8, %arg5 = 0 to 6, %arg6 = 0 to 4
// inner: %arg7, %arg8, %arg9, %arg10 = 0 to 8
// access: A[%arg7 + %arg3 * 8, %arg9 + %arg5 * 8, %arg10 + %arg6 * 8, %arg8 + %arg4 * 8]
#define TILE_SIZE 8

// Padding based on simulation.prologue
// #define ARRAY_0_DIM0 83  // 80 + padding
// #define ARRAY_0_DIM1 53  // 48 + padding
// #define ARRAY_0_DIM2 37  // 32 + padding
// #define ARRAY_0_DIM3 88  // 64 + padding
// #define ARRAY_1_DIM0 53  // 48 + padding
// #define ARRAY_1_DIM1 40  // 32 + padding
// #define ARRAY_2_DIM0 83  // 80 + padding
// #define ARRAY_2_DIM1 88  // 64 + padding

#define ARRAY_0_DIM0 88  // 80 + padding
#define ARRAY_0_DIM1 56  // 48 + padding
#define ARRAY_0_DIM2 40  // 32 + padding
#define ARRAY_0_DIM3 72  // 64 + padding
#define ARRAY_1_DIM0 48  // 48 + padding
#define ARRAY_1_DIM1 32  // 32 + padding
#define ARRAY_2_DIM0 80  // 80 + padding
#define ARRAY_2_DIM1 64  // 64 + padding

// static volatile double A[ARRAY_0_DIM0][ARRAY_0_DIM1][ARRAY_0_DIM2][ARRAY_0_DIM3] __attribute__((aligned(64)));
// static char __pad_A_B[64]; // 4KB padding
// static volatile double B[ARRAY_1_DIM0][ARRAY_1_DIM1] __attribute__((aligned(64)));
// static char __pad_B_C[64]; // 4KB padding
// static volatile double C[ARRAY_2_DIM0][ARRAY_2_DIM1] __attribute__((aligned(64)));

static volatile double A[DIM_I][DIM_K][DIM_L][DIM_J+8] __attribute__((aligned(64)));
static volatile double B[DIM_K][DIM_L+8] __attribute__((aligned(64)));
static volatile double C[DIM_I][DIM_J+8] __attribute__((aligned(64)));

static void tiled_tensor4d_init(void)
{
    for (int i = 0; i < DIM_I; i++) {
        for (int k = 0; k < DIM_K; k++) {
            for (int l = 0; l < DIM_L; l++) {
                for (int j = 0; j < DIM_J; j++) {
                    A[i][k][l][j] = (double)(i * DIM_K * DIM_L * DIM_J + k * DIM_L * DIM_J + l * DIM_J + j);
                }
            }
        }
    }
    
    for (int k = 0; k < DIM_K; k++) {
        for (int l = 0; l < DIM_L; l++) {
            B[k][l] = (double)(k * DIM_L + l);
        }
    }
    
    for (int i = 0; i < DIM_I; i++) {
        for (int j = 0; j < DIM_J; j++) {
            C[i][j] = 0.0;
        }
    }
}

#define TI 8
#define TJ 8
#define TK 8
#define TL 8

static void tiled_tensor4d_execute(void)
{
    // // Tiled loops from MLIR
    // // for %arg3 = 0 to 10 (80/8), %arg4 = 0 to 8 (64/8), %arg5 = 0 to 6 (48/8), %arg6 = 0 to 4 (32/8)
    // // inner: for %arg7, %arg8, %arg9, %arg10 = 0 to 8
    // for (int arg3 = 0; arg3 < 10; arg3++) {
    //     for (int arg4 = 0; arg4 < 8; arg4++) {
    //         for (int arg5 = 0; arg5 < 6; arg5++) {
    //             for (int arg6 = 0; arg6 < 4; arg6++) {
    //                 for (int arg7 = 0; arg7 < 8; arg7++) {
    //                     for (int arg8 = 0; arg8 < 8; arg8++) {
    //                         for (int arg9 = 0; arg9 < 8; arg9++) {
    //                             for (int arg10 = 0; arg10 < 8; arg10++) {
    //                                 // int i = arg7 + arg3 * 8;
    //                                 // int k = arg9 + arg5 * 8;
    //                                 // int l = arg10 + arg6 * 8;
    //                                 // int j = arg8 + arg4 * 8;
    //                                 // double A_iklj = A[i][k][l][j];
    //                                 // double B_kl = B[k][l];
    //                                 // double prod = A_iklj * B_kl;
    //                                 // C[i][j] = prod;
    //                                 C[arg7 + arg3 * 8][arg8 + arg4 * 8] += A[arg7 + arg3 * 8][arg9 + arg5 * 8][arg10 + arg6 * 8][arg8 + arg4 * 8] * B[arg9 + arg5 * 8][arg10 + arg6 * 8];
    //                             }
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    
    for (int i0 = 0; i0 < DIM_I; i0 += TI)
        for (int j0 = 0; j0 < DIM_J; j0 += TJ)
            for (int k0 = 0; k0 < DIM_K; k0 += TK)
                for (int l0 = 0; l0 < DIM_L; l0 += TL)
                    
                    for (int i = i0; i < i0 + TI; ++i)
                        for (int j = j0; j < j0 + TJ; ++j)
                            for (int k = k0; k < k0 + TK; ++k)
                                for (int l = l0; l < l0 + TL; ++l)
                                    C[i][j] += A[i][k][l][j] * B[k][l];
}


// static void tensor4d_execute_tiled(void)
// {
// for (int i0 = 0; i0 < DIM_I; i0 += TI) {
//         // int i_max = (i0 + TI < DIM_I) ? (i0 + TI) : DIM_I;
//         for (int j0 = 0; j0 < DIM_J; j0 += TJ) {
//             int j_max = (j0 + TJ < DIM_J) ? (j0 + TJ) : DIM_J;
//             // // Initialize the output tile (optional if C was zeroed before)
//             // for (int i = i0; i < i_max; ++i)
//             //     for (int j = j0; j < j_max; ++j)
//             //         ; // assume C already zeroed in init
    
//             for (int k0 = 0; k0 < DIM_K; k0 += TK) {
//                 // int k_max = (k0 + TK < DIM_K) ? (k0 + TK) : DIM_K;
//                 for (int l0 = 0; l0 < DIM_L; l0 += TL) {
//                     // int l_max = (l0 + TL < DIM_L) ? (l0 + TL) : DIM_L;
    
//                     // compute small tile: accumulate into C[i][j]
//                     for (int i = i0; i < i0 + TI; ++i) {
//                         for (int k = k0; k < k0 + TK; ++k) {
//                             for (int l = l0; l < l0 + TL; ++l) {
//                                 // iterate j inner to get contiguous access to A[i][k][l][j]
//                                 for (int j = j0; j < j_max; ++j) {
//                                     C[i][j] += A[i][k][l][j] * B[k][l];
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

static void tiled_tensor4d_cleanup(void)
{
    volatile double sink = C[DIM_I - 1][DIM_J - 1];
    fprintf(stderr, "Result: %f (ignore)\n", sink);
}

static void tiled_tensor4d_verify(void)
{
    // Optional: Add verification logic here
}

kernel_t tiled_4d_tensor_kernel = {
    .name = "tiled_4d_tensor",
    .description = "Tiled 4D Tensor contraction (80x48x32x64 * 48x32 -> 80x64)",
    .init = tiled_tensor4d_init,
    .execute = tiled_tensor4d_execute,
    .cleanup = tiled_tensor4d_cleanup,
    .verify = tiled_tensor4d_verify
};
