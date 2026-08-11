module attributes {simulation.prologue = "volatile double ARRAY_0[257][184], ARRAY_1[167][248], ARRAY_2[257][248];"} {
  func.func @matrix_matrix_contraction(%arg0: memref<256x160xf64>, %arg1: memref<160x192xf64>, %arg2: memref<256x192xf64>) {
    affine.for %arg3 = 0 to 32 {
      affine.for %arg4 = 0 to 24 {
        affine.for %arg5 = 0 to 20 {
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              affine.for %arg8 = 0 to 8 {
                %0 = affine.load %arg0[%arg6 + %arg3 * 8, %arg8 + %arg5 * 8] : memref<256x160xf64>
                %1 = affine.load %arg1[%arg8 + %arg5 * 8, %arg7 + %arg4 * 8] : memref<160x192xf64>
                %2 = arith.mulf %0, %1 : f64
                affine.store %2, %arg2[%arg6 + %arg3 * 8, %arg7 + %arg4 * 8] : memref<256x192xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}

