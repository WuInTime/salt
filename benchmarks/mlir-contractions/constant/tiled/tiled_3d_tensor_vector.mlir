module attributes {simulation.prologue = "volatile double ARRAY_0[97][67][56], ARRAY_1[56], ARRAY_2[97][88];"} {
  func.func @tensor3d_vector_contraction(%arg0: memref<96x64x48xf64>, %arg1: memref<48xf64>, %arg2: memref<96x64xf64>) {
    affine.for %arg3 = 0 to 12 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 6 {
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              affine.for %arg8 = 0 to 8 {
                %0 = affine.load %arg0[%arg6 + %arg3 * 8, %arg7 + %arg4 * 8, %arg8 + %arg5 * 8] : memref<96x64x48xf64>
                %1 = affine.load %arg1[%arg8 + %arg5 * 8] : memref<48xf64>
                %2 = arith.mulf %0, %1 : f64
                affine.store %2, %arg2[%arg6 + %arg3 * 8, %arg7 + %arg4 * 8] : memref<96x64xf64>
              }
            }
          }
        }
      }
    }
    return
  }
}

