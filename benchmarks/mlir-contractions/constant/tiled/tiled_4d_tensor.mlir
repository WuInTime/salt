module attributes {simulation.prologue = "volatile double ARRAY_0[83][53][37][88], ARRAY_1[53][40], ARRAY_2[83][88];"} {
  func.func @tensor4d_contraction(%arg0: memref<80x48x32x64xf64>, %arg1: memref<48x32xf64>, %arg2: memref<80x64xf64>) {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 8 {
        affine.for %arg5 = 0 to 6 {
          affine.for %arg6 = 0 to 4 {
            affine.for %arg7 = 0 to 8 {
              affine.for %arg8 = 0 to 8 {
                affine.for %arg9 = 0 to 8 {
                  affine.for %arg10 = 0 to 8 {
                    %0 = affine.load %arg0[%arg7 + %arg3 * 8, %arg9 + %arg5 * 8, %arg10 + %arg6 * 8, %arg8 + %arg4 * 8] : memref<80x48x32x64xf64>
                    %1 = affine.load %arg1[%arg9 + %arg5 * 8, %arg10 + %arg6 * 8] : memref<48x32xf64>
                    %2 = arith.mulf %0, %1 : f64
                    affine.store %2, %arg2[%arg7 + %arg3 * 8, %arg8 + %arg4 * 8] : memref<80x64xf64>
                  }
                }
              }
            }
          }
        }
      }
    }
    return
  }
}

