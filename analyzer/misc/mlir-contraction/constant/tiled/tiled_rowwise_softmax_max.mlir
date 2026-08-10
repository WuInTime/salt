module attributes {simulation.prologue = "volatile double ARRAY_0[5][17][541][536], ARRAY_1[5][17][536];"} {
  func.func @constant_rowwise_softmax_max(%arg0: memref<4x16x512x512xf32>, %arg1: memref<4x16x512xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 64 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 8 {
              affine.for %arg7 = 0 to 8 {
                affine.for %arg8 = 0 to 8 {
                  %0 = affine.load %arg0[%arg5, %arg6 + %arg2 * 8, %arg7 + %arg3 * 8, %arg8 + %arg4 * 8] : memref<4x16x512x512xf32>
                  %1 = arith.maximumf %0, %cst : f32
                  affine.store %1, %arg1[%arg5, %arg6 + %arg2 * 8, %arg7 + %arg3 * 8] : memref<4x16x512xf32>
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

