module attributes {simulation.prologue = "volatile double ARRAY_0[5][17][257][88], ARRAY_1[5][17][257][88], ARRAY_2[5][17][257][296];"} {
  func.func @constant_attention_score(%arg0: memref<4x16x256x64xf32>, %arg1: memref<4x16x256x64xf32>, %arg2: memref<4x16x256x256xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 32 {
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 4 {
              affine.for %arg8 = 0 to 8 {
                affine.for %arg9 = 0 to 8 {
                  affine.for %arg10 = 0 to 8 {
                    affine.for %arg11 = 0 to 8 {
                      %0 = affine.load %arg0[%arg7, %arg8 + %arg3 * 8, %arg9 + %arg4 * 8, %arg11 + %arg6 * 8] : memref<4x16x256x64xf32>
                      %1 = affine.load %arg1[%arg7, %arg8 + %arg3 * 8, %arg10 + %arg5 * 8, %arg11 + %arg6 * 8] : memref<4x16x256x64xf32>
                      %2 = arith.mulf %0, %1 : f32
                      %3 = arith.addf %2, %cst : f32
                      affine.store %3, %arg2[%arg7, %arg8 + %arg3 * 8, %arg9 + %arg4 * 8, %arg10 + %arg5 * 8] : memref<4x16x256x256xf32>
                    }
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

