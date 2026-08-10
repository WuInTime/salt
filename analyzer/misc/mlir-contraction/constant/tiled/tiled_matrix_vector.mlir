module attributes {simulation.prologue = "volatile double ARRAY_0[541][1048], ARRAY_1[1048], ARRAY_2[536];"} {
  func.func @matrix_vector_contraction(%arg0: memref<512x1024xf64>, %arg1: memref<1024xf64>, %arg2: memref<512xf64>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 128 {
        affine.for %arg5 = 0 to 8 {
          affine.for %arg6 = 0 to 8 {
            %0 = affine.load %arg0[%arg5 + %arg3 * 8, %arg6 + %arg4 * 8] : memref<512x1024xf64>
            %1 = affine.load %arg1[%arg6 + %arg4 * 8] : memref<1024xf64>
            %2 = arith.mulf %0, %1 : f64
            affine.store %2, %arg2[%arg5 + %arg3 * 8] : memref<512xf64>
          }
        }
      }
    }
    return
  }
}

