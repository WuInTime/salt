module {
  func.func @stencil5pt(%A: memref<?x?xf64>, %B: memref<?x?xf64>,
                        %N: index, %M: index) {
    affine.for %i = 1 to affine_map<()[s0] -> (s0 - 1)>()[%N] {
      affine.for %j = 1 to affine_map<()[s0] -> (s0 - 1)>()[%M] {
        %bottom = affine.load %A[%i + 1, %j] : memref<?x?xf64>
        %left = affine.load %A[%i, %j - 1] : memref<?x?xf64>
        %center = affine.load %A[%i, %j] : memref<?x?xf64>
        %right = affine.load %A[%i, %j + 1] : memref<?x?xf64>
        %top = affine.load %A[%i - 1, %j] : memref<?x?xf64>

        %sum0 = arith.addf %bottom, %left : f64
        %sum1 = arith.addf %sum0, %center : f64
        %sum2 = arith.addf %sum1, %right : f64
        %sum3 = arith.addf %sum2, %top : f64

        %old = affine.load %B[%i, %j] : memref<?x?xf64>
        %result = arith.addf %old, %sum3 : f64
        affine.store %result, %B[%i, %j] : memref<?x?xf64>
      }
    }
    return
  }
}
