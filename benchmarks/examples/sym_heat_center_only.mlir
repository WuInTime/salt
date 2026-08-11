module {
  func.func @heat_distribution(%A: memref<?x?xf64>, %B: memref<?x?xf64>, %N: index, %M: index) {
    affine.for %i = 1 to %N {
      affine.for %j = 1 to %M {
        %center = affine.load %A[%i, %j] : memref<?x?xf64>
        affine.store %center, %B[%i, %j] : memref<?x?xf64>
      }
    }
    return
  }
}
