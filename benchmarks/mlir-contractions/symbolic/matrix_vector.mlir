module {
    func.func @matrix_vector_contraction(%A: memref<?x?xf64>, %B: memref<?xf64>, %C: memref<?xf64>, %M: index, %N: index) {
        // C[i] = A[i][j] * B[j]
        affine.for %i = 0 to %M {
            affine.for %j = 0 to %N {
                %A_ij = affine.load %A[%i, %j] : memref<?x?xf64>
                %B_j = affine.load %B[%j] : memref<?xf64>
                %prod = arith.mulf %A_ij, %B_j : f64
                affine.store %prod, %C[%i] : memref<?xf64>
            }
        }
        return
    }
}