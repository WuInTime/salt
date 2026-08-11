module {
    func.func @tensor3d_vector_contraction(%A: memref<?x?x?xf64>, %B: memref<?xf64>, %C: memref<?x?xf64>, %M: index, %N: index, %K: index) {
        // C[i][j] = A[i][j][k] * B[k]
        affine.for %i = 0 to %M {
            affine.for %j = 0 to %N {
                affine.for %k = 0 to %K {
                    %A_ijk = affine.load %A[%i, %j, %k] : memref<?x?x?xf64>
                    %B_k = affine.load %B[%k] : memref<?xf64>
                    %prod = arith.mulf %A_ijk, %B_k : f64
                    affine.store %prod, %C[%i, %j] : memref<?x?xf64>
                }
            }
        }
        return
    }
}