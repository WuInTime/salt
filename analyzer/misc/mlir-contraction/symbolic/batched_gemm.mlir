module {
    func.func @batched_gemm(%A: memref<?x?x?xf64>, %B: memref<?x?x?xf64>, %C: memref<?x?x?xf64>, %P: index, %M: index, %N: index, %K: index) {
        // C[p][i][j] = A[p][i][k] * B[p][k][j]
        affine.for %p = 0 to %P {
            affine.for %i = 0 to %M {
                affine.for %j = 0 to %N {
                    affine.for %k = 0 to %K {
                        %A_pik = affine.load %A[%p, %i, %k] : memref<?x?x?xf64>
                        %B_pkj = affine.load %B[%p, %k, %j] : memref<?x?x?xf64>
                        %prod = arith.mulf %A_pik, %B_pkj : f64
                        affine.store %prod, %C[%p, %i, %j] : memref<?x?x?xf64>
                    }
                }
            }
        }
        return
    }
}