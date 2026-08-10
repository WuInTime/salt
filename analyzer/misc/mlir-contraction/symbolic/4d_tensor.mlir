module {
    func.func @tensor4d_contraction(%A: memref<?x?x?x?xf64>, %B: memref<?x?xf64>, %C: memref<?x?xf64>, %M: index, %N: index, %K: index, %L: index) {
        // C[i][j] = A[i][k][l][j] * B[k][l]
        affine.for %i = 0 to %M {
            affine.for %j = 0 to %N {
                affine.for %k = 0 to %K {
                    affine.for %l = 0 to %L {
                        %A_iklj = affine.load %A[%i, %k, %l, %j] : memref<?x?x?x?xf64>
                        %B_kl = affine.load %B[%k, %l] : memref<?x?xf64>
                        %prod = arith.mulf %A_iklj, %B_kl : f64
                        affine.store %prod, %C[%i, %j] : memref<?x?xf64>
                    }
                }
            }
        }
        return
    }
}