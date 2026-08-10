module attributes { "simulation.prologue" = "volatile double ARRAY_0[541][1048], ARRAY_1[1048], ARRAY_2[536];" } {
    func.func @matrix_vector_contraction(%A: memref<512x1024xf64>, %B: memref<1024xf64>, %C: memref<512xf64>) {
        // C[i] = A[i][j] * B[j]
        affine.for %i = 0 to 512 {
            affine.for %j = 0 to 1024 {
                %A_ij = affine.load %A[%i, %j] : memref<512x1024xf64>
                %B_j = affine.load %B[%j] : memref<1024xf64>
                %prod = arith.mulf %A_ij, %B_j : f64
                affine.store %prod, %C[%i] : memref<512xf64>
            }
        }
        return
    }
}