module attributes { "simulation.prologue" = "volatile double ARRAY_0[97][67][56], ARRAY_1[56], ARRAY_2[97][88];" } {
    func.func @tensor3d_vector_contraction(%A: memref<96x64x48xf64>, %B: memref<48xf64>, %C: memref<96x64xf64>) {
        // C[i][j] = A[i][j][k] * B[k]
        affine.for %i = 0 to 96 {
            affine.for %j = 0 to 64 {
                affine.for %k = 0 to 48 {
                    %A_ijk = affine.load %A[%i, %j, %k] : memref<96x64x48xf64>
                    %B_k = affine.load %B[%k] : memref<48xf64>
                    %prod = arith.mulf %A_ijk, %B_k : f64
                    affine.store %prod, %C[%i, %j] : memref<96x64xf64>
                }
            }
        }
        return
    }
}