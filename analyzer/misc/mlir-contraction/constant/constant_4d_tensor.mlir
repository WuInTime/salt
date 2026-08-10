module attributes { "simulation.prologue" = "volatile double ARRAY_0[83][53][37][88], ARRAY_1[53][40], ARRAY_2[83][88];" } {
    func.func @tensor4d_contraction(%A: memref<80x48x32x64xf64>, %B: memref<48x32xf64>, %C: memref<80x64xf64>) {
        // C[i][j] = A[i][k][l][j] * B[k][l]
        affine.for %i = 0 to 80 {
            affine.for %j = 0 to 64 {
                affine.for %k = 0 to 48 {
                    affine.for %l = 0 to 32 {
                        %A_iklj = affine.load %A[%i, %k, %l, %j] : memref<80x48x32x64xf64>
                        %B_kl = affine.load %B[%k, %l] : memref<48x32xf64>
                        %prod = arith.mulf %A_iklj, %B_kl : f64
                        affine.store %prod, %C[%i, %j] : memref<80x64xf64>
                    }
                }
            }
        }
        return
    }
}