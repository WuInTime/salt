// Row-wise Softmax (Max computation) in MLIR affine dialect
// Original C code converted to MLIR with symbolic bounds
// 
// Original C code:
// for (int b = 0; b < B; ++b)
//   for (int h = 0; h < H; ++h)
//     for (int q = 0; q < S_q; ++q)
//       for (int k = 0; k < S_k; ++k)
//         M[b][h][q] = fmaxf(M[b][h][q], S[b][h][q][k]);

func.func @rowwise_softmax_max(%arg0: memref<?x?x?x?xf32>,     // Score tensor S[B][H][S_q][S_k]
                               %arg1: memref<?x?x?xf32>,       // Max tensor M[B][H][S_q]
                               %B: index, %H: index,
                               %S_q: index, %S_k: index) {
  
  // Nested affine loops with symbolic bounds
  // This implements the first step of row-wise softmax: finding max values for numerical stability
  affine.for %b = 0 to %B {
    affine.for %h = 0 to %H {
      affine.for %q = 0 to %S_q {
        affine.for %k = 0 to %S_k {
          // Load current score value: S[b][h][q][k]
          %score_val = affine.load %arg0[%b, %h, %q, %k] : memref<?x?x?x?xf32>
          
          // Load current max value: M[b][h][q]
          // %max_val = affine.load %arg1[%b, %h, %q] : memref<?x?x?xf32>
          %max_val = arith.constant 0.0 : f32
          
          // Compute maximum: M[b][h][q] = fmaxf(M[b][h][q], S[b][h][q][k])
          %new_max = arith.maximumf %max_val, %score_val : f32
          
          // Store result back to max tensor
          affine.store %new_max, %arg1[%b, %h, %q] : memref<?x?x?xf32>
        }
      }
    }
  }
  
  return
}
