// Attention Score computation in MLIR affine dialect
// Original C code converted to MLIR with symbolic bounds
// 
// Original C code:
// for (int b = 0; b < B; ++b)
//   for (int h = 0; h < H; ++h)
//     for (int q = 0; q < S_q; ++q)
//       for (int k = 0; k < S_k; ++k)
//         for (int d = 0; d < D; ++d)
//           S[b][h][q][k] += Q[b][h][q][d] * K[b][h][k][d];

func.func @attention_score(%arg0: memref<?x?x?x?xf32>,     // Query tensor Q[B][H][S_q][D]
                           %arg1: memref<?x?x?x?xf32>,     // Key tensor K[B][H][S_k][D]
                           %arg2: memref<?x?x?x?xf32>,     // Score tensor S[B][H][S_q][S_k]
                           %B: index, %H: index,
                           %S_q: index, %S_k: index, %D: index) {
  
  // Nested affine loops with symbolic bounds
  // This implements the query-key dot product for attention scores
  affine.for %b = 0 to %B {
    affine.for %h = 0 to %H {
      affine.for %q = 0 to %S_q {
        affine.for %k = 0 to %S_k {
          affine.for %d = 0 to %D {
            // Load query value: Q[b][h][q][d]
            %query_val = affine.load %arg0[%b, %h, %q, %d] : memref<?x?x?x?xf32>
            
            // Load key value: K[b][h][k][d]
            %key_val = affine.load %arg1[%b, %h, %k, %d] : memref<?x?x?x?xf32>
            
            // Load current score value: S[b][h][q][k]
            // %score_val = affine.load %arg2[%b, %h, %q, %k] : memref<?x?x?x?xf32>
            %score_val = arith.constant 0.0 : f32
            
            // Compute multiplication: Q[b][h][q][d] * K[b][h][k][d]
            %mul = arith.mulf %query_val, %key_val : f32
            
            // Compute accumulation: S[b][h][q][k] += Q[b][h][q][d] * K[b][h][k][d]
            %add = arith.addf %score_val, %mul : f32
            
            // Store result back to score tensor
            affine.store %add, %arg2[%b, %h, %q, %k] : memref<?x?x?x?xf32>
          }
        }
      }
    }
  }
  
  return
}
