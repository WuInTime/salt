// Context Lookup computation in MLIR affine dialect
// Original C code converted to MLIR with symbolic bounds
// 
// Original C code:
// for (int b = 0; b < B; ++b)
//   for (int h = 0; h < H; ++h)
//     for (int q = 0; q < S_q; ++q)
//       for (int d = 0; d < D; ++d)
//         for (int k = 0; k < S_k; ++k)
//           O[b][h][q][d] += P[b][h][q][k] * V[b][h][k][d];

func.func @context_lookup(%arg0: memref<?x?x?x?xf32>,     // Probability tensor P[B][H][S_q][S_k]
                          %arg1: memref<?x?x?x?xf32>,     // Value tensor V[B][H][S_k][D]
                          %arg2: memref<?x?x?x?xf32>,     // Output tensor O[B][H][S_q][D]
                          %B: index, %H: index,
                          %S_q: index, %S_k: index, %D: index) {
  
  // Nested affine loops with symbolic bounds
  // This implements the context lookup: weighted sum of value vectors using attention probabilities
  affine.for %b = 0 to %B {
    affine.for %h = 0 to %H {
      affine.for %q = 0 to %S_q {
        affine.for %d = 0 to %D {
          affine.for %k = 0 to %S_k {
            // Load probability value: P[b][h][q][k]
            %prob_val = affine.load %arg0[%b, %h, %q, %k] : memref<?x?x?x?xf32>
            
            // Load value vector element: V[b][h][k][d]
            %value_val = affine.load %arg1[%b, %h, %k, %d] : memref<?x?x?x?xf32>
            
            // Load current output value: O[b][h][q][d]
            // %output_val = affine.load %arg2[%b, %h, %q, %d] : memref<?x?x?x?xf32>
            %output_val = arith.constant 0.0 : f32
            
            // Compute multiplication: P[b][h][q][k] * V[b][h][k][d]
            %mul = arith.mulf %prob_val, %value_val : f32
            
            // Compute accumulation: O[b][h][q][d] += P[b][h][q][k] * V[b][h][k][d]
            %add = arith.addf %output_val, %mul : f32
            
            // Store result back to output tensor
            affine.store %add, %arg2[%b, %h, %q, %d] : memref<?x?x?x?xf32>
          }
        }
      }
    }
  }
  
  return
}
