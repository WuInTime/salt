// Row-wise Softmax (Max computation) in MLIR affine dialect with constant bounds
// CPU-feasible sizes with commonly used shapes for transformer models
// 
// \begin{table}[h]
//   \centering
//   \begin{tabular}{|c|c|l|}
//     \hline
//     Parameter & Value & Description \\
//     \hline
//     B & 4 & Batch size \\
//     H & 16 & Number of attention heads \\
//     S_q & 512 & Query sequence length \\
//     S_k & 512 & Key sequence length \\
//     \hline
//   \end{tabular}
//   \caption{Constant assignments for row-wise softmax max computation}
// \end{table}
// 
// Original C code:
// for (int b = 0; b < B; ++b)
//   for (int h = 0; h < H; ++h)
//     for (int q = 0; q < S_q; ++q)
//       for (int k = 0; k < S_k; ++k)
//         M[b][h][q] = fmaxf(M[b][h][q], S[b][h][q][k]);
module attributes { "simulation.prologue" = "volatile double ARRAY_0[5][17][541][536], ARRAY_1[5][17][536];" } {
func.func @constant_rowwise_softmax_max(%arg0: memref<4x16x512x512xf32>,     // Score tensor S[5][17][512][512]
                                        %arg1: memref<4x16x512xf32>) {      // Max tensor M[5][17][512]
  
  // Nested affine loops with constant bounds
  // This implements the first step of row-wise softmax: finding max values for numerical stability
  affine.for %b = 0 to 4 {
    affine.for %h = 0 to 16 {
      affine.for %q = 0 to 512 {
        affine.for %k = 0 to 512 {
          // Load current score value: S[b][h][q][k]
          %score_val = affine.load %arg0[%b, %h, %q, %k] : memref<4x16x512x512xf32>
          
          // Load current max value: M[b][h][q]
          // %max_val = affine.load %arg1[%b, %h, %q] : memref<4x16x512xf32>
          %max_val = arith.constant 0.0 : f32
          
          // Compute maximum: M[b][h][q] = fmaxf(M[b][h][q], S[b][h][q][k])
          %new_max = arith.maximumf %max_val, %score_val : f32
          
          // Store result back to max tensor
          affine.store %new_max, %arg1[%b, %h, %q] : memref<4x16x512xf32>
        }
      }
    }
  }
  
  return
}
}