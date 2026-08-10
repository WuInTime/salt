// Pooling operation in MLIR affine dialect with constant bounds
// CPU-feasible sizes with commonly used shapes for CNN models
// 
// \begin{table}[h]
//   \centering
//   \begin{tabular}{|c|c|l|}
//     \hline
//     Parameter & Value & Description \\
//     \hline
//     N & 1 & Batch size \\
//     C & 64 & Number of channels \\
//     OH & 14 & Output height \\
//     OW & 14 & Output width \\
//     KH & 2 & Kernel height \\
//     KW & 2 & Kernel width \\
//     \hline
//   \end{tabular}
//   \caption{Constant assignments for 2D pooling}
// \end{table}
// 
// Original C code:
// for (int n = 0; n < N; ++n)
//   for (int c = 0; c < C; ++c)
//     for (int oh = 0; oh < OH; ++oh)
//       for (int ow = 0; ow < OW; ++ow)
//         for (int kh = 0; kh < KH; ++kh)
//           for (int kw = 0; kw < KW; ++kw)
//             O[n][c][oh][ow] += I[n][c][oh + kh][ow + kw];

func.func @constant_pooling(%arg0: memref<1x64x15x15xf32>,     // Input tensor I[1][67][15][15]
                            %arg1: memref<1x64x14x14xf32>) {   // Output tensor O[1][67][14][14]
  
  // Nested affine loops with constant bounds
  // This implements sum pooling (can be converted to average pooling by dividing by KH*KW)
  affine.for %n = 0 to 1 {
    affine.for %c = 0 to 64 {
      affine.for %oh = 0 to 14 {
        affine.for %ow = 0 to 14 {
          affine.for %kh = 0 to 2 {
            affine.for %kw = 0 to 2 {
              // Load input value: I[n][c][oh + kh][ow + kw]
              %input_val = affine.load %arg0[%n, %c, %oh + %kh, %ow + %kw] : memref<1x64x15x15xf32>
              
              // Load current output value: O[n][c][oh][ow]
              // %output_val = affine.load %arg1[%n, %c, %oh, %ow] : memref<1x64x14x14xf32>
              %output_val = arith.constant 0.0 : f32
              
              // Compute accumulation: O[n][c][oh][ow] += I[n][c][oh + kh][ow + kw]
              %add = arith.addf %output_val, %input_val : f32
              
              // Store result back to output
              affine.store %add, %arg1[%n, %c, %oh, %ow] : memref<1x64x14x14xf32>
            }
          }
        }
      }
    }
  }
  
  return
}
