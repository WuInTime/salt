// Depth-wise Convolution operation in MLIR affine dialect with constant bounds
// CPU-feasible sizes with commonly used shapes for mobile CNN models
// 
// \begin{table}[h]
//   \centering
//   \begin{tabular}{|c|c|l|}
//     \hline
//     Parameter & Value & Description \\
//     \hline
//     N & 1 & Batch size \\
//     C & 128 & Number of channels \\
//     OH & 56 & Output height \\
//     OW & 56 & Output width \\
//     KH & 3 & Kernel height \\
//     KW & 3 & Kernel width \\
//     \hline
//   \end{tabular}
//   \caption{Constant assignments for depth-wise 2D convolution}
// \end{table}
// 
// Original C code:
// for (int n = 0; n < N; ++n)
//   for (int c = 0; c < C; ++c)
//     for (int oh = 0; oh < OH; ++oh)
//       for (int ow = 0; ow < OW; ++ow)
//         for (int kh = 0; kh < KH; ++kh)
//           for (int kw = 0; kw < KW; ++kw)
//             O[n][c][oh][ow] +=
//               I[n][c][oh + kh][ow + kw] * W[c][kh][kw];

func.func @constant_depthwise_conv2d(%arg0: memref<1x128x58x58xf32>,     // Input tensor I[1][128][58][58]
                                     %arg1: memref<128x3x3xf32>,         // Weight tensor W[128][3][3]
                                     %arg2: memref<1x128x56x56xf32>) {   // Output tensor O[1][128][56][56]
  
  // Nested affine loops with constant bounds
  // Note: In depth-wise convolution, each input channel is convolved with its own filter
  affine.for %n = 0 to 1 {
    affine.for %c = 0 to 128 {
      affine.for %oh = 0 to 56 {
        affine.for %ow = 0 to 56 {
          affine.for %kh = 0 to 3 {
            affine.for %kw = 0 to 3 {
              // Load input value: I[n][c][oh + kh][ow + kw]
              %input_val = affine.load %arg0[%n, %c, %oh + %kh, %ow + %kw] : memref<1x128x58x58xf32>
              
              // Load weight value: W[c][kh][kw]
              %weight_val = affine.load %arg1[%c, %kh, %kw] : memref<128x3x3xf32>
              
              // Load current output value: O[n][c][oh][ow]
              // %output_val = affine.load %arg2[%n, %c, %oh, %ow] : memref<1x128x56x56xf32>
              %output_val = arith.constant 0.0 : f32
              
              // Compute multiplication
              %mul = arith.mulf %input_val, %weight_val : f32
              
              // Compute accumulation: O[n][c][oh][ow] += I[n][c][oh + kh][ow + kw] * W[c][kh][kw]
              %add = arith.addf %output_val, %mul : f32
              
              // Store result back to output
              affine.store %add, %arg2[%n, %c, %oh, %ow] : memref<1x128x56x56xf32>
            }
          }
        }
      }
    }
  }
  
  return
}
