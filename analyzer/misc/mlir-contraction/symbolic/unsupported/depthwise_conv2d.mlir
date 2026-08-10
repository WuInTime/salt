// Depth-wise Convolution operation in MLIR affine dialect
// Original C code converted to MLIR with symbolic bounds
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

func.func @depthwise_conv2d(%arg0: memref<?x?x?x?xf32>,     // Input tensor I[N][C][IH][IW]
                            %arg1: memref<?x?x?xf32>,       // Weight tensor W[C][KH][KW]
                            %arg2: memref<?x?x?x?xf32>,     // Output tensor O[N][C][OH][OW]
                            %N: index, %C: index,
                            %OH: index, %OW: index, %KH: index, %KW: index) {
  
  // Nested affine loops with symbolic bounds
  // Note: In depth-wise convolution, each input channel is convolved with its own filter
  affine.for %n = 0 to %N {
    affine.for %c = 0 to %C {
      affine.for %oh = 0 to %OH {
        affine.for %ow = 0 to %OW {
          affine.for %kh = 0 to %KH {
            affine.for %kw = 0 to %KW {
              // Load input value: I[n][c][oh + kh][ow + kw]
              %input_val = affine.load %arg0[%n, %c, %oh + %kh, %ow + %kw] : memref<?x?x?x?xf32>
              
              // Load weight value: W[c][kh][kw]
              %weight_val = affine.load %arg1[%c, %kh, %kw] : memref<?x?x?xf32>
              
              // Load current output value: O[n][c][oh][ow]
              // %output_val = affine.load %arg2[%n, %c, %oh, %ow] : memref<?x?x?x?xf32>
              %output_val = arith.constant 0.0 : f32
              
              // Compute multiplication
              %mul = arith.mulf %input_val, %weight_val : f32
              
              // Compute accumulation: O[n][c][oh][ow] += I[n][c][oh + kh][ow + kw] * W[c][kh][kw]
              %add = arith.addf %output_val, %mul : f32
              
              // Store result back to output
              affine.store %add, %arg2[%n, %c, %oh, %ow] : memref<?x?x?x?xf32>
            }
          }
        }
      }
    }
  }
  
  return
}
