// Convolution operation in MLIR affine dialect
// Original C code converted to MLIR with symbolic bounds

func.func @conv2d(%arg0: memref<?x?x?x?xf32>,     // Input tensor I[N][IC][IH][IW]
                  %arg1: memref<?x?x?x?xf32>,     // Weight tensor W[OC][IC][KH][KW]
                  %arg2: memref<?x?x?x?xf32>,     // Output tensor O[N][OC][OH][OW]
                  %N: index, %OC: index, %IC: index,
                  %OH: index, %OW: index, %KH: index, %KW: index) {
  
  // Nested affine loops with symbolic bounds
  affine.for %n = 0 to %N {
    affine.for %oc = 0 to %OC {
      affine.for %ic = 0 to %IC {
        affine.for %oh = 0 to %OH {
          affine.for %ow = 0 to %OW {
            affine.for %kh = 0 to %KH {
              affine.for %kw = 0 to %KW {
                // Load input value: I[n][ic][oh + kh][ow + kw]
                %input_val = affine.load %arg0[%n, %ic, %oh + %kh, %ow + %kw] : memref<?x?x?x?xf32>
                
                // Load weight value: W[oc][ic][kh][kw]
                %weight_val = affine.load %arg1[%oc, %ic, %kh, %kw] : memref<?x?x?x?xf32>
                
                // Load current output value: O[n][oc][oh][ow]
                // %output_val = affine.load %arg2[%n, %oc, %oh, %ow] : memref<?x?x?x?xf32>
                %output_val = arith.constant 0.0 : f32
                
                // Compute multiplication
                %mul = arith.mulf %input_val, %weight_val : f32
                
                // Compute accumulation: O[n][oc][oh][ow] += I[n][ic][oh + kh][ow + kw] * W[oc][ic][kh][kw]
                %add = arith.addf %output_val, %mul : f32
                
                // Store result back to output
                affine.store %add, %arg2[%n, %oc, %oh, %ow] : memref<?x?x?x?xf32>
              }
            }
          }
        }
      }
    }
  }
  
  return
}
