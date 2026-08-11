module {
  func.func @stencil5pt_tiled(%A: memref<?x?xf64>, %B: memref<?x?xf64>,
                              %N: index, %M: index, %Ti: index, %Tj: index) {

    affine.for %tile_i = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%N, %Ti] {
      affine.for %tile_j = 0 to affine_map<()[s0, s1] -> (s0 floordiv s1)>()[%M, %Tj] {
        affine.for %i = 0 to %Ti {
          affine.for %j = 0 to %Tj {
            %center = affine.load %A[%tile_i * symbol(%Ti) + %i + 1, %tile_j * symbol(%Tj) + %j + 1] : memref<?x?xf64>
            %top = affine.load %A[%tile_i * symbol(%Ti) + %i, %tile_j * symbol(%Tj) + %j + 1] : memref<?x?xf64>
            %bottom = affine.load %A[%tile_i * symbol(%Ti) + %i + 2, %tile_j * symbol(%Tj) + %j + 1] : memref<?x?xf64>
            %left = affine.load %A[%tile_i * symbol(%Ti) + %i + 1, %tile_j * symbol(%Tj) + %j] : memref<?x?xf64>
            %right = affine.load %A[%tile_i * symbol(%Ti) + %i + 1, %tile_j * symbol(%Tj) + %j + 2] : memref<?x?xf64>

            %sum0 = arith.addf %center, %top : f64
            %sum1 = arith.addf %sum0, %bottom : f64
            %sum2 = arith.addf %sum1, %left : f64
            %sum3 = arith.addf %sum2, %right : f64
            affine.store %sum3, %B[%tile_i * symbol(%Ti) + %i + 1, %tile_j * symbol(%Tj) + %j + 1] : memref<?x?xf64>
          }
        }
      }
    }
    return
  }
}
