#!/usr/bin/env bash

set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
benchmark_dir="$repository_root/analyzer/misc/benchmark"
misc_dir="$repository_root/analyzer/misc"
results_dir=${RESULTS_DIR:-"$benchmark_dir/results/matmul-t0-t2"}
max_cache_blocks=${MAX_CACHE_BLOCKS:-}
cache_step=${CACHE_STEP:-1}

mkdir -p "$results_dir"

cargo build --release -p analyzer --no-default-features --bin analyzer \
    --manifest-path "$repository_root/Cargo.toml"
analyzer="$repository_root/target/release/analyzer"

sources=(matmul matmul-t1 matmul-t2)
mlir_inputs=(
    const_matmul_3acc
    const_matmul_once_tiled
    const_matmul_twice_tiled
)
# Recovered from the marker coordinates in the preserved SVG. T0, T1, and T2
# contain 35, 36, and 38 points respectively.
cache_block_lists=(
    '2,3,4,6,8,9,13,16,19,28,32,42,63,64,94,128,141,211,256,316,474,512,711,1024,1066,1599,2048,2398,3597,4096,5395,8092,8192,16384'
    '2,3,4,6,8,9,13,16,19,28,32,42,63,64,94,128,141,211,256,316,474,512,711,1024,1066,1599,2048,2398,3597,4096,5395,8092,8192,12138,16384'
    '2,3,4,6,8,9,13,16,19,28,32,42,63,64,94,128,141,211,256,316,474,512,711,1024,1066,1599,2048,2398,3597,4096,5395,8092,8192,12138,16384,18207'
)

for index in "${!sources[@]}"; do
    source_name=${sources[$index]}
    mlir_name=${mlir_inputs[$index]}
    cache_blocks=${cache_block_lists[$index]}

    echo "Generating SALT result for $source_name..."
    "$analyzer" \
        -i "$misc_dir/$mlir_name.mlir" \
        --json \
        -o "$results_dir/$source_name-salt.json" \
        salt \
        --block-size=8

    echo "Running Cachegrind sweep for $source_name..."
    cachegrind_args=(
        --src "$benchmark_dir/$source_name.c"
        --block 32
        --output "$results_dir/$source_name.json"
    )
    if [[ -n $max_cache_blocks ]]; then
        cachegrind_args+=(--max-cache "$max_cache_blocks" --step "$cache_step")
    else
        cachegrind_args+=(--blocks "$cache_blocks")
    fi
    python3 "$benchmark_dir/parallel_runner.py" "${cachegrind_args[@]}"
done

MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/autolala-matplotlib} \
python3 "$repository_root/scripts/graph_mm_salt_vs_cg.py" \
    --data-dir "$results_dir" \
    --svg-output "$results_dir/matmul_t0-t2_salt_cachegrind.svg"

echo "Evaluation artifacts are in $results_dir"
