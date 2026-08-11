#!/usr/bin/env bash

set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
benchmark_dir="$repository_root/analyzer/misc/benchmark"
misc_dir="$repository_root/analyzer/misc"
results_dir=${RESULTS_DIR:-"$benchmark_dir/results/matmul-t0-t2"}
max_cache_blocks=${MAX_CACHE_BLOCKS:-65536}
cache_step=${CACHE_STEP:-1}
cache_sampling=${CACHE_SAMPLING:-geometric}
cache_growth_factor=${CACHE_GROWTH_FACTOR:-1.5}

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
for index in "${!sources[@]}"; do
    source_name=${sources[$index]}
    mlir_name=${mlir_inputs[$index]}

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
        --max-cache "$max_cache_blocks"
        --sampling "$cache_sampling"
        --growth-factor "$cache_growth_factor"
        --step "$cache_step"
        --output "$results_dir/$source_name.json"
    )
    python3 "$benchmark_dir/parallel_runner.py" "${cachegrind_args[@]}"
done

matplotlib_config_dir=${MPLCONFIGDIR:-/tmp/autolala-matplotlib}
mkdir -p "$matplotlib_config_dir"
env MPLCONFIGDIR="$matplotlib_config_dir" \
    python3 "$repository_root/scripts/graph_mm_salt_vs_cg.py" \
    --data-dir "$results_dir" \
    --svg-output "$results_dir/matmul_t0-t2_salt_cachegrind.svg"

echo "Evaluation artifacts are in $results_dir"
