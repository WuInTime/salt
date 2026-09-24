#!/usr/bin/env bash

# RESULTS_DIR=/tmp/mlir-contraction-smoke ./scripts/run-mlir-contraction-evaluation.sh --smoke-test

set -euo pipefail

export SYMBOLICA_HIDE_BANNER=${SYMBOLICA_HIDE_BANNER:-1}

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
contraction_root="$repository_root/benchmarks/mlir-contractions"
constant_dir="$contraction_root/constant"
source_tiled_dir="$constant_dir/tiled"
results_dir=${RESULTS_DIR:-"$repository_root/results/mlir-contractions"}
build_dir="$repository_root/target"
work_dir="$results_dir/work"
staged_constant_dir="$work_dir/constant"
tiled_dir="$staged_constant_dir/tiled"
cache_limit_bytes=${CACHE_LIMIT_BYTES:-65536}
smoke_test=false

if [[ ${1:-} == "--smoke-test" ]]; then
    smoke_test=true
    cache_limit_bytes=${CACHE_LIMIT_BYTES:-1024}
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--smoke-test]" >&2
    exit 2
fi

echo "Building Barvinok/SALT analyzer and Cachegrind runner..."
cargo build --locked --release -p analyzer --bin analyzer \
    --features analyzer/barvinok \
    --manifest-path "$repository_root/Cargo.toml"
cargo build --locked --release -p cachegrind-runner --bin cachegrind-runner \
    --manifest-path "$repository_root/Cargo.toml"

analyzer="$build_dir/release/analyzer"
cachegrind_runner="$build_dir/release/cachegrind-runner"

# Stage inputs only after the potentially long builds. This also recreates the
# output tree if an earlier result directory was cleaned while Cargo ran.
mkdir -p "$tiled_dir" "$results_dir"

for input in "$constant_dir"/constant_*.mlir; do
    cp "$input" "$staged_constant_dir/"
done
for input in "$source_tiled_dir"/tiled_*.mlir; do
    cp "$input" "$tiled_dir/"
done

if [[ ! -f $staged_constant_dir/constant_3d_tensor_vector.mlir ]] ||
   [[ ! -f $tiled_dir/tiled_3d_tensor_vector.mlir ]]; then
    echo "Failed to stage MLIR contraction inputs under $work_dir" >&2
    exit 1
fi

# echo "Regenerating tile-size-8 MLIR..."
# for input in "$staged_constant_dir"/constant_*.mlir; do
#     name=$(basename "$input")
#     name=${name#constant_}
#     mlir-opt "$input" \
#         --affine-loop-tile="tile-size=8" \
#         --affine-loop-normalize="promote-single-iter" \
#         --affine-simplify-structures \
#         --canonicalize \
#         -o "$tiled_dir/tiled_$name"
# done

if $smoke_test; then
    inputs=(
        # Use staged inputs so SALT output and regenerated tiled MLIR never
        # modify the checked-in benchmark artifacts.
        "$staged_constant_dir/constant_3d_tensor_vector.mlir"
        "$tiled_dir/tiled_3d_tensor_vector.mlir"
    )
else
    inputs=(
        "$staged_constant_dir"/constant_*.mlir
        "$tiled_dir"/tiled_*.mlir
    )
fi

fully_db="$results_dir/data-fully-associative.db"
way8_db="$results_dir/data-8way-associative.db"
way12_db="$results_dir/data-12way-associative.db"
simulation_timing="$results_dir/timing-simulation-wall.tsv"

# Keep the redirection below safe even if an external cleanup happened after
# staging, and report missing staged inputs before starting any simulations.
mkdir -p "$results_dir"
for input in "${inputs[@]}"; do
    if [[ ! -f $input ]]; then
        echo "Staged MLIR input disappeared before evaluation: $input" >&2
        exit 1
    fi
done

for database in "$fully_db" "$way8_db" "$way12_db"; do
    if [[ -e $database ]]; then
        echo "Refusing to mix new records with existing database: $database" >&2
        echo "Choose an empty RESULTS_DIR." >&2
        exit 1
    fi
done

printf 'kernel\tseconds\n' > "$simulation_timing"

for input in "${inputs[@]}"; do
    program=$(basename "$input" .mlir)
    if [[ $program == tiled_* ]]; then
        salt_output="$tiled_dir/$program-salt.json"
    else
        salt_output="$staged_constant_dir/$program-salt.json"
    fi

    echo "Generating SALT prediction for $program..."
    "$analyzer" \
        -i "$input" \
        --json \
        -o "$salt_output" \
        salt \
        --block-size=8

    echo "Running fully associative Cachegrind sweep for $program..."
    fully_start_ns=$(date +%s%N)
    "$cachegrind_runner" \
        -i "$input" \
        -C "$cache_limit_bytes" -B64 \
        -c65536 -b64 -a16 \
        --database "$fully_db" \
        --batched
    fully_end_ns=$(date +%s%N)
    if [[ $program == tiled_* ]]; then
        kernel=${program#tiled_}
        elapsed_ns=$((fully_end_ns - fully_start_ns))
        printf '%s\t%d.%09d\n' \
            "$kernel" "$((elapsed_ns / 1000000000))" "$((elapsed_ns % 1000000000))" \
            >> "$simulation_timing"
    fi

    echo "Running 8-way Cachegrind sweep for $program..."
    "$cachegrind_runner" \
        -i "$input" \
        -C "$cache_limit_bytes" -B64 -A8 \
        -c65536 -b64 -a16 \
        --database "$way8_db" \
        --batched

    echo "Running 12-way Cachegrind sweep for $program..."
    "$cachegrind_runner" \
        -i "$input" \
        -C "$cache_limit_bytes" -B64 -A12 \
        -c65536 -b64 -a16 \
        --database "$way12_db" \
        --batched
done

echo "Generating log- and linear-scale figures..."
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/salt-matplotlib} \
python3 "$repository_root/scripts/graph_contractions_salt_vs_cg.py" \
    --fully-db "$fully_db" \
    --8way-db "$way8_db" \
    --12way-db "$way12_db" \
    --constant-dir "$staged_constant_dir" \
    --output-dir "$results_dir"

if ! $smoke_test; then
    echo "Measuring symbolic Barvinok/SALT and generating selected timing figure..."
    python3 "$repository_root/scripts/measure_timing_selected.py" \
        --analyzer "$analyzer" \
        --simulation-times "$simulation_timing" \
        --simulation-db "$fully_db" \
        --cache-limit-bytes "$cache_limit_bytes" \
        --out "$results_dir/timing-selected.json"
    MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/salt-matplotlib} \
    python3 "$repository_root/scripts/timing_selected.py" \
        --data "$results_dir/timing-selected.json" \
        --out "$results_dir/timing-selected.svg"
else
    echo "Skipping the eight-kernel timing figure in smoke-test mode."
fi

echo "Evaluation artifacts are in $results_dir"
