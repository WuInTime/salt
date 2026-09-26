#!/usr/bin/env bash

set -euo pipefail

export SYMBOLICA_HIDE_BANNER=${SYMBOLICA_HIDE_BANNER:-1}

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
package_dir="$repository_root/salt_vs_hw_misses_package"
results_dir=${RESULTS_DIR:-"$repository_root/results/salt-vs-hardware"}
salt_json_dir="$results_dir/salt-json"
pmc_input="$package_dir/data/pmu_i7-7700_result.csv"
smoke_test=false
cache_size_bytes=32768
cache_line_bytes=64
elements_per_cache_line=8

usage() {
    cat >&2 <<EOF
Usage: $0 [--smoke-test] [--pmc FILE]
          [--cache-size-bytes N] [--cache-line-bytes N]
          [--elements-per-cache-line N]

Without --pmc, the packaged Intel i7-7700 measurements are used.
Fresh hardware-counter collection is documented in $package_dir/README.md.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)
            smoke_test=true
            shift
            ;;
        --pmc|--cache-size-bytes|--cache-line-bytes|--elements-per-cache-line)
            if [[ $# -lt 2 ]]; then
                usage
                exit 2
            fi
            case $1 in
                --pmc) pmc_input=$2 ;;
                --cache-size-bytes) cache_size_bytes=$2 ;;
                --cache-line-bytes) cache_line_bytes=$2 ;;
                --elements-per-cache-line) elements_per_cache_line=$2 ;;
            esac
            shift 2
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

mkdir -p "$results_dir"

analyzer=/usr/local/bin/analyzer
if [[ ! -x $analyzer ]]; then
    echo "Required artifact binary is missing or not executable: $analyzer" >&2
    echo "Rebuild the Docker image before running this evaluation." >&2
    exit 1
fi

python_args=(
    --analyzer "$analyzer"
    --salt-json-dir "$salt_json_dir"
    --pmc "$pmc_input"
    --results-output "$results_dir/salt_vs_hw_misses_results.csv"
    --plot-output "$results_dir/salt_vs_hw_misses.svg"
    --cache-size-bytes "$cache_size_bytes"
    --cache-line-bytes "$cache_line_bytes"
    --elements-per-cache-line "$elements_per_cache_line"
)

if $smoke_test; then
    python_args+=(--smoke-test)
fi

MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/salt-matplotlib} \
python3 "$package_dir/salt_vs_hw_misses.py" "${python_args[@]}"

echo "Evaluation artifacts are in $results_dir"
