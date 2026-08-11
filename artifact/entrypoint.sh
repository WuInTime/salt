#!/usr/bin/env bash

set -euo pipefail

cd /artifact

results_root=${RESULTS_ROOT:-/artifact/results}

require_absent() {
    local path
    for path in "$@"; do
        if [[ -e $path ]]; then
            echo "Refusing to mix results: $path already exists." >&2
            echo "Mount or select an empty results directory." >&2
            exit 1
        fi
    done
}

print_versions() {
    echo "Artifact revision: ${ARTIFACT_REVISION:-unknown}"

    echo "System:"
    uname -a

    echo "Rust:"
    rustc --version
    cargo --version

    echo "LLVM/MLIR:"
    mlir-opt --version
    clang++ --version | head -n 1

    echo "Experiment tools:"
    valgrind --version
    python3 --version

    echo "Python packages:"
    python3 - <<'PY'
import matplotlib
import numpy
import pandas

print("matplotlib", matplotlib.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
PY
}

run_tests() {
    local analyzer_test
    local -a analyzer_tests=(
        salt::tests::coefficient_check_inspects_every_map_result
        salt::tests::reuse_check_does_not_leak_ivars_between_sibling_loops
        salt::tests::reuse_check_detects_an_omitted_enclosing_ivar
        salt::tests::perfect_nesting_requires_a_nonempty_access_body
        salt::tests::curve_adjustments_accumulate_at_tied_highest_intervals
        salt::tests::reference_adjustment_is_normalized_by_unreferenced_trip_counts
    )

    # Symbolica's free restricted mode permits one instance at a time. Libtest
    # creates a fresh worker thread for each test even with --test-threads=1,
    # so isolate analyzer tests in separate processes. Run all other workspace
    # tests in the usual process first.
    cargo test --release --locked --workspace -- --skip salt::tests::
    for analyzer_test in "${analyzer_tests[@]}"; do
        cargo test --release --locked --workspace "$analyzer_test" -- \
            --exact --test-threads=1
    done
}

case "${1:-help}" in
    test)
        print_versions
        run_tests
        ;;

    smoke)
        contraction_results="$results_root/contraction-smoke"
        matmul_results="$results_root/matmul-smoke"

        mkdir -p "$results_root"
        require_absent "$contraction_results" "$matmul_results"
        print_versions | tee "$results_root/environment-smoke.txt"

        RESULTS_DIR="$contraction_results" \
            bash scripts/run-mlir-contraction-evaluation.sh --smoke-test

        RESULTS_DIR="$matmul_results" \
        MAX_CACHE_BLOCKS=8 \
            bash scripts/run-matmul-t0-t2-evaluation.sh

        echo "Smoke tests completed successfully."
        echo "Results: $results_root"
        ;;

    reproduce)
        contraction_results="$results_root/mlir-contractions"
        matmul_results="$results_root/matmul-t0-t2"

        mkdir -p "$results_root"
        require_absent "$contraction_results" "$matmul_results"
        print_versions | tee "$results_root/environment-full.txt"

        RESULTS_DIR="$contraction_results" \
            bash scripts/run-mlir-contraction-evaluation.sh

        RESULTS_DIR="$matmul_results" \
            bash scripts/run-matmul-t0-t2-evaluation.sh

        echo "Full evaluation completed."
        echo "Results: $results_root"
        ;;

    shell)
        shift
        exec /bin/bash "$@"
        ;;

    help|--help|-h)
        cat <<'EOF'
Usage:
  autolala-artifact test
  autolala-artifact smoke
  autolala-artifact reproduce
  autolala-artifact shell

Commands:
  test       Check versions and run all Rust tests (free Symbolica mode safe)
  smoke      Run reduced versions of both experiments
  reproduce  Run the complete evaluation
  shell      Open an interactive shell
EOF
        ;;

    *)
        echo "Unknown command: $1" >&2
        exit 2
        ;;
esac
