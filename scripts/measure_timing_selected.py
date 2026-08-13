#!/usr/bin/env python3
"""Measure the three series used by timing_selected.py."""

import argparse
import json
import platform
import sqlite3
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACTION_ROOT = REPOSITORY_ROOT / "benchmarks" / "mlir-contractions"
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "results" / "mlir-contractions"
KERNELS = {
    "3D Tensor Vector": "3d_tensor_vector",
    "4D Tensor": "4d_tensor",
    "Attention Score": "attention_score",
    "Batched GEMM": "batched_gemm",
    "Context Lookup": "context_lookup",
    "Matrix Matrix": "matrix_matrix",
    "Matrix Vector": "matrix_vector",
    "Rowwise SoftMax": "rowwise_softmax_max",
}


def command_version(command):
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


def measure(command, repetitions, cwd, label):
    samples = []
    for repetition in range(1, repetitions + 1):
        print(f"[{label}] repetition {repetition}/{repetitions}", flush=True)
        start = time.perf_counter()
        subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        samples.append(time.perf_counter() - start)
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Measure only the series used in the selected timing graph."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "timing-selected.json",
        help="output timing manifest",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--cache-limit-bytes", type=int, default=65536)
    parser.add_argument(
        "--simulation-times",
        type=Path,
        help="reuse tiled fully-associative wall times from the evaluation TSV",
    )
    parser.add_argument(
        "--simulation-db",
        type=Path,
        help="fully-associative database corresponding to --simulation-times",
    )
    parser.add_argument(
        "--analyzer", type=Path, default=REPOSITORY_ROOT / "target/release/analyzer"
    )
    parser.add_argument(
        "--cachegrind-runner",
        type=Path,
        default=REPOSITORY_ROOT / "target/release/cachegrind-runner",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    analyzer = args.analyzer.resolve()
    runner = args.cachegrind_runner.resolve()
    required_executables = (analyzer,) if args.simulation_times else (analyzer, runner)
    for executable in required_executables:
        if not executable.is_file():
            parser.error(f"missing executable: {executable}")
    help_text = subprocess.run(
        [analyzer, "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout
    if "barvinok" not in help_text:
        parser.error("analyzer was built without the default Barvinok feature")

    measurements = {
        "Simulation Fully (Tiled)": {},
        "Barvinok (Symbolic)": {},
        "SALT (Symbolic)": {},
    }
    samples = {series: {} for series in measurements}
    commands = {series: {} for series in measurements}
    observed_cache_sizes = {}
    reused_simulation_times = {}
    if args.simulation_times:
        lines = args.simulation_times.read_text().splitlines()
        for line in lines[1:]:
            stem, seconds = line.split("\t")
            reused_simulation_times[stem] = float(seconds)
        missing = sorted(set(KERNELS.values()) - set(reused_simulation_times))
        if missing:
            parser.error(f"simulation timing file is missing kernels: {', '.join(missing)}")
        if not args.simulation_db:
            parser.error("--simulation-db is required with --simulation-times")

    with tempfile.TemporaryDirectory(prefix="salt-selected-timing-") as temporary:
        temporary_root = Path(temporary)
        for display_name, stem in KERNELS.items():
            symbolic = CONTRACTION_ROOT / "symbolic" / f"{stem}.mlir"
            tiled = CONTRACTION_ROOT / "constant" / "tiled" / f"tiled_{stem}.mlir"
            method_commands = {
                "Simulation Fully (Tiled)": [
                    str(runner), "-i", str(tiled), "-C", str(args.cache_limit_bytes),
                    "-B64", "-c65536", "-b64", "-a16", "--batched",
                ],
                "Barvinok (Symbolic)": [
                    str(analyzer), "-i", str(symbolic), "-m", "/dev/null",
                    "barvinok", "--block-size=8", "--barvinok-arg=--approximation-method=scale", "--infinite-repeat"
                ],
                "SALT (Symbolic)": [
                    str(analyzer), "-i", str(symbolic), "-m", "/dev/null",
                    "salt", "--block-size=8",
                ],
            }
            for series, command in method_commands.items():
                if series == "Simulation Fully (Tiled)" and args.simulation_times:
                    measurements[series][display_name] = reused_simulation_times[stem]
                    samples[series][display_name] = [reused_simulation_times[stem]]
                    commands[series][display_name] = command + [
                        "--database", str(args.simulation_db)
                    ]
                    with sqlite3.connect(args.simulation_db) as connection:
                        observed_cache_sizes[display_name] = [
                            row[0]
                            for row in connection.execute(
                                "SELECT d1_cache_size FROM records WHERE program = ? "
                                "ORDER BY d1_cache_size",
                                (f"tiled_{stem}.mlir",),
                            )
                        ]
                    continue
                run_command = list(command)
                if series == "Simulation Fully (Tiled)":
                    database = temporary_root / f"{stem}-{len(samples[series])}.db"
                    run_command.extend(["--database", str(database)])
                elapsed = measure(
                    run_command, args.repetitions, REPOSITORY_ROOT, f"{series}: {display_name}"
                )
                if series == "Simulation Fully (Tiled)":
                    with sqlite3.connect(database) as connection:
                        observed_cache_sizes[display_name] = [
                            row[0]
                            for row in connection.execute(
                                "SELECT DISTINCT d1_cache_size FROM records ORDER BY d1_cache_size"
                            )
                        ]
                samples[series][display_name] = elapsed
                measurements[series][display_name] = statistics.median(elapsed)
                commands[series][display_name] = (
                    command + ["--database", "<temporary-database>"]
                    if series == "Simulation Fully (Tiled)"
                    else command
                )

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
    )

    document = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "repository_commit": revision,
        "repository_dirty": dirty,
        "repetitions": args.repetitions,
        "aggregation": "median",
        "cache_limit_bytes": args.cache_limit_bytes,
        "cache_line_bytes": 64,
        "ll_cache": {"bytes": 65536, "associativity": 16, "line_bytes": 64},
        "observed_fully_associative_cache_sizes": observed_cache_sizes,
        "versions": {
            "valgrind": command_version(["valgrind", "--version"]),
            "clang": command_version(["clang++", "--version"]),
        },
        "measurements": measurements,
        "samples": samples,
        "commands": commands,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary_output.write_text(json.dumps(document, indent=2) + "\n")
    temporary_output.replace(args.out)
    print(f"Saved timing manifest to: {args.out}")


if __name__ == "__main__":
    main()
