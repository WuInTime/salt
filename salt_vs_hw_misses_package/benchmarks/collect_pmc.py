#!/usr/bin/env python3
"""Collect L1D read misses for the 17 shipped benchmarks."""
# python3 collect_pmc.py --cpu 1 --repeats 3

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "bin" / "benchmark_runner"
DEFAULT_OUTPUT = HERE.parent / "data" / "pmu_results_17_new.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--cpu",
        type=int,
        help="pin each run to this logical CPU with taskset (recommended)",
    )
    return parser.parse_args()


def run_checked(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    run_checked(["make"], cwd=HERE)
    names_result = run_checked([str(RUNNER), "--list"], capture_output=True)
    names = names_result.stdout.splitlines()
    if len(names) != 17:
        raise RuntimeError(f"runner exposed {len(names)} benchmarks instead of 17")

    rows: list[dict[str, object]] = []
    for index, name in enumerate(names, start=1):
        values = []
        print(f"[{index:02d}/17] {name}", file=sys.stderr, flush=True)
        for repeat in range(1, args.repeats + 1):
            command = [str(RUNNER), name]
            if args.cpu is not None:
                command = ["taskset", "-c", str(args.cpu), *command]
            result = run_checked(command, capture_output=True)
            try:
                value = int(result.stdout.strip())
            except ValueError as error:
                raise RuntimeError(
                    f"unexpected counter output for {name}: {result.stdout!r}"
                ) from error
            values.append(value)
            print(f"  repeat {repeat}: {value}", file=sys.stderr, flush=True)
        rows.append(
            {
                "program": name,
                "csv_l1d_load_miss": round(statistics.median(values)),
                "repeats": args.repeats,
                "all_l1d_load_miss_values": ";".join(map(str, values)),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
