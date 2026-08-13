#!/usr/bin/env python3
"""Generate SALT miss counts, compare them with PMC results, and plot the figure.

The SALT JSON parsing and piecewise cache-size lookup are extracted from
tools/compare_pmu_cachegrind.py. Cachegrind database handling is intentionally
excluded because the SALT-vs-hardware graph does not use it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
DEFAULT_PMC = HERE / "data" / "pmu_i7-7700_result.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "salt-vs-hardware"
DEFAULT_RESULTS = DEFAULT_OUTPUT_DIR / "salt_vs_hw_misses_results.csv"
DEFAULT_PLOT = DEFAULT_OUTPUT_DIR / "salt_vs_hw_misses.svg"
DEFAULT_SALT_JSON_DIR = DEFAULT_OUTPUT_DIR / "salt-json"
DEFAULT_ANALYZER = REPOSITORY_ROOT / "target" / "release" / "analyzer"
CONTRACTION_DIR = REPOSITORY_ROOT / "benchmarks" / "mlir-contractions" / "constant"
STENCIL_INPUT = REPOSITORY_ROOT / "benchmarks" / "examples" / "const_stencil5pt.mlir"

BENCHMARKS = (
    "orig_3d_tensor_vector",
    "orig_4d_tensor",
    "orig_attention_score",
    "orig_batched_gemm",
    "orig_context_lookup",
    "orig_matrix_matrix",
    "orig_matrix_vector",
    "orig_rowwise_softmax_max",
    "tiled_3d_tensor_vector",
    "tiled_4d_tensor",
    "tiled_attention_score",
    "tiled_batched_gemm",
    "tiled_context_lookup",
    "tiled_matrix_matrix",
    "tiled_matrix_vector",
    "tiled_rowwise_softmax_max",
    "orig_stencil5pt",
)
SMOKE_BENCHMARKS = (
    "orig_3d_tensor_vector",
    "tiled_3d_tensor_vector",
    "orig_stencil5pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--salt-json-dir",
        type=Path,
        default=DEFAULT_SALT_JSON_DIR,
        help=(
            "directory in which to generate *-salt.json files "
            f"(default: {DEFAULT_SALT_JSON_DIR})"
        ),
    )
    parser.add_argument(
        "--analyzer",
        type=Path,
        default=DEFAULT_ANALYZER,
        help=f"prebuilt SALT analyzer (default: {DEFAULT_ANALYZER})",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run a representative original, tiled, and stencil subset",
    )
    parser.add_argument(
        "--pmc",
        type=Path,
        default=DEFAULT_PMC,
        help=f"PMC CSV (default: {DEFAULT_PMC.relative_to(HERE)})",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=DEFAULT_RESULTS,
        help="derived SALT/PMC comparison CSV",
    )
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--cache-size-bytes", type=int, default=32768)
    parser.add_argument("--cache-line-bytes", type=int, default=64)
    parser.add_argument(
        "--elements-per-cache-line",
        type=int,
        default=8,
        help="target elements per modeled cache line (default: 8)",
    )
    return parser.parse_args()


def normalize_program(name: str) -> str:
    name = name.removesuffix(".mlir")
    if name.startswith("constant_"):
        return "orig_" + name.removeprefix("constant_")
    return name


def json_name_for(program: str) -> str:
    if program.startswith("orig_"):
        return "constant_" + program.removeprefix("orig_") + "-salt.json"
    return program + "-salt.json"


def mlir_input_for(program: str) -> Path:
    if program == "orig_stencil5pt":
        return STENCIL_INPUT
    if program.startswith("orig_"):
        name = "constant_" + program.removeprefix("orig_") + ".mlir"
        return CONTRACTION_DIR / name
    if program.startswith("tiled_"):
        return CONTRACTION_DIR / "tiled" / f"{program}.mlir"
    raise ValueError(f"unknown benchmark name: {program}")


def generate_salt_jsons(
    root: Path,
    analyzer: Path,
    benchmarks: tuple[str, ...],
    elements_per_cache_line: int,
) -> None:
    if not analyzer.is_file():
        raise FileNotFoundError(
            f"analyzer does not exist: {analyzer}; build it with "
            "cargo build --locked --release -p analyzer --bin analyzer"
        )

    missing_inputs = [mlir_input_for(program) for program in benchmarks]
    missing_inputs = [path for path in missing_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "missing MLIR inputs: " + ", ".join(map(str, missing_inputs))
        )

    for program in benchmarks:
        input_path = mlir_input_for(program)
        output_dir = root / "tiled" if program.startswith("tiled_") else root
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / json_name_for(program)

        print(f"Generating SALT prediction for {program}...")
        subprocess.run(
            [
                str(analyzer),
                "-i",
                str(input_path),
                "--json",
                "-o",
                str(output_path),
                "salt",
                f"--block-size={elements_per_cache_line}",
            ],
            check=True,
        )


def index_json_files(root: Path, benchmarks: tuple[str, ...]) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"SALT JSON directory does not exist: {root}")

    expected = {json_name_for(program) for program in benchmarks}
    matches: dict[str, list[Path]] = {}
    for path in root.rglob("*.json"):
        if path.name in expected:
            matches.setdefault(path.name, []).append(path)

    duplicates = {name: paths for name, paths in matches.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(
            f"{name}: {', '.join(map(str, paths))}"
            for name, paths in sorted(duplicates.items())
        )
        raise ValueError(f"duplicate SALT JSON filenames: {details}")
    return {name: paths[0] for name, paths in matches.items()}


def numeric_list(value: Any, label: str, path: Path) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: {label} must be a JSON array")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {label} contains a non-numeric value") from error
    if not result or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{path}: {label} must contain finite values")
    return result


def load_salt_curve(path: Path) -> tuple[list[float], list[float], int]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")

    nested = data.get("miss_ratio_curve")
    curve = nested if isinstance(nested, dict) else data
    ratios_value = curve.get("miss_ratio")
    if ratios_value is None:
        ratios_value = curve.get("miss_ratios")
    ratios = numeric_list(ratios_value, "miss_ratio", path)
    points = numeric_list(curve.get("turning_points"), "turning_points", path)
    if len(ratios) != len(points):
        raise ValueError(
            f"{path}: miss_ratio and turning_points lengths differ "
            f"({len(ratios)} != {len(points)})"
        )
    if any(right < left for left, right in zip(points, points[1:])):
        raise ValueError(f"{path}: turning_points must be nondecreasing")

    total_value = data.get("total_count")
    if total_value is None:
        total_value = curve.get("total_count")
    try:
        total_count = int(float(total_value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: missing or invalid total_count") from error
    if total_count <= 0:
        raise ValueError(f"{path}: total_count must be positive")
    return ratios, points, total_count


def miss_ratio_at(
    ratios: list[float], turning_points: list[float], cache_lines: float
) -> float:
    if cache_lines < turning_points[0]:
        raise ValueError(
            f"cache size {cache_lines} lines precedes first turning point "
            f"{turning_points[0]}"
        )
    for index in range(len(turning_points) - 1):
        if turning_points[index] <= cache_lines < turning_points[index + 1]:
            return ratios[index]
    return ratios[-1]


def load_pmc(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        program_column = "program" if "program" in fields else "kernel"
        if program_column not in fields:
            raise ValueError(f"{path}: expected a program or kernel column")
        if "csv_l1d_load_miss" in fields:
            miss_column = "csv_l1d_load_miss"
        elif "L1D.load_miss" in fields:
            miss_column = "L1D.load_miss"
        else:
            raise ValueError(
                f"{path}: expected csv_l1d_load_miss or L1D.load_miss column"
            )

        values: dict[str, float] = {}
        for row in reader:
            program = normalize_program(row[program_column].strip())
            if program in values:
                raise ValueError(f"{path}: duplicate PMC row for {program}")
            try:
                value = float(row[miss_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid PMC value for {program}") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{path}: invalid PMC value for {program}: {value}")
            values[program] = value
    return values


def derive_results(
    args: argparse.Namespace, benchmarks: tuple[str, ...]
) -> list[dict[str, Any]]:
    if args.cache_size_bytes <= 0 or args.cache_line_bytes <= 0:
        raise ValueError("cache and line sizes must be positive")
    cache_lines = args.cache_size_bytes / args.cache_line_bytes
    pmc = load_pmc(args.pmc)
    json_files = index_json_files(args.salt_json_dir, benchmarks)

    missing_pmc = [program for program in benchmarks if program not in pmc]
    missing_json = [
        json_name_for(program)
        for program in benchmarks
        if json_name_for(program) not in json_files
    ]
    if missing_pmc:
        raise ValueError(f"missing PMC results for: {', '.join(missing_pmc)}")
    if missing_json:
        raise FileNotFoundError(
            f"missing SALT JSON files beneath {args.salt_json_dir}: "
            + ", ".join(missing_json)
        )

    rows: list[dict[str, Any]] = []
    for program in benchmarks:
        json_path = json_files[json_name_for(program)]
        ratios, points, total_count = load_salt_curve(json_path)
        ratio = miss_ratio_at(ratios, points, cache_lines)
        measured = pmc[program]
        estimated = ratio * total_count
        relative_error = (
            (estimated - measured) / measured if measured != 0 else math.nan
        )
        rows.append(
            {
                "program": program,
                "csv_l1d_load_miss": measured,
                "salt_estimated_miss_count": round(estimated, 1),
                "relative_error": round(relative_error, 4),
                "salt_miss_ratio": round(ratio, 4),
                "total_access": total_count,
                "cache_size_bytes": args.cache_size_bytes,
                "cache_line_bytes": args.cache_line_bytes,
                "salt_json": str(json_path),
            }
        )
    return rows


def write_results(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, Any]], path: Path) -> tuple[float, float]:
    measured = np.asarray([row["csv_l1d_load_miss"] for row in rows], dtype=float)
    estimated = np.asarray(
        [row["salt_estimated_miss_count"] for row in rows], dtype=float
    )
    relative = (estimated - measured) / np.where(measured == 0, np.nan, measured)
    mare = float(np.nanmean(np.abs(relative)))
    pearson_r, _ = stats.pearsonr(measured, estimated)

    plt.style.use("seaborn-v0_8")
    sns.set_context("talk")
    plt.rcParams["svg.hashsalt"] = "salt-vs-hw-misses"
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.scatter(measured, estimated, s=100, alpha=0.8)
    maximum = float(np.nanmax(np.concatenate([measured, estimated])))
    ax.plot([0, maximum], [0, maximum], "k--", alpha=0.6)
    ax.set_xlabel("PMC L1D load misses")
    ax.set_ylabel("Cache misses")
    annotation = f" N={len(rows)}\nMARE={mare:.4f}\nPearson r={pearson_r:.3f}"
    ax.text(
        0.05,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"fc": "white", "alpha": 0.8},
    )
    if maximum > 1e4:
        ax.set_xscale("log")
        ax.set_yscale("log")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=1200, metadata={"Date": None})
    plt.close(fig)
    return mare, float(pearson_r)


def main() -> int:
    args = parse_args()
    if args.elements_per_cache_line <= 0:
        raise ValueError("--elements-per-cache-line must be positive")
    args.salt_json_dir = args.salt_json_dir.resolve()
    args.analyzer = args.analyzer.resolve()
    benchmarks = SMOKE_BENCHMARKS if args.smoke_test else BENCHMARKS
    generate_salt_jsons(
        args.salt_json_dir,
        args.analyzer,
        benchmarks,
        args.elements_per_cache_line,
    )
    rows = derive_results(args, benchmarks)
    write_results(rows, args.results_output)
    mare, pearson_r = plot_results(rows, args.plot_output)
    print(
        f"benchmarks={len(rows)} MARE={mare:.4f} Pearson_r={pearson_r:.4f}\n"
        f"wrote {args.results_output}\n"
        f"wrote {args.plot_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
