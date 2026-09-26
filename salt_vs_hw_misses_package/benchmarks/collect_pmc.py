#!/usr/bin/env python3
"""Collect the Linux generic L1D read-miss event for 17 benchmarks."""
# python3 collect_pmc.py --cpu 1 --repeats 3

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "bin" / "benchmark_runner"
DEFAULT_OUTPUT = HERE.parent / "data" / "pmu_results_17_new.csv"
CPU_SYSFS_ROOT = Path("/sys/devices/system/cpu")
PMU_EVENT = "L1D.REPLACEMENT"
PMU_SELECTOR = "PERF_TYPE_HW_CACHE:L1D:READ:MISS"


def parse_cpu_list(value: str) -> set[int]:
    """Parse Linux CPU-list syntax such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if last < first:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(item))
    return cpus


def cpu_is_online(cpu: int, sysfs_root: Path = CPU_SYSFS_ROOT) -> bool:
    cpu_dir = sysfs_root / f"cpu{cpu}"
    if not cpu_dir.is_dir():
        raise RuntimeError(f"logical CPU {cpu} does not exist under {sysfs_root}")
    online_path = cpu_dir / "online"
    # Linux omits this file for CPUs that cannot be offlined, normally CPU 0.
    if not online_path.exists():
        return True
    state = online_path.read_text().strip()
    if state not in {"0", "1"}:
        raise RuntimeError(f"unexpected CPU online state {state!r} in {online_path}")
    return state == "1"


def sibling_cpus(cpu: int, sysfs_root: Path = CPU_SYSFS_ROOT) -> set[int]:
    if cpu < 0:
        raise ValueError("--cpu must be non-negative")
    if not cpu_is_online(cpu, sysfs_root):
        raise RuntimeError(f"selected logical CPU {cpu} is offline")

    siblings_path = sysfs_root / f"cpu{cpu}" / "topology" / "thread_siblings_list"
    try:
        siblings = parse_cpu_list(siblings_path.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(
            f"cannot read physical-core topology: missing {siblings_path}"
        ) from error
    if cpu not in siblings:
        raise RuntimeError(f"CPU {cpu} is absent from its sibling set {siblings}")
    return siblings


def set_cpu_online(
    cpu: int, online: bool, sysfs_root: Path = CPU_SYSFS_ROOT
) -> None:
    online_path = sysfs_root / f"cpu{cpu}" / "online"
    action = "online" if online else "offline"
    if not online_path.exists():
        raise RuntimeError(
            f"cannot set logical CPU {cpu} {action}: missing {online_path}"
        )

    state = "1\n" if online else "0\n"
    try:
        online_path.write_text(state)
    except PermissionError:
        print(
            f"requesting sudo permission to set logical CPU {cpu} {action}",
            file=sys.stderr,
        )
        try:
            run_checked(
                ["sudo", "tee", str(online_path)],
                input=state,
                stdout=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"cannot set logical CPU {cpu} {action} through {online_path}; "
                "grant sysfs CPU-hotplug permission or change it manually"
            ) from error
    except OSError as error:
        raise RuntimeError(
            f"cannot set logical CPU {cpu} {action} through {online_path}: {error}"
        ) from error

    if cpu_is_online(cpu, sysfs_root) != online:
        raise RuntimeError(f"logical CPU {cpu} did not become {action}")


@contextmanager
def isolate_core(
    cpu: int, sysfs_root: Path = CPU_SYSFS_ROOT
) -> Iterator[set[int]]:
    siblings = sibling_cpus(cpu, sysfs_root)
    online_siblings = sorted(
        sibling
        for sibling in siblings
        if sibling != cpu and cpu_is_online(sibling, sysfs_root)
    )
    changed: list[int] = []
    try:
        for sibling in online_siblings:
            print(f"temporarily offlining sibling CPU {sibling}", file=sys.stderr)
            set_cpu_online(sibling, False, sysfs_root)
            changed.append(sibling)
        yield siblings
    finally:
        restoration_errors = []
        for sibling in reversed(changed):
            try:
                print(f"restoring sibling CPU {sibling}", file=sys.stderr)
                set_cpu_online(sibling, True, sysfs_root)
            except RuntimeError as error:
                restoration_errors.append(str(error))
        if restoration_errors:
            raise RuntimeError("; ".join(restoration_errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--cpu",
        type=int,
        help=(
            "pin each run to this logical CPU; sibling hyperthreads are "
            "temporarily offlined and restored"
        ),
    )
    return parser.parse_args()


def run_checked(
    command: list[str], **kwargs: object
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    if args.cpu is None:
        print(
            "warning: no --cpu was supplied; this exploratory collection is not "
            "suitable for comparison with the isolated reference data",
            file=sys.stderr,
        )
        return collect(args)

    with isolate_core(args.cpu) as siblings:
        print(
            f"using logical CPU {args.cpu}; sibling set "
            f"{','.join(map(str, sorted(siblings)))} is isolated",
            file=sys.stderr,
        )
        return collect(args)


def collect(args: argparse.Namespace) -> int:
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
                "pmu_event": PMU_EVENT,
                "pmu_selector": PMU_SELECTOR,
                "pmc_count": round(statistics.median(values)),
                "repeats": args.repeats,
                "all_pmc_values": ";".join(map(str, values)),
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
    try:
        exit_code = main()
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    raise SystemExit(exit_code)
