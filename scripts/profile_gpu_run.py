#!/usr/bin/env python3
"""Sample NVIDIA GPU usage while a pipeline run is executed manually.

The script is intentionally independent from the application. Start it in one
terminal, run the interface/pipeline normally in another one, and stop the
sampler with Ctrl+C when the run finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


GPU_QUERY = [
    "timestamp",
    "index",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
]

PROCESS_QUERY = [
    "pid",
    "process_name",
    "used_memory",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    clean = []
    for char in value.strip().lower():
        if char.isalnum():
            clean.append(char)
        elif char in {"-", "_"}:
            clean.append(char)
        elif char.isspace():
            clean.append("-")
    return "".join(clean).strip("-") or "run"


def _run_command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def _parse_number(value: str) -> float | None:
    raw = value.strip()
    if not raw or raw.upper() == "N/A":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _query_gpu_rows() -> list[dict[str, Any]]:
    output = _run_command(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY)}",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    for parsed in csv.reader(output.splitlines()):
        if len(parsed) != len(GPU_QUERY):
            continue
        row = {key: parsed[idx].strip() for idx, key in enumerate(GPU_QUERY)}
        rows.append(
            {
                "nvidia_timestamp": row["timestamp"],
                "gpu_index": int(_parse_number(row["index"]) or 0),
                "gpu_name": row["name"],
                "memory_total_mib": _parse_number(row["memory.total"]),
                "memory_used_mib": _parse_number(row["memory.used"]),
                "utilization_gpu_percent": _parse_number(row["utilization.gpu"]),
                "utilization_memory_percent": _parse_number(row["utilization.memory"]),
                "power_draw_w": _parse_number(row["power.draw"]),
            }
        )
    return rows


def _query_process_rows() -> list[dict[str, Any]]:
    try:
        output = _run_command(
            [
                "nvidia-smi",
                f"--query-compute-apps={','.join(PROCESS_QUERY)}",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        return []
    rows: list[dict[str, Any]] = []
    for parsed in csv.reader(output.splitlines()):
        if len(parsed) != len(PROCESS_QUERY):
            continue
        pid_value = _parse_number(parsed[0])
        if pid_value is None:
            continue
        rows.append(
            {
                "pid": int(pid_value),
                "process_name": parsed[1].strip(),
                "used_memory_mib": _parse_number(parsed[2]),
            }
        )
    return rows


def _process_command(pid: int, cache: dict[int, str]) -> str:
    if pid in cache:
        return cache[pid]
    try:
        cmd = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "args="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        cmd = ""
    cache[pid] = cmd
    return cmd


def _write_header(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def _append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def _avg(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return round(mean(filtered), 3) if filtered else None


def _max(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return round(max(filtered), 3) if filtered else None


def _min(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return round(min(filtered), 3) if filtered else None


def _summarize(
    gpu_samples: list[dict[str, Any]],
    process_samples: list[dict[str, Any]],
    started_at_utc: str,
    stopped_at_utc: str,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    elapsed_values = [float(sample["elapsed_s"]) for sample in gpu_samples]
    duration_seconds = round(max(elapsed_values), 3) if elapsed_values else 0.0
    total_memory = None
    gpu_name = None
    if gpu_samples:
        gpu_name = gpu_samples[0].get("gpu_name")
        total_memory = gpu_samples[0].get("memory_total_mib")

    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in process_samples:
        grouped[
            (
                int(sample["pid"]),
                str(sample.get("process_name") or ""),
                str(sample.get("command") or ""),
            )
        ].append(sample)

    process_summary = []
    for (pid, process_name, command), samples in grouped.items():
        mem_values = [sample.get("used_memory_mib") for sample in samples]
        elapsed = [float(sample["elapsed_s"]) for sample in samples]
        process_summary.append(
            {
                "pid": pid,
                "process_name": process_name,
                "command": command,
                "samples": len(samples),
                "first_seen_s": round(min(elapsed), 3),
                "last_seen_s": round(max(elapsed), 3),
                "peak_memory_mib": _max(mem_values),
                "avg_memory_mib": _avg(mem_values),
            }
        )
    process_summary.sort(key=lambda item: (-(item["peak_memory_mib"] or 0), item["pid"]))

    summary = {
        "label": label,
        "started_at_utc": started_at_utc,
        "stopped_at_utc": stopped_at_utc,
        "duration_seconds": duration_seconds,
        "output_dir": str(output_dir),
        "gpu": {
            "name": gpu_name,
            "memory_total_mib": total_memory,
            "sample_count": len(gpu_samples),
            "memory_used_min_mib": _min([sample.get("memory_used_mib") for sample in gpu_samples]),
            "memory_used_avg_mib": _avg([sample.get("memory_used_mib") for sample in gpu_samples]),
            "memory_used_max_mib": _max([sample.get("memory_used_mib") for sample in gpu_samples]),
            "utilization_gpu_avg_percent": _avg(
                [sample.get("utilization_gpu_percent") for sample in gpu_samples]
            ),
            "utilization_gpu_max_percent": _max(
                [sample.get("utilization_gpu_percent") for sample in gpu_samples]
            ),
            "utilization_memory_avg_percent": _avg(
                [sample.get("utilization_memory_percent") for sample in gpu_samples]
            ),
            "utilization_memory_max_percent": _max(
                [sample.get("utilization_memory_percent") for sample in gpu_samples]
            ),
            "power_draw_avg_w": _avg([sample.get("power_draw_w") for sample in gpu_samples]),
            "power_draw_max_w": _max([sample.get("power_draw_w") for sample in gpu_samples]),
        },
        "processes": process_summary,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile NVIDIA GPU usage while the football narrator pipeline runs."
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Optional duration limit in seconds.")
    parser.add_argument("--label", default="pipeline", help="Short label used in the output directory name.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/gpu_profile"),
        help="Directory where CSV and summary files will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_root / f"{timestamp}_{_safe_slug(args.label)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_csv = output_dir / "gpu_samples.csv"
    process_csv = output_dir / "process_samples.csv"
    summary_json = output_dir / "summary.json"

    gpu_fields = [
        "sample_index",
        "elapsed_s",
        "timestamp_utc",
        "nvidia_timestamp",
        "gpu_index",
        "gpu_name",
        "memory_total_mib",
        "memory_used_mib",
        "utilization_gpu_percent",
        "utilization_memory_percent",
        "power_draw_w",
    ]
    process_fields = [
        "sample_index",
        "elapsed_s",
        "timestamp_utc",
        "pid",
        "process_name",
        "command",
        "used_memory_mib",
    ]
    _write_header(gpu_csv, gpu_fields)
    _write_header(process_csv, process_fields)

    started_at_utc = _utc_now_iso()
    start = time.monotonic()
    sample_index = 0
    stop_requested = False
    command_cache: dict[int, str] = {}
    gpu_samples: list[dict[str, Any]] = []
    process_samples: list[dict[str, Any]] = []

    def _request_stop(signum: int, frame: object) -> None:  # noqa: ARG001
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    print(f"Writing GPU profile to {output_dir}")
    print("Run the interface/pipeline now. Press Ctrl+C here when it finishes.")

    try:
        while not stop_requested:
            now_utc = _utc_now_iso()
            elapsed_s = round(time.monotonic() - start, 3)

            gpu_rows = []
            for row in _query_gpu_rows():
                sample = {
                    "sample_index": sample_index,
                    "elapsed_s": elapsed_s,
                    "timestamp_utc": now_utc,
                    **row,
                }
                gpu_rows.append(sample)
                gpu_samples.append(sample)

            process_rows = []
            for row in _query_process_rows():
                command = _process_command(int(row["pid"]), command_cache)
                sample = {
                    "sample_index": sample_index,
                    "elapsed_s": elapsed_s,
                    "timestamp_utc": now_utc,
                    **row,
                    "command": command,
                }
                process_rows.append(sample)
                process_samples.append(sample)

            _append_rows(gpu_csv, gpu_fields, gpu_rows)
            _append_rows(process_csv, process_fields, process_rows)

            sample_index += 1
            if args.duration is not None and elapsed_s >= args.duration:
                break
            time.sleep(args.interval)
    finally:
        stopped_at_utc = _utc_now_iso()
        summary = _summarize(
            gpu_samples,
            process_samples,
            started_at_utc,
            stopped_at_utc,
            output_dir,
            args.label,
        )
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        gpu = summary["gpu"]
        print("\nSummary")
        print(f"- duration: {summary['duration_seconds']} s")
        print(f"- GPU: {gpu['name']}")
        print(
            "- memory used: "
            f"avg {gpu['memory_used_avg_mib']} MiB, "
            f"max {gpu['memory_used_max_mib']} MiB / {gpu['memory_total_mib']} MiB"
        )
        print(
            "- GPU utilization: "
            f"avg {gpu['utilization_gpu_avg_percent']} %, "
            f"max {gpu['utilization_gpu_max_percent']} %"
        )
        print(f"- summary: {summary_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
