import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "analysis" / "tracking_model_benchmark"

VIDEO_SHORTCUTS = [
    "video_prueba",
    "clasico_30s",
    "ferro_30s",
    "ucl_30s",
]

MODEL_PATHS = [
    "models/yolo/v11/yolo11m.pt",
    "models/finetuning/yolov11m/weights/best.pt",
    "models/finetuning/con_arbitro/dfl-bundesliga/weights/best.pt",
]


def sanitize_token(raw_value: str) -> str:
    token = str(raw_value).strip().lower()
    token = token.replace("/", "_").replace("\\", "_").replace(" ", "_")
    token = token.replace(".", "_").replace("-", "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def build_run_label(video_shortcut: str, model_path: str) -> str:
    return f"{sanitize_token(video_shortcut)}__{sanitize_token(Path(model_path).stem)}"


def load_config() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))


def resolve_video_path(config: dict, video_shortcut: str) -> str:
    relative_path = config["paths"]["data"][video_shortcut]
    return str((PROJECT_ROOT / relative_path).resolve())


def resolve_model_path(model_path: str) -> str:
    return str((PROJECT_ROOT / model_path).resolve())


def collect_git_metadata() -> dict:
    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": run_git("status", "--short"),
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def run_single_experiment(
    python_executable: str,
    video_shortcut: str,
    model_path: str,
    run_dir: Path,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        "scripts/track.py",
        video_shortcut,
        "--model-path",
        model_path,
        "--output-root",
        str(run_dir),
        "--experiment-label",
        build_run_label(video_shortcut, model_path),
        "--force-four-panel-debug",
        "--skip-render-video",
        "--skip-metrics-dataset",
    ]

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    runtime_seconds = time.perf_counter() - t0
    ended_at = datetime.now(timezone.utc).isoformat()

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    summary_path = run_dir / "summary.json"
    tracks_path = run_dir / "tracks.json"
    debug_frames_path = run_dir / "debug_frames.json"

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    manifest = {
        "video_shortcut": video_shortcut,
        "model_path": model_path,
        "command": command,
        "returncode": completed.returncode,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "runtime_seconds": runtime_seconds,
        "artifacts": {
            "summary_json": str(summary_path),
            "tracks_json": str(tracks_path),
            "debug_frames_json": str(debug_frames_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)

    result_row = {
        "run_label": build_run_label(video_shortcut, model_path),
        "video_shortcut": video_shortcut,
        "model_path": model_path,
        "runtime_seconds": round(runtime_seconds, 4),
        "returncode": completed.returncode,
        "summary_json": str(summary_path),
        "tracks_json": str(tracks_path),
        "debug_frames_json": str(debug_frames_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    for key, value in summary.items():
        result_row[str(key)] = value
    return result_row


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta un benchmark 4x3 sobre scripts/track.py y guarda artefactos "
            "aislados por corrida."
        )
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Directorio raíz donde guardar el benchmark completo.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Intérprete Python con el que lanzar scripts/track.py.",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        default=None,
        help="Subset opcional de shortcuts de vídeo a ejecutar.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset opcional de rutas de modelo a ejecutar.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_videos = args.videos if args.videos else VIDEO_SHORTCUTS
    selected_models = args.models if args.models else MODEL_PATHS

    benchmark_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_bin": args.python_bin,
        "project_root": str(PROJECT_ROOT),
        "videos": [
            {
                "shortcut": shortcut,
                "resolved_path": resolve_video_path(config, shortcut),
            }
            for shortcut in selected_videos
        ],
        "models": [
            {
                "path": model_path,
                "resolved_path": resolve_model_path(model_path),
            }
            for model_path in selected_models
        ],
        "git": collect_git_metadata(),
    }
    write_json(output_root / "benchmark_manifest.json", benchmark_manifest)
    (output_root / "config_snapshot.yaml").write_text(
        (PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    results: list[dict] = []
    total_runs = len(selected_videos) * len(selected_models)
    run_index = 0

    for video_shortcut in selected_videos:
        for model_path in selected_models:
            run_index += 1
            run_label = build_run_label(video_shortcut, model_path)
            run_dir = output_root / "runs" / run_label
            print(
                f"[{run_index}/{total_runs}] Ejecutando {video_shortcut} con {model_path}",
                flush=True,
            )
            result_row = run_single_experiment(
                python_executable=args.python_bin,
                video_shortcut=video_shortcut,
                model_path=model_path,
                run_dir=run_dir,
            )
            results.append(result_row)
            write_csv(output_root / "comparison_summary.csv", results)
            write_json(output_root / "comparison_summary.json", results)

    failures = [row for row in results if int(row.get("returncode", 1)) != 0]
    benchmark_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    benchmark_manifest["total_runs"] = total_runs
    benchmark_manifest["failed_runs"] = len(failures)
    benchmark_manifest["comparison_summary_csv"] = str(output_root / "comparison_summary.csv")
    benchmark_manifest["comparison_summary_json"] = str(output_root / "comparison_summary.json")
    write_json(output_root / "benchmark_manifest.json", benchmark_manifest)

    if failures:
        print(f"Benchmark completado con {len(failures)} fallos.", flush=True)
        return 1

    print(f"Benchmark completado correctamente. Artefactos en {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
