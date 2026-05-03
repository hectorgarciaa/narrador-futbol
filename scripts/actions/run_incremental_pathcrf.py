from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.actions_incremental import (  # noqa: E402
    ActionsDetectorConfig,
    ActionsRuntime,
    ActionsRuntimeConfig,
)
from football_ai.core.config import Config  # noqa: E402
from football_ai.pipeline.paths import resolve_video_path  # noqa: E402
from football_ai.positions.data import resolve_tracks_path_for_video  # noqa: E402

TRACK_CLASSES = ("player", "goalkeeper", "referee", "ball")


def _resolve_tracks(source: str | None) -> Path:
    if source is None:
        config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
        video_path, _ = resolve_video_path(config, None)
        return resolve_tracks_path_for_video(PROJECT_ROOT, Path(video_path)).resolve()
    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    if candidate.exists() and candidate.suffix.lower() == ".json":
        return candidate
    if candidate.exists():
        return resolve_tracks_path_for_video(PROJECT_ROOT, candidate).resolve()
    config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
    video_path, _ = resolve_video_path(config, source)
    return resolve_tracks_path_for_video(PROJECT_ROOT, Path(video_path)).resolve()


def _load_tracks(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_frame_count(tracks: dict[str, Any]) -> int:
    return max((len(tracks.get(class_name, [])) for class_name in TRACK_CLASSES), default=0)


def _clean_packet(tracks: dict[str, Any], frame_index: int) -> dict[str, Any]:
    return {
        "tracks_frame": {
            class_name: (
                tracks.get(class_name, [])[frame_index]
                if frame_index < len(tracks.get(class_name, []))
                else {}
            )
            for class_name in TRACK_CLASSES
        },
        "possession": (
            tracks.get("possession", [])[frame_index]
            if frame_index < len(tracks.get("possession", []))
            else {}
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline incremental aislado: actualiza estado frame a frame y "
            "ejecuta inferencia PathCRF cada N frames tras warmup."
        )
    )
    parser.add_argument("source", nargs="?", default=None, help="Video key/ruta o tracks.json.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Salida. Default: output/actions/pathcrf_incremental/<video>/")
    parser.add_argument("--repo-path", type=Path, default=Path("football_ai/actions/repo/pathcrf"))
    parser.add_argument("--trial", type=int, default=120)
    parser.add_argument("--model-file", type=str, default="state_dict_best_acc.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--cadence-frames", type=int, default=10)
    parser.add_argument("--min-frames-warmup", type=int, default=50)
    parser.add_argument("--confirmation-cooldown-frames", type=int, default=15)
    parser.add_argument("--window-size-frames", type=int, default=None)
    parser.add_argument("--decode", choices=("indep", "greedy", "viterbi"), default="indep")
    parser.add_argument("--no-crf", action="store_true")
    parser.add_argument("--correct-episode-lasts", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--sample-freq", type=int, default=None)
    parser.add_argument("--min-event-duration", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tracks_path = _resolve_tracks(args.source)
    if not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON esperado: {tracks_path}")

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / "output" / "actions" / "pathcrf_incremental" / tracks_path.stem.replace("_tracks", "")).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = _load_tracks(tracks_path)
    frame_count = _infer_frame_count(tracks)
    if args.max_frames is not None:
        frame_count = min(frame_count, int(args.max_frames))

    runtime = ActionsRuntime(
        ActionsRuntimeConfig(
            detector=ActionsDetectorConfig(
                fps=float(args.fps),
                window_size_frames=int(args.window_size_frames) if args.window_size_frames is not None else None,
            ),
            repo_path=args.repo_path,
            trial=int(args.trial),
            model_file=str(args.model_file),
            use_crf=not bool(args.no_crf),
            decode=str(args.decode),
            correct_episode_lasts=bool(args.correct_episode_lasts),
            evaluate=bool(args.evaluate),
            cadence_frames=int(args.cadence_frames),
            min_frames_warmup=int(args.min_frames_warmup),
            min_event_duration=int(args.min_event_duration),
            window_seconds=float(args.window_seconds) if args.window_seconds is not None else None,
            fps=float(args.fps),
            sample_freq=int(args.sample_freq) if args.sample_freq is not None else None,
            device=str(args.device),
            confirmation_cooldown_frames=int(args.confirmation_cooldown_frames),
        )
    )

    checkpoints: list[dict[str, Any]] = []
    internal_edges_by_frame: dict[int, dict[str, Any]] = {}
    for frame_index in range(frame_count):
        result = runtime.process_frame(frame_index, _clean_packet(tracks, frame_index))
        metadata = dict(result.action_metadata)
        if metadata.get("should_infer"):
            for edge_payload in result.raw_edge_batch:
                edge_frame = edge_payload.get("frame_id")
                if edge_frame is None:
                    continue
                try:
                    edge_frame_int = int(edge_frame)
                except (TypeError, ValueError):
                    continue
                internal_edges_by_frame[edge_frame_int] = dict(edge_payload)
            checkpoints.append(
                {
                    "frame_id": int(frame_index),
                    "raw_edge": result.raw_edge,
                    "raw_edge_batch": result.raw_edge_batch,
                    "confirmed_action": result.confirmed_action,
                    "metadata": metadata,
                    "summary": asdict(result.summary),
                }
            )

    tracking_df, summary = runtime.detector.build_tracking_dataframe()
    edge_sequence_df, semantic_events_df = runtime._run_pathcrf(tracking_df)

    tracking_path = out_dir / "tracking.parquet"
    edge_path = out_dir / "edge_sequence.parquet"
    events_path = out_dir / "events_semantic.parquet"
    summary_path = out_dir / "summary.json"
    checkpoints_path = out_dir / "runtime_checkpoints.json"
    internal_edges_path = out_dir / "internal_edges_per_frame.parquet"
    tracking_df.to_parquet(tracking_path, index=False)
    edge_sequence_df.to_parquet(edge_path, index=False)
    semantic_events_df.to_parquet(events_path, index=False)
    internal_edges_df = (
        pd.DataFrame(
            [value for _, value in sorted(internal_edges_by_frame.items(), key=lambda item: item[0])]
        )
        if internal_edges_by_frame
        else pd.DataFrame(columns=["frame_id", "edge_src", "edge_dst", "edge_src_track_id", "edge_dst_track_id"])
    )
    internal_edges_df.to_parquet(internal_edges_path, index=False)

    with checkpoints_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoints, f, ensure_ascii=False, indent=2)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            _json_safe(
                {
                    "tracks_path": str(tracks_path),
                    "frame_count": int(frame_count),
                    "tracking_path": str(tracking_path),
                    "edge_sequence_path": str(edge_path),
                    "events_semantic_path": str(events_path),
                    "runtime_checkpoints_path": str(checkpoints_path),
                    "internal_edges_per_frame_path": str(internal_edges_path),
                    "runtime_checkpoints": int(len(checkpoints)),
                    "internal_edges_rows": int(len(internal_edges_df)),
                    "detector_summary": asdict(summary),
                    "runtime_config": asdict(runtime.config),
                }
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Output dir: {out_dir}")
    print(f"Frames analizados: {frame_count}")
    print(f"Runtime checkpoints: {len(checkpoints)}")
    print(f"Tracking: {tracking_path}")
    print(f"Edge sequence: {edge_path}")
    print(f"Semantic events: {events_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
