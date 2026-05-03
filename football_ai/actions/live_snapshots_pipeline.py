from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .pathcrf_adapter import PathCRFAdapterConfig
from .pathcrf_wrapper import (
    PathCRFInferenceConfig,
    PathCRFPipelineResult,
    PathCRFRenderConfig,
    run_pathcrf_pipeline,
)

TRACK_CLASSES = ("player", "goalkeeper", "referee", "ball")


@dataclass(frozen=True)
class LiveSnapshotsConfig:
    snapshot_interval_frames: int = 75
    min_frames: int = 50
    fps: float = 25.0
    adapter_config: PathCRFAdapterConfig | None = None
    inference_config: PathCRFInferenceConfig | None = None


@dataclass(frozen=True)
class LiveSnapshotsRun:
    frame_id: int
    observed_seconds: float
    snapshot_tracks_path: Path
    output_dir: Path
    result: PathCRFPipelineResult


@dataclass(frozen=True)
class LiveSnapshotsPipelineResult:
    tracks_path: Path
    output_dir: Path
    frame_count: int
    snapshot_runs: list[LiveSnapshotsRun]
    sparse_edge_sequence_path: Path
    summary_path: Path


def _infer_frame_count(tracks: dict[str, Any]) -> int:
    return max((len(tracks.get(class_name, [])) for class_name in TRACK_CLASSES), default=0)


def _load_tracks(tracks_path: Path) -> dict[str, Any]:
    with tracks_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _snapshot_tracks(tracks: dict[str, Any], frame_id: int) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in tracks.items():
        if isinstance(value, list):
            snapshot[key] = value[: frame_id + 1]
        else:
            snapshot[key] = value
    return snapshot


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


def run_live_snapshots_pipeline(
    *,
    tracks_path: str | Path,
    output_dir: str | Path,
    config: LiveSnapshotsConfig | None = None,
) -> LiveSnapshotsPipelineResult:
    cfg = config or LiveSnapshotsConfig()
    tracks_path = Path(tracks_path).expanduser().resolve()
    if not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON esperado: {tracks_path}")

    output_dir = Path(output_dir).expanduser().resolve()
    snapshots_root = output_dir / "snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)

    adapter_config = cfg.adapter_config or PathCRFAdapterConfig(fps=float(cfg.fps))
    inference_config = cfg.inference_config or PathCRFInferenceConfig()
    render_config = PathCRFRenderConfig(enabled=False)

    tracks = _load_tracks(tracks_path)
    frame_count = _infer_frame_count(tracks)
    if frame_count <= 0:
        raise ValueError(f"No se encontraron frames válidos en {tracks_path}")

    snapshot_runs: list[LiveSnapshotsRun] = []
    last_submitted = -1
    for frame_id in range(frame_count):
        if (frame_id + 1) < int(cfg.min_frames):
            continue
        if last_submitted >= 0 and (frame_id - last_submitted) < int(cfg.snapshot_interval_frames):
            continue
        snapshot_tracks = _snapshot_tracks(tracks, frame_id)
        snapshot_output_dir = snapshots_root / f"frame_{frame_id:06d}"
        result = run_pathcrf_pipeline(
            output_dir=snapshot_output_dir,
            tracks_dict=snapshot_tracks,
            adapter_config=adapter_config,
            inference_config=inference_config,
            render_config=render_config,
        )
        snapshot_runs.append(
            LiveSnapshotsRun(
                frame_id=int(frame_id),
                observed_seconds=float(frame_id + 1) / max(float(cfg.fps), 1e-6),
                snapshot_tracks_path=snapshot_output_dir / f"tracks_snapshot_{frame_id:06d}.json",
                output_dir=snapshot_output_dir,
                result=result,
            )
        )
        last_submitted = frame_id

    sparse_rows: list[dict[str, Any]] = []
    for run in snapshot_runs:
        edge_df = pd.read_parquet(run.result.edge_sequence_path)
        if edge_df.empty:
            continue
        if "frame_id" in edge_df.columns:
            row = edge_df[edge_df["frame_id"] == int(run.frame_id)]
            if row.empty:
                row = edge_df.tail(1)
        else:
            row = edge_df.tail(1)
        edge_row = row.iloc[0].to_dict()
        sparse_rows.append(
            {
                "frame_id": int(run.frame_id),
                "edge_src": edge_row.get("edge_src"),
                "edge_dst": edge_row.get("edge_dst"),
                "edge_team": edge_row.get("edge_team"),
            }
        )
    sparse_edge_df = pd.DataFrame(sparse_rows, columns=["frame_id", "edge_src", "edge_dst", "edge_team"])
    sparse_edge_sequence_path = output_dir / "live_sparse_edge_sequence.parquet"
    sparse_edge_df.to_parquet(sparse_edge_sequence_path, index=False)

    summary_path = output_dir / "live_snapshots_summary.json"
    summary_payload = {
        "tracks_path": str(tracks_path),
        "output_dir": str(output_dir),
        "frame_count": int(frame_count),
        "snapshot_interval_frames": int(cfg.snapshot_interval_frames),
        "min_frames": int(cfg.min_frames),
        "sparse_edge_sequence_path": str(sparse_edge_sequence_path),
        "adapter_config": asdict(adapter_config),
        "inference_config": asdict(inference_config),
        "snapshot_runs": [
            {
                "frame_id": int(run.frame_id),
                "observed_seconds": float(run.observed_seconds),
                "snapshot_tracks_path": str(run.snapshot_tracks_path),
                "output_dir": str(run.output_dir),
                "tracking_path": str(run.result.tracking_path),
                "edge_sequence_path": str(run.result.edge_sequence_path),
                "events_semantic_path": str(run.result.semantic_events_path) if run.result.semantic_events_path else None,
                "summary_path": str(run.result.summary_path),
            }
            for run in snapshot_runs
        ],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary_payload), f, ensure_ascii=False, indent=2)

    return LiveSnapshotsPipelineResult(
        tracks_path=tracks_path,
        output_dir=output_dir,
        frame_count=frame_count,
        snapshot_runs=snapshot_runs,
        sparse_edge_sequence_path=sparse_edge_sequence_path,
        summary_path=summary_path,
    )
