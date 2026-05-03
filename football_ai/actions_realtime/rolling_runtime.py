from __future__ import annotations

import importlib
import json
import sys
import types
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from football_ai.actions.pathcrf_adapter import PathCRFAdapterConfig, PathCRFTracksAdapter
from football_ai.actions.pathcrf_wrapper import (
    PathCRFInferenceConfig,
    PathCRFRenderConfig,
    run_pathcrf_pipeline,
)
from football_ai.pathcrf_slot_mapping import (
    person_slot_to_canonical_id,
    referee_slot_to_canonical_id,
)


TRACK_CLASSES = ("player", "goalkeeper", "referee", "ball")


@dataclass(frozen=True)
class RollingRuntimeConfig:
    cadence_frames: int = 10
    context_frames: int = 250
    emit_delay_frames: int = 25
    emit_frames: int = 10
    fps: float = 25.0
    sample_freq: int = 5
    window_seconds: float = 10.0
    use_crf: bool = True
    decode: str = "indep"
    min_event_duration: int = 10
    min_frames_warmup: int = 50
    repo_path: Path = Path("football_ai/actions/repo/pathcrf")
    trial: int = 120
    model_file: str = "state_dict_best_acc.pt"
    device: str = "auto"
    correct_episode_lasts: bool = False
    evaluate: bool = False
    smooth_edges: bool = False
    edge_smooth_window: int = 3
    tracking_diff_enabled: bool = False
    tracking_diff_atol: float = 1e-6
    tracking_diff_max_rows: int = 5000


@dataclass
class RollingCheckpoint:
    checkpoint_frame: int
    context_start_frame: int
    context_end_frame: int
    context_frames_count: int
    emit_start_frame: int
    emit_end_frame: int
    emitted_edges_count: int
    timing_pathcrf_infer_ms: float
    tracking_rows: int
    edge_rows: int
    snapshot_tracks_path: str | None = None
    snapshot_output_dir: str | None = None
    snapshot_tracking_path: str | None = None
    snapshot_edge_sequence_path: str | None = None
    tracking_diff_path: str | None = None
    tracking_diff_summary: dict[str, Any] | None = None


@dataclass
class RollingEmittedEdge:
    frame_id: int
    checkpoint_frame: int
    context_start_frame: int
    context_end_frame: int
    emit_start_frame: int
    emit_end_frame: int
    edge_src: str | None
    edge_dst: str | None
    canonical_src: str | None
    canonical_dst: str | None
    latency_frames: int
    latency_seconds: float


@dataclass
class RollingRuntimeResult:
    tracking_path: Path
    rolling_edges_path: Path
    emitted_edges_path: Path
    checkpoints_path: Path
    summary_path: Path
    frame_count: int
    num_checkpoints: int
    total_emitted_edges: int


def _resolve_repo_path(repo_path: str | Path) -> Path:
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existe el repo de PathCRF: {path}")
    return path


def _ensure_import_path(repo_path: Path) -> None:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _install_torch_geometric_stub() -> None:
    if "torch_geometric" in sys.modules:
        return
    tg_module = types.ModuleType("torch_geometric")
    tg_data_module = types.ModuleType("torch_geometric.data")

    class _DummyData:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class _DummyBatch:
        @staticmethod
        def from_data_list(data_list: list[Any]) -> list[Any]:
            return data_list

    tg_data_module.Data = _DummyData
    tg_data_module.Batch = _DummyBatch
    tg_module.data = tg_data_module
    sys.modules["torch_geometric"] = tg_module
    sys.modules["torch_geometric.data"] = tg_data_module


def _has_real_torch_geometric() -> bool:
    try:
        importlib.import_module("torch_geometric.data")
        return True
    except ModuleNotFoundError:
        return False


def _select_device(device_name: str) -> torch.device:
    normalized = str(device_name or "auto").strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _slot_to_canonical(slot_name: Any) -> str | None:
    if slot_name is None:
        return None
    text = str(slot_name).strip()
    if not text:
        return None
    person = person_slot_to_canonical_id(text)
    if person is not None:
        return str(person)
    referee = referee_slot_to_canonical_id(text)
    if referee is not None:
        return str(referee)
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _infer_frame_count(tracks: Mapping[str, Any]) -> int:
    return max((len(tracks.get(class_name, [])) for class_name in TRACK_CLASSES), default=0)


def _init_tracks_buffer(tracks: Mapping[str, Any]) -> dict[str, Any]:
    buffer: dict[str, Any] = {}
    for key, value in tracks.items():
        buffer[key] = [] if isinstance(value, list) else value
    return buffer


def _append_tracks_frame(
    buffer: dict[str, Any],
    tracks: Mapping[str, Any],
    frame_index: int,
) -> None:
    for key, value in tracks.items():
        if not isinstance(value, list):
            continue
        if frame_index < len(value):
            buffer[key].append(value[frame_index])


def _ensure_frame_id(edge_df: pd.DataFrame) -> pd.DataFrame:
    if "frame_id" in edge_df.columns:
        return edge_df.copy()
    result = edge_df.reset_index()
    if "frame_id" in result.columns:
        return result
    if "index" in result.columns:
        return result.rename(columns={"index": "frame_id"})
    result["frame_id"] = range(len(result))
    return result


def _build_pathcrf_configs(config: RollingRuntimeConfig) -> tuple[PathCRFAdapterConfig, PathCRFInferenceConfig, PathCRFRenderConfig]:
    adapter_config = PathCRFAdapterConfig(fps=float(config.fps))
    inference_config = PathCRFInferenceConfig(
        repo_path=config.repo_path,
        trial=int(config.trial),
        model_file=str(config.model_file),
        use_crf=bool(config.use_crf),
        decode=str(config.decode),
        correct_episode_lasts=bool(config.correct_episode_lasts),
        evaluate=bool(config.evaluate),
        window_seconds=(float(config.window_seconds) if config.window_seconds is not None else None),
        fps=float(config.fps),
        sample_freq=int(config.sample_freq),
        min_event_duration=int(config.min_event_duration),
        device=str(config.device),
    )
    render_config = PathCRFRenderConfig(enabled=False)
    return adapter_config, inference_config, render_config


def _build_tracking_diff(
    snapshot_df: pd.DataFrame,
    offline_df: pd.DataFrame,
    *,
    atol: float,
    max_rows: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary: dict[str, Any] = {
        "row_count_snapshot": int(len(snapshot_df)),
        "row_count_offline": int(len(offline_df)),
        "diff_cells": 0,
        "diff_rows": 0,
        "diff_columns": {},
        "columns_only_snapshot": [],
        "columns_only_offline": [],
    }
    if snapshot_df.empty or offline_df.empty:
        return pd.DataFrame(), summary

    snap_cols = set(snapshot_df.columns)
    off_cols = set(offline_df.columns)
    summary["columns_only_snapshot"] = sorted(list(snap_cols - off_cols))
    summary["columns_only_offline"] = sorted(list(off_cols - snap_cols))

    common_cols = [col for col in snapshot_df.columns if col in off_cols]
    if not common_cols:
        return pd.DataFrame(), summary

    if "frame_id" in common_cols:
        snapshot_df = snapshot_df.sort_values("frame_id")
        offline_df = offline_df.sort_values("frame_id")
        merged = snapshot_df[common_cols].merge(
            offline_df[common_cols],
            on="frame_id",
            suffixes=("_snapshot", "_offline"),
            how="inner",
        )
        frame_key = "frame_id"
    else:
        aligned_len = min(len(snapshot_df), len(offline_df))
        snapshot_df = snapshot_df.iloc[:aligned_len]
        offline_df = offline_df.iloc[:aligned_len]
        merged = pd.concat(
            [snapshot_df[common_cols].reset_index(drop=True), offline_df[common_cols].reset_index(drop=True)],
            axis=1,
        )
        merged["frame_id"] = np.arange(aligned_len, dtype=np.int64)
        frame_key = "frame_id"

    diff_rows: list[dict[str, Any]] = []
    for col in common_cols:
        if col == frame_key:
            continue
        col_snap = merged[f"{col}_snapshot"] if f"{col}_snapshot" in merged.columns else merged[col]
        col_off = merged[f"{col}_offline"] if f"{col}_offline" in merged.columns else merged[col]
        if pd.api.types.is_numeric_dtype(col_snap) and pd.api.types.is_numeric_dtype(col_off):
            mismatch = ~np.isclose(col_snap.to_numpy(), col_off.to_numpy(), atol=float(atol), equal_nan=True)
            delta = (col_snap - col_off).abs()
            max_delta = float(delta[mismatch].max()) if mismatch.any() else 0.0
        else:
            snap_vals = col_snap.astype(object).where(col_snap.notna(), None)
            off_vals = col_off.astype(object).where(col_off.notna(), None)
            mismatch = snap_vals != off_vals
            max_delta = None

        mismatch_count = int(np.sum(mismatch))
        if mismatch_count <= 0:
            continue
        summary["diff_cells"] += mismatch_count
        summary["diff_columns"][col] = {
            "mismatch_count": mismatch_count,
            "max_abs_diff": max_delta,
        }

        if len(diff_rows) < int(max_rows):
            indices = np.flatnonzero(mismatch)
            for idx in indices[: max_rows - len(diff_rows)]:
                row = {
                    "frame_id": int(merged[frame_key].iloc[idx]),
                    "column": col,
                    "snapshot_value": _json_safe(col_snap.iloc[idx]),
                    "offline_value": _json_safe(col_off.iloc[idx]),
                }
                if pd.api.types.is_numeric_dtype(col_snap) and pd.api.types.is_numeric_dtype(col_off):
                    row["abs_diff"] = float(abs(col_snap.iloc[idx] - col_off.iloc[idx]))
                diff_rows.append(row)

    summary["diff_rows"] = int(len(diff_rows))
    if summary["diff_cells"] > 0:
        summary["diff_rows_truncated"] = len(diff_rows) >= int(max_rows)
    return pd.DataFrame(diff_rows), summary


class RollingRuntime:
    def __init__(self, config: RollingRuntimeConfig | None = None, tracking_df: pd.DataFrame | None = None):
        self.config = config or RollingRuntimeConfig()
        self._tracking_df = tracking_df
        self._repo_path = _resolve_repo_path(self.config.repo_path)
        self._load_pathcrf_modules()
        self._device = _select_device(self.config.device)
        self._build_model()
        self.reset()

    def reset(self) -> None:
        self._checkpoints: list[RollingCheckpoint] = []
        self._emitted_edges: list[RollingEmittedEdge] = []
        self._rolling_edge_buffer: list[dict[str, Any]] = []
        self._last_inference_frame = -10**9
        self._edge_src_history: deque[dict[int, str | None]] = deque(maxlen=self.config.edge_smooth_window)
        self._edge_dst_history: deque[dict[int, str | None]] = deque(maxlen=self.config.edge_smooth_window)

    def _load_pathcrf_modules(self) -> None:
        _ensure_import_path(self._repo_path)
        save_path = self._repo_path / "saved" / f"{int(self.config.trial):03d}"
        with (save_path / "args.json").open("r", encoding="utf-8") as f:
            self._trial_args = json.load(f)
        agent_model = str(self._trial_args.get("agent_model", "")).strip().lower()
        if agent_model != "gat" and not _has_real_torch_geometric():
            _install_torch_geometric_stub()
        elif agent_model == "gat" and not _has_real_torch_geometric():
            raise ModuleNotFoundError(
                "El checkpoint de PathCRF usa `agent_model=gat`, pero `torch_geometric` no esta instalado en el entorno."
            )
        self._pathcrf_utils = importlib.import_module("models.utils")
        self._pathcrf_inference = importlib.import_module("inference")
        self._pathcrf_postprocess = importlib.import_module("datatools.postprocess")

    def _build_model(self) -> None:
        self._model = self._pathcrf_utils.build_model(self._trial_args, device=self._device)
        model_path = Path(
            self._pathcrf_utils.resolve_model_path(
                str(self._repo_path / "saved" / f"{int(self.config.trial):03d}"),
                self.config.model_file,
            )
        ).resolve()
        state_dict = torch.load(model_path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def process_frame(self, frame_index: int) -> dict[str, Any]:
        cadence_frames = max(int(self.config.cadence_frames), 1)
        should_infer = (
            (int(frame_index) + 1) >= int(self.config.min_frames_warmup)
            and (
                self._last_inference_frame < 0
                or (int(frame_index) - int(self._last_inference_frame)) >= cadence_frames
            )
        )

        if not should_infer:
            return {
                "should_infer": False,
                "frame_id": int(frame_index),
            }

        self._last_inference_frame = int(frame_index)

        context_start = max(0, int(frame_index) - int(self.config.context_frames) + 1)
        context_end = int(frame_index)
        context_df = self._extract_context_window(context_start, context_end)
        if context_df.empty:
            return {
                "should_infer": True,
                "frame_id": int(frame_index),
                "context_start": context_start,
                "context_end": context_end,
                "skipped": True,
                "reason": "empty_context",
            }

        started = perf_counter()
        edge_sequence_df = self._run_pathcrf_inference(context_df)
        timing_infer_ms = (perf_counter() - started) * 1000.0

        emit_end = int(frame_index) - int(self.config.emit_delay_frames)
        emit_start = max(0, emit_end - int(self.config.emit_frames) + 1)
        if emit_start > emit_end:
            emit_start = emit_end

        checkpoint = RollingCheckpoint(
            checkpoint_frame=int(frame_index),
            context_start_frame=context_start,
            context_end_frame=context_end,
            context_frames_count=len(context_df),
            emit_start_frame=emit_start,
            emit_end_frame=emit_end,
            emitted_edges_count=0,
            timing_pathcrf_infer_ms=timing_infer_ms,
            tracking_rows=len(self._tracking_df) if self._tracking_df is not None else 0,
            edge_rows=len(edge_sequence_df),
        )

        fps = float(self.config.fps)

        for row_idx, row in edge_sequence_df.iterrows():
            local_frame = _coerce_int(row.get("frame_id"))
            if local_frame is None:
                local_frame = _coerce_int(row_idx)
            if local_frame is None:
                continue
            global_frame = context_start + int(local_frame)
            edge_src = _coerce_optional_str(row.get("edge_src"))
            edge_dst = _coerce_optional_str(row.get("edge_dst"))

            rolling_record = {
                "frame_id": global_frame,
                "checkpoint_frame": int(frame_index),
                "context_start_frame": context_start,
                "context_end_frame": context_end,
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "canonical_src": _slot_to_canonical(edge_src),
                "canonical_dst": _slot_to_canonical(edge_dst),
            }
            self._rolling_edge_buffer.append(rolling_record)

            if not (emit_start <= global_frame <= emit_end):
                continue

            emit_src = self._get_smoothed_edge_src(global_frame, edge_src)
            emit_dst = self._get_smoothed_edge_dst(global_frame, edge_dst)
            latency_frames = int(frame_index) - global_frame
            latency_seconds = latency_frames / max(fps, 1e-6)

            emitted = RollingEmittedEdge(
                frame_id=global_frame,
                checkpoint_frame=int(frame_index),
                context_start_frame=context_start,
                context_end_frame=context_end,
                emit_start_frame=emit_start,
                emit_end_frame=emit_end,
                edge_src=emit_src,
                edge_dst=emit_dst,
                canonical_src=_slot_to_canonical(emit_src),
                canonical_dst=_slot_to_canonical(emit_dst),
                latency_frames=latency_frames,
                latency_seconds=latency_seconds,
            )
            self._emitted_edges.append(emitted)

        checkpoint.emitted_edges_count = sum(
            1 for e in self._emitted_edges if e.checkpoint_frame == int(frame_index)
        )
        self._checkpoints.append(checkpoint)

        return {
            "should_infer": True,
            "frame_id": int(frame_index),
            "context_start": context_start,
            "context_end": context_end,
            "context_frames_count": len(context_df),
            "emit_start": emit_start,
            "emit_end": emit_end,
            "edge_rows": len(edge_sequence_df),
            "emitted_edges_this_checkpoint": checkpoint.emitted_edges_count,
            "timing_pathcrf_infer_ms": timing_infer_ms,
        }

    def _extract_context_window(self, start: int, end: int) -> pd.DataFrame:
        if self._tracking_df is None:
            return pd.DataFrame()
        if "frame_id" not in self._tracking_df.columns:
            return self._tracking_df.iloc[start : end + 1].copy()
        mask = (self._tracking_df["frame_id"] >= start) & (self._tracking_df["frame_id"] <= end)
        df = self._tracking_df.loc[mask].copy()
        if "frame_id" in df.columns:
            df["frame_id"] = df["frame_id"] - start
        return df

    def _run_pathcrf_inference(self, tracking_df: pd.DataFrame) -> pd.DataFrame:
        effective_fps = float(self.config.fps)
        effective_sample_freq = int(self.config.sample_freq)
        effective_window_seconds = (
            float(self.config.window_seconds)
            if self.config.window_seconds is not None
            else self._trial_args.get("window_seconds")
        )

        _, _, micro_pred_df, _ = self._pathcrf_inference.inference(
            model=self._model,
            tracking=tracking_df,
            use_crf=bool(self.config.use_crf),
            decode=str(self.config.decode),
            correct_episode_lasts=bool(self.config.correct_episode_lasts),
            evaluate=bool(self.config.evaluate),
            window_seconds=effective_window_seconds,
            fps=effective_fps,
            sample_freq=effective_sample_freq,
        )

        if {"edge_src", "edge_dst"}.issubset(micro_pred_df.columns):
            return micro_pred_df.copy()
        return self._pathcrf_postprocess.edge_probs_to_seq(micro_pred_df)

    def _get_smoothed_edge_src(self, frame_id: int, raw_src: str | None) -> str | None:
        self._edge_src_history.append({frame_id: raw_src})
        if not self.config.smooth_edges or len(self._edge_src_history) < 2:
            return raw_src
        return _mode_edge_label(self._edge_src_history)

    def _get_smoothed_edge_dst(self, frame_id: int, raw_dst: str | None) -> str | None:
        self._edge_dst_history.append({frame_id: raw_dst})
        if not self.config.smooth_edges or len(self._edge_dst_history) < 2:
            return raw_dst
        return _mode_edge_label(self._edge_dst_history)

    def build_result(self, output_dir: str | Path) -> RollingRuntimeResult:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        tracking_path = output_dir / "tracking.parquet"
        if self._tracking_df is not None:
            self._tracking_df.to_parquet(tracking_path, index=False)
        else:
            pd.DataFrame().to_parquet(tracking_path, index=False)

        rolling_edges_path = output_dir / "rolling_edges_per_frame.parquet"
        rolling_edges_df = pd.DataFrame(self._rolling_edge_buffer)
        rolling_edges_df.to_parquet(rolling_edges_path, index=False)

        emitted_edges_path = output_dir / "emitted_edges.parquet"
        emitted_list = [asdict(e) for e in self._emitted_edges]
        emitted_df = pd.DataFrame(emitted_list)
        emitted_df.to_parquet(emitted_edges_path, index=False)

        checkpoints_path = output_dir / "runtime_checkpoints.json"
        checkpoints_payload = [_json_safe(asdict(cp)) for cp in self._checkpoints]
        with checkpoints_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoints_payload, f, ensure_ascii=False, indent=2)

        total_emitted = len(self._emitted_edges)
        frame_count = len(self._tracking_df) if self._tracking_df is not None else 0
        summary_path = output_dir / "summary.json"
        summary_payload = {
            "config": {
                "cadence_frames": int(self.config.cadence_frames),
                "context_frames": int(self.config.context_frames),
                "emit_delay_frames": int(self.config.emit_delay_frames),
                "emit_frames": int(self.config.emit_frames),
                "fps": float(self.config.fps),
                "sample_freq": int(self.config.sample_freq),
                "window_seconds": float(self.config.window_seconds) if self.config.window_seconds else None,
                "use_crf": bool(self.config.use_crf),
                "decode": str(self.config.decode),
                "min_event_duration": int(self.config.min_event_duration),
                "min_frames_warmup": int(self.config.min_frames_warmup),
                "trial": int(self.config.trial),
                "model_file": str(self.config.model_file),
            },
            "tracking_path": str(tracking_path),
            "rolling_edges_path": str(rolling_edges_path),
            "emitted_edges_path": str(emitted_edges_path),
            "checkpoints_path": str(checkpoints_path),
            "frame_count": frame_count,
            "num_checkpoints": len(self._checkpoints),
            "total_emitted_edges": total_emitted,
            "total_rolling_edges": len(self._rolling_edge_buffer),
            "tracking_source": "diagnostic_tracking_parquet",
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(summary_payload), f, ensure_ascii=False, indent=2)

        return RollingRuntimeResult(
            tracking_path=tracking_path,
            rolling_edges_path=rolling_edges_path,
            emitted_edges_path=emitted_edges_path,
            checkpoints_path=checkpoints_path,
            summary_path=summary_path,
            frame_count=frame_count,
            num_checkpoints=len(self._checkpoints),
            total_emitted_edges=total_emitted,
        )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, (pd.Timestamp,)):
            return None
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _mode_edge_label(history: deque[dict[int, str | None]]) -> str | None:
    all_vals = []
    for h in history:
        for v in h.values():
            if v is not None:
                all_vals.append(v)
    if not all_vals:
        return None
    counts: dict[str, int] = {}
    for v in all_vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def _build_tracking_from_adapter(tracks: dict[str, Any], config: RollingRuntimeConfig) -> pd.DataFrame:
    adapter_config = PathCRFAdapterConfig(
        fps=float(config.fps),
    )
    adapter = PathCRFTracksAdapter(adapter_config)
    frame_count = max(
        (len(tracks.get(class_name, [])) for class_name in TRACK_CLASSES),
        default=0,
    )
    if frame_count <= 0:
        raise ValueError("No se encontraron frames en los tracks")
    records = adapter._build_track_records(tracks, frame_count)
    person_assignments, referee_assignments, _ = adapter._assign_slots(records)
    frame_df = adapter._build_tracking_dataframe(
        tracks=tracks,
        frame_count=frame_count,
        records=records,
        person_assignments=person_assignments,
        referee_assignments=referee_assignments,
    )
    return frame_df


def _run_snapshot_rolling_pipeline(
    *,
    tracks_path: Path,
    output_dir: Path,
    config: RollingRuntimeConfig,
    max_frames: int | None,
) -> RollingRuntimeResult:
    with tracks_path.open("r", encoding="utf-8") as f:
        tracks = json.load(f)

    frame_count = _infer_frame_count(tracks)
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 0:
        raise ValueError("No se encontraron frames en los tracks")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_root = output_dir / "snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)

    adapter_config, inference_config, render_config = _build_pathcrf_configs(config)

    offline_tracking_df: pd.DataFrame | None = None
    offline_tracking_path: Path | None = None
    last_snapshot_tracking_path: Path | None = None
    if config.tracking_diff_enabled:
        offline_tracking_df = _build_tracking_from_adapter(tracks, config)
        offline_tracking_path = output_dir / "offline_tracking.parquet"
        offline_tracking_df.to_parquet(offline_tracking_path, index=False)

    buffer = _init_tracks_buffer(tracks)
    rolling_edges: list[dict[str, Any]] = []
    emitted_edges: list[RollingEmittedEdge] = []
    checkpoints: list[RollingCheckpoint] = []
    edge_src_history: deque[dict[int, str | None]] = deque(maxlen=config.edge_smooth_window)
    edge_dst_history: deque[dict[int, str | None]] = deque(maxlen=config.edge_smooth_window)
    last_inference_frame = -10**9

    def _maybe_smooth_edge_src(frame_id: int, raw_src: str | None) -> str | None:
        edge_src_history.append({frame_id: raw_src})
        if not config.smooth_edges or len(edge_src_history) < 2:
            return raw_src
        return _mode_edge_label(edge_src_history)

    def _maybe_smooth_edge_dst(frame_id: int, raw_dst: str | None) -> str | None:
        edge_dst_history.append({frame_id: raw_dst})
        if not config.smooth_edges or len(edge_dst_history) < 2:
            return raw_dst
        return _mode_edge_label(edge_dst_history)

    diff_root = output_dir / "tracking_diffs"
    if config.tracking_diff_enabled:
        diff_root.mkdir(parents=True, exist_ok=True)

    for frame_index in range(frame_count):
        _append_tracks_frame(buffer, tracks, frame_index)

        should_infer = (
            (int(frame_index) + 1) >= int(config.min_frames_warmup)
            and (
                last_inference_frame < 0
                or (int(frame_index) - int(last_inference_frame)) >= int(config.cadence_frames)
            )
        )
        if not should_infer:
            continue

        last_inference_frame = int(frame_index)

        snapshot_tracks_path = snapshots_root / f"tracks_snapshot_{frame_index:06d}.json"

        snapshot_output_dir = snapshots_root / f"frame_{frame_index:06d}"
        started = perf_counter()
        result = run_pathcrf_pipeline(
            output_dir=snapshot_output_dir,
            tracks_dict=buffer,
            adapter_config=adapter_config,
            inference_config=inference_config,
            render_config=render_config,
        )
        timing_ms = (perf_counter() - started) * 1000.0

        edge_df = pd.read_parquet(result.edge_sequence_path)
        edge_df = _ensure_frame_id(edge_df)

        tracking_df = pd.read_parquet(result.tracking_path)
        tracking_rows = int(len(tracking_df))
        last_snapshot_tracking_path = result.tracking_path

        emit_end = int(frame_index) - int(config.emit_delay_frames)
        emit_start = emit_end - int(config.emit_frames) + 1
        if emit_start > emit_end:
            emit_start = emit_end

        emitted_this = 0
        for row in edge_df.to_dict(orient="records"):
            local_frame = _coerce_int(row.get("frame_id"))
            if local_frame is None:
                continue
            global_frame = int(local_frame)
            edge_src = _coerce_optional_str(row.get("edge_src"))
            edge_dst = _coerce_optional_str(row.get("edge_dst"))
            rolling_edges.append(
                {
                    "frame_id": global_frame,
                    "checkpoint_frame": int(frame_index),
                    "context_start_frame": 0,
                    "context_end_frame": int(frame_index),
                    "emit_start_frame": int(emit_start),
                    "emit_end_frame": int(emit_end),
                    "edge_src": edge_src,
                    "edge_dst": edge_dst,
                    "canonical_src": _slot_to_canonical(edge_src),
                    "canonical_dst": _slot_to_canonical(edge_dst),
                }
            )

            if not (int(emit_start) <= global_frame <= int(emit_end)):
                continue

            emit_src = _maybe_smooth_edge_src(global_frame, edge_src)
            emit_dst = _maybe_smooth_edge_dst(global_frame, edge_dst)
            latency_frames = int(frame_index) - global_frame
            latency_seconds = latency_frames / max(float(config.fps), 1e-6)

            emitted_edges.append(
                RollingEmittedEdge(
                    frame_id=global_frame,
                    checkpoint_frame=int(frame_index),
                    context_start_frame=0,
                    context_end_frame=int(frame_index),
                    emit_start_frame=int(emit_start),
                    emit_end_frame=int(emit_end),
                    edge_src=emit_src,
                    edge_dst=emit_dst,
                    canonical_src=_slot_to_canonical(emit_src),
                    canonical_dst=_slot_to_canonical(emit_dst),
                    latency_frames=latency_frames,
                    latency_seconds=latency_seconds,
                )
            )
            emitted_this += 1

        diff_path = None
        diff_summary = None
        if config.tracking_diff_enabled and offline_tracking_df is not None:
            if "frame_id" in offline_tracking_df.columns:
                offline_slice = offline_tracking_df[offline_tracking_df["frame_id"] <= int(frame_index)].copy()
            else:
                offline_slice = offline_tracking_df.iloc[:tracking_rows].copy()
            diff_df, diff_summary = _build_tracking_diff(
                tracking_df,
                offline_slice,
                atol=float(config.tracking_diff_atol),
                max_rows=int(config.tracking_diff_max_rows),
            )
            if not diff_df.empty:
                diff_path = diff_root / f"tracking_diff_{frame_index:06d}.parquet"
                diff_df.to_parquet(diff_path, index=False)

        checkpoint = RollingCheckpoint(
            checkpoint_frame=int(frame_index),
            context_start_frame=0,
            context_end_frame=int(frame_index),
            context_frames_count=int(tracking_rows),
            emit_start_frame=int(emit_start),
            emit_end_frame=int(emit_end),
            emitted_edges_count=int(emitted_this),
            timing_pathcrf_infer_ms=float(timing_ms),
            tracking_rows=int(tracking_rows),
            edge_rows=int(len(edge_df)),
            snapshot_tracks_path=str(snapshot_tracks_path),
            snapshot_output_dir=str(snapshot_output_dir),
            snapshot_tracking_path=str(result.tracking_path),
            snapshot_edge_sequence_path=str(result.edge_sequence_path),
            tracking_diff_path=str(diff_path) if diff_path is not None else None,
            tracking_diff_summary=diff_summary,
        )
        checkpoints.append(checkpoint)

        print(
            f"[frame {frame_index}] "
            f"snapshot=0..{frame_index} "
            f"edges={len(edge_df)} "
            f"emit=[{emit_start},{emit_end}] "
            f"emitted={emitted_this} "
            f"infer_ms={timing_ms:.0f}"
        )

    rolling_edges_path = output_dir / "rolling_edges_per_frame.parquet"
    pd.DataFrame(rolling_edges).to_parquet(rolling_edges_path, index=False)

    emitted_edges_path = output_dir / "emitted_edges.parquet"
    emitted_df = pd.DataFrame([asdict(item) for item in emitted_edges])
    emitted_df.to_parquet(emitted_edges_path, index=False)

    checkpoints_path = output_dir / "runtime_checkpoints.json"
    with checkpoints_path.open("w", encoding="utf-8") as f:
        json.dump([_json_safe(asdict(cp)) for cp in checkpoints], f, ensure_ascii=False, indent=2)

    summary_path = output_dir / "summary.json"
    summary_payload = {
        "config": {
            "cadence_frames": int(config.cadence_frames),
            "emit_delay_frames": int(config.emit_delay_frames),
            "emit_frames": int(config.emit_frames),
            "fps": float(config.fps),
            "sample_freq": int(config.sample_freq),
            "window_seconds": float(config.window_seconds) if config.window_seconds else None,
            "use_crf": bool(config.use_crf),
            "decode": str(config.decode),
            "min_event_duration": int(config.min_event_duration),
            "min_frames_warmup": int(config.min_frames_warmup),
            "trial": int(config.trial),
            "model_file": str(config.model_file),
            "tracking_diff_enabled": bool(config.tracking_diff_enabled),
        },
        "tracking_path": (
            str(offline_tracking_path)
            if offline_tracking_path is not None
            else (str(last_snapshot_tracking_path) if last_snapshot_tracking_path is not None else None)
        ),
        "rolling_edges_path": str(rolling_edges_path),
        "emitted_edges_path": str(emitted_edges_path),
        "checkpoints_path": str(checkpoints_path),
        "frame_count": int(frame_count),
        "num_checkpoints": int(len(checkpoints)),
        "total_emitted_edges": int(len(emitted_edges)),
        "total_rolling_edges": int(len(rolling_edges)),
        "tracking_source": "snapshot_adapter_rolling",
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary_payload), f, ensure_ascii=False, indent=2)

    return RollingRuntimeResult(
        tracking_path=(
            offline_tracking_path
            if offline_tracking_path is not None
            else (last_snapshot_tracking_path if last_snapshot_tracking_path is not None else emitted_edges_path)
        ),
        rolling_edges_path=rolling_edges_path,
        emitted_edges_path=emitted_edges_path,
        checkpoints_path=checkpoints_path,
        summary_path=summary_path,
        frame_count=int(frame_count),
        num_checkpoints=int(len(checkpoints)),
        total_emitted_edges=int(len(emitted_edges)),
    )


def run_rolling_pipeline(
    *,
    tracks_path: str | Path,
    output_dir: str | Path,
    config: RollingRuntimeConfig | None = None,
    max_frames: int | None = None,
    tracking_path: str | Path | None = None,
) -> RollingRuntimeResult:
    tracks_path = Path(tracks_path).expanduser().resolve()
    if not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON: {tracks_path}")

    output_dir = Path(output_dir).expanduser().resolve()
    config = config or RollingRuntimeConfig()

    if tracking_path is None:
        print("Running snapshot-backed rolling (adapter parity with live snapshots)...")
        return _run_snapshot_rolling_pipeline(
            tracks_path=tracks_path,
            output_dir=output_dir,
            config=config,
            max_frames=max_frames,
        )

    tracking_path = Path(tracking_path).expanduser().resolve()
    if not tracking_path.exists():
        raise FileNotFoundError(f"No existe el tracking parquet: {tracking_path}")
    print(f"Loading pre-built tracking from: {tracking_path}")
    tracking_df = pd.read_parquet(tracking_path)
    if max_frames is not None:
        target_end = min(len(tracking_df), int(max_frames))
        tracking_df = tracking_df.iloc[:target_end]

    total_frames = len(tracking_df)
    print(f"Tracking ready: {total_frames} frames")

    runtime = RollingRuntime(config=config, tracking_df=tracking_df)

    for frame_index in range(total_frames):
        meta = runtime.process_frame(frame_index)
        if meta.get("should_infer"):
            print(
                f"[frame {meta['frame_id']}] "
                f"context=[{meta.get('context_start')},{meta.get('context_end')}] "
                f"edges={meta.get('edge_rows')} "
                f"emit=[{meta.get('emit_start')},{meta.get('emit_end')}] "
                f"emitted={meta.get('emitted_edges_this_checkpoint')} "
                f"infer_ms={meta.get('timing_pathcrf_infer_ms', 0):.0f}"
            )

    return runtime.build_result(output_dir)