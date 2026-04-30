from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import pandas as pd
import torch

from .detector import ActionsDetector, ActionsDetectorConfig, ActionsDetectorSummary
from .pathcrf_setpieces import classify_episode_starts
from .pathcrf_shot import apply_shot_heuristic


@dataclass(frozen=True)
class ActionsRuntimeConfig:
    detector: ActionsDetectorConfig = ActionsDetectorConfig()
    repo_path: Path = Path("football_ai/actions/repo/pathcrf")
    trial: int = 120
    model_file: str = "state_dict_best_acc.pt"
    use_crf: bool = True
    decode: str = "indep"
    correct_episode_lasts: bool = False
    evaluate: bool = False
    cadence_frames: int = 10
    min_frames_warmup: int = 50
    min_event_duration: int = 10
    window_seconds: float | None = None
    fps: float | None = None
    sample_freq: int | None = None
    device: str = "auto"
    confirmation_cooldown_frames: int = 15


@dataclass
class ActionsRuntimeResult:
    raw_edge: dict[str, Any] | None
    confirmed_action: dict[str, Any] | None
    action_metadata: dict[str, Any]
    summary: ActionsDetectorSummary


class ActionsRuntime:
    def __init__(self, config: ActionsRuntimeConfig | None = None):
        self.config = config or ActionsRuntimeConfig()
        self.detector = ActionsDetector(self.config.detector)
        self._repo_path = self._resolve_repo_path(self.config.repo_path)
        self._trial_args = self._load_trial_args(self._repo_path, self.config.trial)
        self._pathcrf_utils, self._pathcrf_inference, self._pathcrf_postprocess = self._import_pathcrf_modules(
            self._repo_path,
            self._trial_args,
        )
        self._device = self._select_device(self.config.device)
        self._model = self._build_model()
        self.reset()

    def reset(self) -> None:
        self.detector.reset()
        self._last_raw_edge: dict[str, Any] | None = None
        self._last_confirmed_action: dict[str, Any] | None = None
        self._last_confirmed_frame = -10**9
        self._last_inference_frame = -10**9
        self._emitted_event_keys: set[tuple[Any, ...]] = set()

    def process_frame(self, frame_index: int, clean_packet: Mapping[str, Any]) -> ActionsRuntimeResult:
        timings: dict[str, float] = {}

        started = perf_counter()
        snapshot = self.detector.update(int(frame_index), clean_packet)
        timings["state_update_ms"] = (perf_counter() - started) * 1000.0

        cadence_frames = max(int(self.config.cadence_frames), 1)
        should_infer = (
            (int(frame_index) + 1) >= int(self.config.min_frames_warmup)
            and (self._last_inference_frame < 0 or (int(frame_index) - int(self._last_inference_frame)) >= cadence_frames)
        )
        if not should_infer:
            metadata = {
                "should_infer": False,
                "window_start_frame": int(snapshot.window_start_frame),
                "window_end_frame": int(snapshot.window_end_frame),
                "frame_count": int(snapshot.frame_count),
                **timings,
                "feature_update_ms": 0.0,
                "pathcrf_infer_ms": 0.0,
                "postprocess_ms": 0.0,
                "total_actions_ms": sum(timings.values()),
            }
            return ActionsRuntimeResult(
                raw_edge=self._last_raw_edge,
                confirmed_action=None,
                action_metadata=metadata,
                summary=self.detector.build_materialized_slot_state()[1],
            )

        started = perf_counter()
        tracking_df, summary = self.detector.build_tracking_dataframe()
        timings["feature_update_ms"] = (perf_counter() - started) * 1000.0
        self._last_inference_frame = int(frame_index)

        started = perf_counter()
        edge_sequence_df, semantic_events_df = self._run_pathcrf(tracking_df)
        timings["pathcrf_infer_ms"] = (perf_counter() - started) * 1000.0

        started = perf_counter()
        raw_edge = self._build_latest_raw_edge(
            edge_sequence_df=edge_sequence_df,
            window_start_frame=snapshot.window_start_frame,
            window_end_frame=snapshot.window_end_frame,
            summary=summary,
        )
        confirmed_action = self._build_confirmed_action(
            semantic_events_df=semantic_events_df,
            window_start_frame=snapshot.window_start_frame,
            window_end_frame=snapshot.window_end_frame,
            summary=summary,
        )
        timings["postprocess_ms"] = (perf_counter() - started) * 1000.0

        self._last_raw_edge = raw_edge
        if confirmed_action is not None:
            self._last_confirmed_action = confirmed_action

        metadata = {
            "should_infer": True,
            "window_start_frame": int(snapshot.window_start_frame),
            "window_end_frame": int(snapshot.window_end_frame),
            "frame_count": int(snapshot.frame_count),
            "edge_rows": int(len(edge_sequence_df)),
            "event_rows": int(len(semantic_events_df)),
            **timings,
        }
        metadata["total_actions_ms"] = sum(float(value) for value in timings.values())

        return ActionsRuntimeResult(
            raw_edge=raw_edge,
            confirmed_action=confirmed_action,
            action_metadata=metadata,
            summary=summary,
        )

    def _build_model(self):
        model = self._pathcrf_utils.build_model(self._trial_args, device=self._device)
        model_path = Path(
            self._pathcrf_utils.resolve_model_path(
                str(self._repo_path / "saved" / f"{int(self.config.trial):03d}"),
                self.config.model_file,
            )
        ).resolve()
        state_dict = torch.load(model_path, map_location=self._device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _run_pathcrf(self, tracking_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        effective_fps = float(self.config.fps) if self.config.fps is not None else float(self._trial_args.get("fps", 25.0))
        effective_sample_freq = (
            int(self.config.sample_freq) if self.config.sample_freq is not None else int(self._trial_args.get("sample_freq", 5))
        )
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
            edge_sequence_df = micro_pred_df.copy()
        else:
            edge_sequence_df = self._pathcrf_postprocess.edge_probs_to_seq(micro_pred_df)

        events_df = self._pathcrf_postprocess.detect_events(
            tracking=tracking_df,
            edge_seq=edge_sequence_df,
            min_dur=int(self.config.min_event_duration),
        )
        semantic_events_df = classify_episode_starts(events_df)
        semantic_events_df = apply_shot_heuristic(semantic_events_df, fps=effective_fps)
        return edge_sequence_df, semantic_events_df

    def _build_latest_raw_edge(
        self,
        *,
        edge_sequence_df: pd.DataFrame,
        window_start_frame: int,
        window_end_frame: int,
        summary: ActionsDetectorSummary,
    ) -> dict[str, Any] | None:
        if edge_sequence_df.empty:
            return None
        row = edge_sequence_df.iloc[-1].to_dict()
        local_frame = self._coerce_int(row.get("frame_id"))
        absolute_frame = window_end_frame if local_frame is None else window_start_frame + local_frame
        edge_src = self._coerce_optional_str(row.get("edge_src"))
        edge_dst = self._coerce_optional_str(row.get("edge_dst"))
        payload = {
            "frame_id": absolute_frame,
            "window_frame_id": local_frame,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_team": self._coerce_optional_str(row.get("edge_team")) if "edge_team" in row else None,
            "edge_src_track_id": self._slot_to_track_id(edge_src, summary.person_slot_assignments),
            "edge_dst_track_id": self._slot_to_track_id(edge_dst, summary.person_slot_assignments),
        }
        for key, value in row.items():
            if key in payload:
                continue
            payload[key] = self._coerce_scalar(value)
        return payload

    def _build_confirmed_action(
        self,
        *,
        semantic_events_df: pd.DataFrame,
        window_start_frame: int,
        window_end_frame: int,
        summary: ActionsDetectorSummary,
    ) -> dict[str, Any] | None:
        if semantic_events_df.empty:
            return None
        for _, row in semantic_events_df.sort_values("frame_id").iterrows():
            local_frame = self._coerce_int(row.get("frame_id"))
            absolute_frame = window_end_frame if local_frame is None else window_start_frame + local_frame
            if absolute_frame > window_end_frame:
                continue
            event_key = (
                self._coerce_optional_str(row.get("event_type_semantic") or row.get("event_type")),
                self._coerce_optional_str(row.get("player_id")),
                self._coerce_optional_str(row.get("receiver_id")),
                absolute_frame,
            )
            if event_key in self._emitted_event_keys:
                continue
            if absolute_frame - self._last_confirmed_frame < int(self.config.confirmation_cooldown_frames):
                continue
            payload = {column: self._coerce_scalar(value) for column, value in row.to_dict().items()}
            player_slot_id = self._coerce_optional_str(payload.get("player_id"))
            receiver_slot_id = self._coerce_optional_str(payload.get("receiver_id"))
            payload["frame_id"] = absolute_frame
            payload["window_frame_id"] = local_frame
            payload["event_type"] = payload.get("event_type_semantic") or payload.get("event_type")
            payload["player_slot_id"] = player_slot_id
            payload["receiver_slot_id"] = receiver_slot_id
            payload["player_track_id"] = self._slot_to_track_id(player_slot_id, summary.person_slot_assignments)
            payload["receiver_track_id"] = self._slot_to_track_id(receiver_slot_id, summary.person_slot_assignments)
            self._emitted_event_keys.add(event_key)
            self._last_confirmed_frame = absolute_frame
            return payload
        return None

    @staticmethod
    def _resolve_repo_path(repo_path: str | Path) -> Path:
        path = Path(repo_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"No existe el repo de PathCRF: {path}")
        return path

    @staticmethod
    def _load_trial_args(repo_path: Path, trial: int) -> dict[str, Any]:
        args_path = repo_path / "saved" / f"{int(trial):03d}" / "args.json"
        with args_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _ensure_import_path(repo_path: Path) -> None:
        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    @staticmethod
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

    @staticmethod
    def _has_real_torch_geometric() -> bool:
        try:
            importlib.import_module("torch_geometric.data")
            return True
        except ModuleNotFoundError:
            return False

    @classmethod
    def _import_pathcrf_modules(cls, repo_path: Path, trial_args: dict[str, Any]):
        cls._ensure_import_path(repo_path)
        agent_model = str(trial_args.get("agent_model", "")).strip().lower()
        if agent_model != "gat" and not cls._has_real_torch_geometric():
            cls._install_torch_geometric_stub()
        elif agent_model == "gat" and not cls._has_real_torch_geometric():
            raise ModuleNotFoundError(
                "El checkpoint de PathCRF usa `agent_model=gat`, pero `torch_geometric` no está instalado en el entorno."
            )
        return (
            importlib.import_module("models.utils"),
            importlib.import_module("inference"),
            importlib.import_module("datatools.postprocess"),
        )

    @staticmethod
    def _select_device(device_name: str) -> torch.device:
        normalized = str(device_name or "auto").strip().lower()
        if normalized == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None or pd.isna(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_scalar(value: Any) -> Any:
        if value is None:
            return None
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @staticmethod
    def _slot_to_track_id(slot_id: str | None, slot_assignments: Mapping[str, str]) -> str | None:
        if slot_id is None:
            return None
        target_slot = str(slot_id)
        for track_id, assigned_slot in slot_assignments.items():
            if str(assigned_slot) == target_slot:
                return str(track_id)
        return None
