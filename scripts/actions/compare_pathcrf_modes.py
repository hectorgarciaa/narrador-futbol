from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

# Permite ejecutar `python3 scripts/actions/compare_pathcrf_modes.py` sin instalar el paquete.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.actions.pathcrf_adapter import PathCRFAdapterConfig, PathCRFTracksAdapter  # noqa: E402
from football_ai.actions.pathcrf_setpieces import classify_episode_starts as classify_episode_starts_legacy  # noqa: E402
from football_ai.actions.pathcrf_shot import apply_shot_heuristic as apply_shot_heuristic_legacy  # noqa: E402
from football_ai.actions.pathcrf_wrapper import (  # noqa: E402
    PathCRFInferenceConfig,
    _import_pathcrf_modules,
    _resolve_repo_path,
    _select_device,
)
from football_ai.actions_incremental import (  # noqa: E402
    ActionsDetector,
    ActionsDetectorConfig,
    ActionsRuntime,
    ActionsRuntimeConfig,
)
from football_ai.core.config import Config  # noqa: E402
from football_ai.core.serialization import convert_to_serializable  # noqa: E402
from football_ai.pathcrf_slot_mapping import (  # noqa: E402
    all_person_slots,
    all_referee_slots,
    person_slot_to_canonical_id,
    referee_slot_to_canonical_id,
)
from football_ai.pipeline.paths import resolve_video_path, sanitize_video_stem  # noqa: E402
from football_ai.positions.data import resolve_tracks_path_for_video  # noqa: E402


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}
TRACK_CLASSES = ("player", "goalkeeper", "referee", "ball")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compara el PathCRF legacy (adaptador offline) frente al flujo incremental actual "
            "y exporta artefactos de depuración para entradas, edges, eventos y checkpoints."
        ),
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=(
            "Shortcut de video de config.yaml, ruta a `<video>_tracks.json` o ruta a un vídeo."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta base de salida. Por defecto: output/actions/pathcrf_compare/<video>/",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="FPS para timestamps y derivadas.")
    parser.add_argument("--trial", type=int, default=120, help="Trial/checkpoint de PathCRF.")
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path("football_ai/actions/repo/pathcrf"),
        help="Ruta al repo clonado de PathCRF.",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="state_dict_best_acc.pt",
        help="Checkpoint dentro de `saved/<trial>/model` o ruta absoluta.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de inferencia (`auto`, `cpu`, `cuda:0`, ...).",
    )
    parser.add_argument(
        "--decode",
        choices=("indep", "greedy", "viterbi"),
        default="indep",
        help="Modo de decodificación si `--no-crf` está activo.",
    )
    parser.add_argument("--no-crf", action="store_true", help="Desactiva el CRF del checkpoint.")
    parser.add_argument(
        "--correct-episode-lasts",
        action="store_true",
        help="Activa la corrección del último nodo por episodio si el checkpoint la soporta.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Activa métricas internas de PathCRF.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Override de la ventana temporal de PathCRF.",
    )
    parser.add_argument(
        "--sample-freq",
        type=int,
        default=None,
        help="Override del sample frequency de PathCRF.",
    )
    parser.add_argument(
        "--min-event-duration",
        type=int,
        default=10,
        help="Duración mínima para compactar self-loops al detectar eventos.",
    )
    parser.add_argument(
        "--cadence-frames",
        type=int,
        default=10,
        help="Cadencia del runtime incremental para lanzar inferencia online.",
    )
    parser.add_argument(
        "--min-frames-warmup",
        type=int,
        default=50,
        help="Warmup mínimo del runtime incremental antes de inferir.",
    )
    parser.add_argument(
        "--confirmation-cooldown-frames",
        type=int,
        default=15,
        help="Cooldown del postproceso incremental entre acciones confirmadas.",
    )
    parser.add_argument(
        "--window-size-frames",
        type=int,
        default=None,
        help="Ventana máxima del detector incremental. Por defecto usa todo el histórico.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Límite opcional de frames para una comparación corta/smoke test.",
    )
    parser.add_argument(
        "--checkpoint-limit",
        type=int,
        default=None,
        help="Límite opcional del número de checkpoints legacy-on-snapshot a exportar.",
    )
    return parser


def _infer_video_for_tracks(tracks_path: Path) -> Path | None:
    config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
    target_stem = sanitize_video_stem(tracks_path.stem.replace("_tracks", ""))
    for _, value in config.paths.get("data", {}).items():
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (config.project_root / candidate).resolve()
        if not candidate.exists() or candidate.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        if sanitize_video_stem(candidate.stem) == target_stem:
            return candidate.resolve()
    return None


def resolve_source(source: str | None) -> tuple[Path, Path | None]:
    if source is None:
        config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
        video_path, _ = resolve_video_path(config, None)
        video_obj = Path(video_path).resolve()
        return resolve_tracks_path_for_video(PROJECT_ROOT, video_obj).resolve(), video_obj

    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()

    if candidate.exists():
        suffix = candidate.suffix.lower()
        if suffix == ".json":
            return candidate, _infer_video_for_tracks(candidate)
        if suffix in VIDEO_SUFFIXES:
            return resolve_tracks_path_for_video(PROJECT_ROOT, candidate).resolve(), candidate.resolve()

    config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
    video_path, _ = resolve_video_path(config, source)
    video_obj = Path(video_path).resolve()
    return resolve_tracks_path_for_video(PROJECT_ROOT, video_obj).resolve(), video_obj


def default_output_dir(tracks_path: Path) -> Path:
    stem = sanitize_video_stem(tracks_path.stem.replace("_tracks", ""))
    return PROJECT_ROOT / "output" / "actions" / "pathcrf_compare" / stem


def write_json(path: Path, payload: Any) -> None:
    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [_json_safe(item) for item in value]
        return convert_to_serializable(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def load_tracks(tracks_path: Path, max_frames: int | None = None) -> dict[str, Any]:
    with tracks_path.open("r", encoding="utf-8") as f:
        tracks = json.load(f)
    if max_frames is None:
        return tracks
    trimmed: dict[str, Any] = {}
    for key, value in tracks.items():
        if isinstance(value, list):
            trimmed[key] = value[:max_frames]
        else:
            trimmed[key] = value
    return trimmed


def build_clean_packet(tracks: Mapping[str, Any], frame_index: int) -> dict[str, Any]:
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


def infer_frame_count(tracks: Mapping[str, Any]) -> int:
    return max((len(tracks.get(class_name, [])) for class_name in TRACK_CLASSES), default=0)


class LegacyPathCRFRunner:
    def __init__(self, config: PathCRFInferenceConfig):
        self.config = config
        self.repo_path = _resolve_repo_path(config.repo_path)
        self.save_path = self.repo_path / "saved" / f"{int(config.trial):03d}"
        if not self.save_path.exists():
            raise FileNotFoundError(f"No existe el trial de PathCRF: {self.save_path}")
        with (self.save_path / "args.json").open("r", encoding="utf-8") as f:
            self.trial_args = json.load(f)
        self.pathcrf_utils, self.pathcrf_inference, self.pathcrf_postprocess = _import_pathcrf_modules(
            self.repo_path,
            self.trial_args,
        )
        self.device = _select_device(config.device)
        self.model = self.pathcrf_utils.build_model(self.trial_args, device=self.device)
        model_path = Path(self.pathcrf_utils.resolve_model_path(str(self.save_path), config.model_file)).resolve()
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model_path = model_path

    def infer(self, tracking_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        effective_fps = (
            float(self.config.fps)
            if self.config.fps is not None
            else float(self.trial_args.get("fps", 25.0))
        )
        effective_sample_freq = (
            int(self.config.sample_freq)
            if self.config.sample_freq is not None
            else int(self.trial_args.get("sample_freq", 5))
        )
        effective_window_seconds = (
            float(self.config.window_seconds)
            if self.config.window_seconds is not None
            else self.trial_args.get("window_seconds")
        )
        _, _, micro_pred_df, stats = self.pathcrf_inference.inference(
            model=self.model,
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
            edge_sequence_df = self.pathcrf_postprocess.edge_probs_to_seq(micro_pred_df)
        events_df = self.pathcrf_postprocess.detect_events(
            tracking=tracking_df,
            edge_seq=edge_sequence_df,
            min_dur=int(self.config.min_event_duration),
        )
        semantic_events_df = classify_episode_starts_legacy(events_df)
        semantic_events_df = apply_shot_heuristic_legacy(semantic_events_df, fps=effective_fps)
        return edge_sequence_df, events_df, semantic_events_df, stats


def build_legacy_tracking(
    tracks: Mapping[str, Any],
    adapter_config: PathCRFAdapterConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter = PathCRFTracksAdapter(config=adapter_config)
    frame_count = adapter._infer_frame_count(tracks)
    if frame_count <= 0:
        raise ValueError("No hay frames para construir el tracking legacy.")
    records = adapter._build_track_records(tracks, frame_count)
    person_assignments, referee_assignments, synthetic_slots = adapter._assign_slots(records)
    tracking_df = adapter._build_tracking_dataframe(
        tracks=tracks,
        frame_count=frame_count,
        records=records,
        person_assignments=person_assignments,
        referee_assignments=referee_assignments,
    )
    summary = {
        "track_count_by_class": {key: len(value) for key, value in records.items()},
        "person_slot_assignments": {raw_id: slot for slot, raw_id in person_assignments.items()},
        "referee_slot_assignments": {raw_id: slot for slot, raw_id in referee_assignments.items()},
        "synthetic_home_slots": list(synthetic_slots["home"]),
        "synthetic_away_slots": list(synthetic_slots["away"]),
        "synthetic_referee_slots": list(synthetic_slots["referee"]),
        "frames": frame_count,
    }
    return tracking_df, summary


def build_incremental_outputs(
    tracks: Mapping[str, Any],
    runtime_config: ActionsRuntimeConfig,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    runtime = ActionsRuntime(runtime_config)
    frame_count = infer_frame_count(tracks)
    checkpoints: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        result = runtime.process_frame(frame_index, build_clean_packet(tracks, frame_index))
        metadata = dict(result.action_metadata)
        if metadata.get("should_infer"):
            checkpoints.append(
                {
                    "frame_id": int(frame_index),
                    "metadata": metadata,
                    "raw_edge": result.raw_edge,
                    "confirmed_action": result.confirmed_action,
                    "summary": asdict(result.summary),
                }
            )
    tracking_df, summary = runtime.detector.build_tracking_dataframe()
    edge_sequence_df, semantic_events_df = runtime._run_pathcrf(tracking_df)
    return tracking_df, asdict(summary), edge_sequence_df, semantic_events_df, checkpoints


def all_slots(expected_players_per_team: int = 11, expected_referees: int = 3) -> list[str]:
    return all_person_slots(expected_players_per_team) + all_referee_slots(expected_referees) + ["ball"]


def build_tracking_diff(
    legacy_df: pd.DataFrame,
    incremental_df: pd.DataFrame,
    slots: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = legacy_df[["frame_id"]].merge(
        incremental_df[["frame_id"]],
        on="frame_id",
        how="outer",
    ).sort_values("frame_id", kind="stable")
    rows: list[dict[str, Any]] = []
    summary_by_slot: dict[str, Any] = {}
    legacy_indexed = legacy_df.set_index("frame_id", drop=False)
    incremental_indexed = incremental_df.set_index("frame_id", drop=False)

    for slot_name in slots:
        slot_rows: list[dict[str, Any]] = []
        for frame_id in merged["frame_id"].tolist():
            legacy_row = legacy_indexed.loc[frame_id] if frame_id in legacy_indexed.index else None
            incremental_row = incremental_indexed.loc[frame_id] if frame_id in incremental_indexed.index else None
            legacy_x = legacy_row.get(f"{slot_name}_x") if legacy_row is not None else np.nan
            legacy_y = legacy_row.get(f"{slot_name}_y") if legacy_row is not None else np.nan
            incremental_x = incremental_row.get(f"{slot_name}_x") if incremental_row is not None else np.nan
            incremental_y = incremental_row.get(f"{slot_name}_y") if incremental_row is not None else np.nan
            distance = np.nan
            if pd.notna(legacy_x) and pd.notna(legacy_y) and pd.notna(incremental_x) and pd.notna(incremental_y):
                distance = float(np.linalg.norm(np.asarray([legacy_x - incremental_x, legacy_y - incremental_y])))
            row = {
                "frame_id": int(frame_id),
                "slot": slot_name,
                "legacy_x": legacy_x,
                "legacy_y": legacy_y,
                "incremental_x": incremental_x,
                "incremental_y": incremental_y,
                "distance_m": distance,
            }
            rows.append(row)
            slot_rows.append(row)
        slot_df = pd.DataFrame(slot_rows)
        valid = slot_df["distance_m"].dropna()
        summary_by_slot[slot_name] = {
            "rows": int(len(slot_df)),
            "valid_rows": int(valid.shape[0]),
            "mean_distance_m": float(valid.mean()) if not valid.empty else None,
            "p95_distance_m": float(valid.quantile(0.95)) if not valid.empty else None,
            "max_distance_m": float(valid.max()) if not valid.empty else None,
        }
    return pd.DataFrame(rows), summary_by_slot


def build_edge_diff(legacy_edges: pd.DataFrame, incremental_edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    legacy = legacy_edges.copy().rename(
        columns={
            "edge_src": "legacy_edge_src",
            "edge_dst": "legacy_edge_dst",
            "edge_team": "legacy_edge_team",
        }
    )
    incremental = incremental_edges.copy().rename(
        columns={
            "edge_src": "incremental_edge_src",
            "edge_dst": "incremental_edge_dst",
            "edge_team": "incremental_edge_team",
        }
    )
    keep_legacy = [column for column in ("frame_id", "legacy_edge_src", "legacy_edge_dst", "legacy_edge_team") if column in legacy.columns]
    keep_incremental = [column for column in ("frame_id", "incremental_edge_src", "incremental_edge_dst", "incremental_edge_team") if column in incremental.columns]
    diff_df = legacy[keep_legacy].merge(
        incremental[keep_incremental],
        on="frame_id",
        how="outer",
    ).sort_values("frame_id", kind="stable")
    for column in ("legacy_edge_src", "legacy_edge_dst", "incremental_edge_src", "incremental_edge_dst"):
        if column not in diff_df.columns:
            diff_df[column] = None
    diff_df["same_edge"] = (
        diff_df["legacy_edge_src"].fillna("").eq(diff_df["incremental_edge_src"].fillna(""))
        & diff_df["legacy_edge_dst"].fillna("").eq(diff_df["incremental_edge_dst"].fillna(""))
    )
    summary = {
        "rows": int(len(diff_df)),
        "matching_rows": int(diff_df["same_edge"].sum()),
        "different_rows": int((~diff_df["same_edge"]).sum()),
    }
    return diff_df, summary


def build_event_diff(legacy_events: pd.DataFrame, incremental_events: pd.DataFrame) -> dict[str, Any]:
    def normalize(df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        result = df.copy()
        if "event_type_semantic" not in result.columns and "event_type" in result.columns:
            result["event_type_semantic"] = result["event_type"]
        columns = [
            column
            for column in ("frame_id", "event_type_semantic", "player_id", "receiver_id")
            if column in result.columns
        ]
        result = result[columns].fillna("")
        return [
            {
                "frame_id": int(row.get("frame_id", -1)),
                "event_type_semantic": str(row.get("event_type_semantic", "")),
                "player_id": str(row.get("player_id", "")),
                "receiver_id": str(row.get("receiver_id", "")),
            }
            for row in result.to_dict(orient="records")
        ]

    legacy_norm = normalize(legacy_events)
    incremental_norm = normalize(incremental_events)
    legacy_keys = {tuple(item.values()) for item in legacy_norm}
    incremental_keys = {tuple(item.values()) for item in incremental_norm}
    return {
        "legacy_rows": int(len(legacy_norm)),
        "incremental_rows": int(len(incremental_norm)),
        "shared_rows": int(len(legacy_keys & incremental_keys)),
        "legacy_only_rows": [dict(zip(("frame_id", "event_type_semantic", "player_id", "receiver_id"), item)) for item in sorted(legacy_keys - incremental_keys)],
        "incremental_only_rows": [dict(zip(("frame_id", "event_type_semantic", "player_id", "receiver_id"), item)) for item in sorted(incremental_keys - legacy_keys)],
        "legacy_type_counts": (
            legacy_events["event_type_semantic"].value_counts(dropna=False).astype(int).to_dict()
            if "event_type_semantic" in legacy_events.columns
            else {}
        ),
        "incremental_type_counts": (
            incremental_events["event_type_semantic"].value_counts(dropna=False).astype(int).to_dict()
            if "event_type_semantic" in incremental_events.columns
            else {}
        ),
    }


def canonical_slot_observations(
    tracks: Mapping[str, Any],
    expected_players_per_team: int,
    expected_referees: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frame_count = infer_frame_count(tracks)
    for frame_index in range(frame_count):
        for slot_name in all_person_slots(expected_players_per_team):
            canonical_id = person_slot_to_canonical_id(slot_name)
            if canonical_id is None:
                continue
            payload = None
            for class_name in ("goalkeeper", "player"):
                frame_map = tracks.get(class_name, [])
                if frame_index < len(frame_map):
                    candidate = frame_map[frame_index]
                    if isinstance(candidate, Mapping):
                        payload = candidate.get(str(canonical_id))
                if payload is not None:
                    break
            if not isinstance(payload, Mapping):
                continue
            if bool(payload.get("synthetic_seed")):
                continue
            field_position = payload.get("field_position_m")
            if not isinstance(field_position, (list, tuple)) or len(field_position) < 2:
                continue
            rows.append(
                {
                    "frame_id": int(frame_index),
                    "slot": slot_name,
                    "observed_x": float(field_position[0]),
                    "observed_y": float(field_position[1]),
                }
            )
        for slot_name in all_referee_slots(expected_referees):
            canonical_id = referee_slot_to_canonical_id(slot_name)
            if canonical_id is None:
                continue
            frame_map = tracks.get("referee", [])
            if frame_index >= len(frame_map) or not isinstance(frame_map[frame_index], Mapping):
                continue
            payload = frame_map[frame_index].get(str(canonical_id))
            if not isinstance(payload, Mapping) or bool(payload.get("synthetic_seed")):
                continue
            field_position = payload.get("field_position_m")
            if not isinstance(field_position, (list, tuple)) or len(field_position) < 2:
                continue
            rows.append(
                {
                    "frame_id": int(frame_index),
                    "slot": slot_name,
                    "observed_x": float(field_position[0]),
                    "observed_y": float(field_position[1]),
                }
            )
    return pd.DataFrame(rows, columns=["frame_id", "slot", "observed_x", "observed_y"])


def build_observation_error(
    tracking_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    empty_df = pd.DataFrame(
        columns=["label", "frame_id", "slot", "observed_x", "observed_y", "tracking_x", "tracking_y", "distance_m"]
    )
    if observations_df.empty:
        return empty_df, {"label": label, "slots": {}}
    merged_rows: list[dict[str, Any]] = []
    summary_by_slot: dict[str, Any] = {}
    indexed = tracking_df.set_index("frame_id", drop=False)
    for slot_name, group in observations_df.groupby("slot", sort=True):
        slot_rows: list[dict[str, Any]] = []
        for obs in group.to_dict(orient="records"):
            frame_id = int(obs["frame_id"])
            if frame_id not in indexed.index:
                continue
            row = indexed.loc[frame_id]
            model_x = row.get(f"{slot_name}_x")
            model_y = row.get(f"{slot_name}_y")
            distance = float(
                np.linalg.norm(
                    np.asarray([float(model_x) - float(obs["observed_x"]), float(model_y) - float(obs["observed_y"])])
                )
            )
            item = {
                "label": label,
                "frame_id": frame_id,
                "slot": slot_name,
                "observed_x": float(obs["observed_x"]),
                "observed_y": float(obs["observed_y"]),
                "tracking_x": float(model_x),
                "tracking_y": float(model_y),
                "distance_m": distance,
            }
            merged_rows.append(item)
            slot_rows.append(item)
        slot_df = pd.DataFrame(slot_rows)
        if slot_df.empty:
            continue
        first_frame = int(slot_df["frame_id"].min())
        early_mask = slot_df["frame_id"] < (first_frame + 100)
        early_dist = slot_df.loc[early_mask, "distance_m"]
        summary_by_slot[slot_name] = {
            "first_observed_frame": first_frame,
            "observed_frames": int(len(slot_df)),
            "mean_distance_m": float(slot_df["distance_m"].mean()),
            "p95_distance_m": float(slot_df["distance_m"].quantile(0.95)),
            "max_distance_m": float(slot_df["distance_m"].max()),
            "first_100_observed_frames_mean_distance_m": float(early_dist.mean()) if not early_dist.empty else None,
        }
    return pd.DataFrame(merged_rows, columns=empty_df.columns), {"label": label, "slots": summary_by_slot}


def last_row_snapshot(df: pd.DataFrame, slots: list[str]) -> dict[str, Any]:
    if df.empty:
        return {}
    row = df.iloc[-1]
    snapshot = {
        "frame_id": int(row["frame_id"]),
        "timestamp": float(row["timestamp"]) if "timestamp" in row else None,
    }
    for slot_name in slots:
        x_key = f"{slot_name}_x"
        y_key = f"{slot_name}_y"
        if x_key in row and y_key in row:
            snapshot[slot_name] = {
                "x": float(row[x_key]) if pd.notna(row[x_key]) else None,
                "y": float(row[y_key]) if pd.notna(row[y_key]) else None,
            }
    return snapshot


def compare_checkpoint_outputs(
    legacy_tracking_df: pd.DataFrame,
    legacy_edge_df: pd.DataFrame,
    legacy_semantic_events_df: pd.DataFrame,
    incremental_tracking_df: pd.DataFrame,
    incremental_checkpoint: Mapping[str, Any],
    slots: list[str],
) -> dict[str, Any]:
    legacy_last_edge = legacy_edge_df.iloc[-1].to_dict() if not legacy_edge_df.empty else None
    legacy_last_event = legacy_semantic_events_df.iloc[-1].to_dict() if not legacy_semantic_events_df.empty else None
    edge_match = (
        isinstance(legacy_last_edge, dict)
        and isinstance(incremental_checkpoint.get("raw_edge"), Mapping)
        and str(legacy_last_edge.get("edge_src")) == str(incremental_checkpoint["raw_edge"].get("edge_src"))
        and str(legacy_last_edge.get("edge_dst")) == str(incremental_checkpoint["raw_edge"].get("edge_dst"))
    )
    event_match = (
        isinstance(legacy_last_event, dict)
        and isinstance(incremental_checkpoint.get("confirmed_action"), Mapping)
        and str(legacy_last_event.get("event_type_semantic") or legacy_last_event.get("event_type"))
        == str(incremental_checkpoint["confirmed_action"].get("event_type"))
    )
    return {
        "frame_id": int(incremental_checkpoint["frame_id"]),
        "legacy_last_edge": legacy_last_edge,
        "incremental_raw_edge": incremental_checkpoint.get("raw_edge"),
        "legacy_last_event": legacy_last_event,
        "incremental_confirmed_action": incremental_checkpoint.get("confirmed_action"),
        "edge_match": bool(edge_match),
        "event_type_match": bool(event_match),
        "legacy_tracking_last_row": last_row_snapshot(legacy_tracking_df, slots),
        "incremental_tracking_last_row": last_row_snapshot(incremental_tracking_df, slots),
    }


def main() -> None:
    args = build_parser().parse_args()

    tracks_path, video_path = resolve_source(args.source)
    if not tracks_path.exists():
        raise FileNotFoundError(f"No existe el tracks JSON esperado: {tracks_path}")

    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_output_dir(tracks_path)
    offline_dir = output_dir / "offline"
    incremental_dir = output_dir / "incremental"
    checkpoints_dir = output_dir / "checkpoints"
    compare_dir = output_dir / "compare"
    for directory in (offline_dir, incremental_dir, checkpoints_dir, compare_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(tracks_path, max_frames=args.max_frames)
    frame_count = infer_frame_count(tracks)
    if frame_count <= 0:
        raise ValueError(f"No se encontraron frames en {tracks_path}")

    adapter_config = PathCRFAdapterConfig(fps=float(args.fps))
    inference_config = PathCRFInferenceConfig(
        repo_path=args.repo_path,
        trial=int(args.trial),
        model_file=str(args.model_file),
        use_crf=not bool(args.no_crf),
        decode=str(args.decode),
        correct_episode_lasts=bool(args.correct_episode_lasts),
        evaluate=bool(args.evaluate),
        window_seconds=float(args.window_seconds) if args.window_seconds is not None else None,
        fps=float(args.fps),
        sample_freq=int(args.sample_freq) if args.sample_freq is not None else None,
        min_event_duration=int(args.min_event_duration),
        device=str(args.device),
    )
    runtime_config = ActionsRuntimeConfig(
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
    slots = all_slots(
        expected_players_per_team=runtime_config.detector.expected_players_per_team,
        expected_referees=runtime_config.detector.expected_referees,
    )

    legacy_runner = LegacyPathCRFRunner(inference_config)

    legacy_tracking_df, legacy_summary = build_legacy_tracking(tracks, adapter_config)
    legacy_edge_df, legacy_events_df, legacy_semantic_df, legacy_stats = legacy_runner.infer(legacy_tracking_df)
    legacy_tracking_path = offline_dir / "tracking.parquet"
    legacy_edge_path = offline_dir / "edge_sequence.parquet"
    legacy_events_path = offline_dir / "events.parquet"
    legacy_semantic_path = offline_dir / "events_semantic.parquet"
    legacy_tracking_df.to_parquet(legacy_tracking_path, index=False)
    legacy_edge_df.to_parquet(legacy_edge_path, index=False)
    legacy_events_df.to_parquet(legacy_events_path, index=False)
    legacy_semantic_df.to_parquet(legacy_semantic_path, index=False)
    write_json(
        offline_dir / "summary.json",
        {
            "tracking_path": str(legacy_tracking_path),
            "edge_sequence_path": str(legacy_edge_path),
            "events_path": str(legacy_events_path),
            "events_semantic_path": str(legacy_semantic_path),
            "conversion_summary": legacy_summary,
            "stats": legacy_stats,
        },
    )

    incremental_tracking_df, incremental_summary, incremental_edge_df, incremental_semantic_df, incremental_checkpoints = build_incremental_outputs(
        tracks,
        runtime_config,
    )
    incremental_tracking_path = incremental_dir / "tracking.parquet"
    incremental_edge_path = incremental_dir / "edge_sequence.parquet"
    incremental_semantic_path = incremental_dir / "events_semantic.parquet"
    incremental_tracking_df.to_parquet(incremental_tracking_path, index=False)
    incremental_edge_df.to_parquet(incremental_edge_path, index=False)
    incremental_semantic_df.to_parquet(incremental_semantic_path, index=False)
    write_json(
        incremental_dir / "summary.json",
        {
            "tracking_path": str(incremental_tracking_path),
            "edge_sequence_path": str(incremental_edge_path),
            "events_semantic_path": str(incremental_semantic_path),
            "conversion_summary": incremental_summary,
            "runtime_checkpoints": incremental_checkpoints,
        },
    )

    checkpoint_detector = ActionsDetector(runtime_config.detector)
    incremental_runtime = ActionsRuntime(runtime_config)
    checkpoint_comparisons: list[dict[str, Any]] = []
    checkpoints_done = 0
    for frame_index in range(frame_count):
        packet = build_clean_packet(tracks, frame_index)
        checkpoint_detector.update(frame_index, packet)
        result = incremental_runtime.process_frame(frame_index, packet)
        if not result.action_metadata.get("should_infer"):
            continue
        if args.checkpoint_limit is not None and checkpoints_done >= int(args.checkpoint_limit):
            break
        snapshot_tracks = checkpoint_detector.build_tracks_snapshot()
        legacy_snapshot_tracking_df, legacy_snapshot_summary = build_legacy_tracking(snapshot_tracks, adapter_config)
        legacy_snapshot_edge_df, legacy_snapshot_events_df, legacy_snapshot_semantic_df, _ = legacy_runner.infer(
            legacy_snapshot_tracking_df
        )
        incremental_snapshot_tracking_df, _ = incremental_runtime.detector.build_tracking_dataframe()
        checkpoint_payload = compare_checkpoint_outputs(
            legacy_tracking_df=legacy_snapshot_tracking_df,
            legacy_edge_df=legacy_snapshot_edge_df,
            legacy_semantic_events_df=legacy_snapshot_semantic_df,
            incremental_tracking_df=incremental_snapshot_tracking_df,
            incremental_checkpoint={
                "frame_id": frame_index,
                "raw_edge": result.raw_edge,
                "confirmed_action": result.confirmed_action,
            },
            slots=slots,
        )
        checkpoint_payload["legacy_snapshot_summary"] = legacy_snapshot_summary
        checkpoint_payload["incremental_metadata"] = result.action_metadata
        checkpoint_frame_dir = checkpoints_dir / f"frame_{frame_index:06d}"
        checkpoint_frame_dir.mkdir(parents=True, exist_ok=True)
        legacy_snapshot_tracking_df.to_parquet(checkpoint_frame_dir / "legacy_tracking.parquet", index=False)
        legacy_snapshot_edge_df.to_parquet(checkpoint_frame_dir / "legacy_edge_sequence.parquet", index=False)
        legacy_snapshot_semantic_df.to_parquet(checkpoint_frame_dir / "legacy_events_semantic.parquet", index=False)
        incremental_snapshot_tracking_df.to_parquet(checkpoint_frame_dir / "incremental_tracking.parquet", index=False)
        write_json(checkpoint_frame_dir / "comparison.json", checkpoint_payload)
        checkpoint_comparisons.append(checkpoint_payload)
        checkpoints_done += 1

    tracking_diff_df, tracking_diff_summary = build_tracking_diff(legacy_tracking_df, incremental_tracking_df, slots)
    tracking_diff_path = compare_dir / "tracking_diff.parquet"
    tracking_diff_df.to_parquet(tracking_diff_path, index=False)

    edge_diff_df, edge_diff_summary = build_edge_diff(legacy_edge_df, incremental_edge_df)
    edge_diff_path = compare_dir / "edge_diff.parquet"
    edge_diff_df.to_parquet(edge_diff_path, index=False)

    event_diff_summary = build_event_diff(legacy_semantic_df, incremental_semantic_df)
    observations_df = canonical_slot_observations(
        tracks,
        expected_players_per_team=runtime_config.detector.expected_players_per_team,
        expected_referees=runtime_config.detector.expected_referees,
    )
    observations_path = compare_dir / "canonical_slot_observations.parquet"
    observations_df.to_parquet(observations_path, index=False)

    legacy_obs_error_df, legacy_obs_error_summary = build_observation_error(legacy_tracking_df, observations_df, "legacy")
    incremental_obs_error_df, incremental_obs_error_summary = build_observation_error(
        incremental_tracking_df,
        observations_df,
        "incremental",
    )
    observation_error_df = pd.concat([legacy_obs_error_df, incremental_obs_error_df], ignore_index=True)
    observation_error_path = compare_dir / "observation_error.parquet"
    observation_error_df.to_parquet(observation_error_path, index=False)

    checkpoint_summary = {
        "checkpoints": int(len(checkpoint_comparisons)),
        "edge_matches": int(sum(1 for item in checkpoint_comparisons if item.get("edge_match"))),
        "event_type_matches": int(sum(1 for item in checkpoint_comparisons if item.get("event_type_match"))),
    }
    top_level_summary = {
        "source_tracks_path": str(tracks_path),
        "video_path": str(video_path) if video_path is not None else None,
        "output_dir": str(output_dir),
        "frame_count": int(frame_count),
        "python_executable": sys.executable,
        "legacy_model_path": str(legacy_runner.model_path),
        "offline": {
            "tracking_path": str(legacy_tracking_path),
            "edge_sequence_path": str(legacy_edge_path),
            "events_semantic_path": str(legacy_semantic_path),
            "summary_path": str(offline_dir / "summary.json"),
        },
        "incremental": {
            "tracking_path": str(incremental_tracking_path),
            "edge_sequence_path": str(incremental_edge_path),
            "events_semantic_path": str(incremental_semantic_path),
            "summary_path": str(incremental_dir / "summary.json"),
        },
        "compare": {
            "tracking_diff_path": str(tracking_diff_path),
            "edge_diff_path": str(edge_diff_path),
            "observation_error_path": str(observation_error_path),
            "canonical_slot_observations_path": str(observations_path),
            "checkpoint_dir": str(checkpoints_dir),
        },
        "tracking_diff_summary_by_slot": tracking_diff_summary,
        "edge_diff_summary": edge_diff_summary,
        "event_diff_summary": event_diff_summary,
        "observation_error_summary": {
            "legacy": legacy_obs_error_summary,
            "incremental": incremental_obs_error_summary,
        },
        "checkpoint_summary": checkpoint_summary,
        "runtime_config": asdict(runtime_config),
        "legacy_inference_config": asdict(inference_config),
    }
    write_json(compare_dir / "event_diff_summary.json", event_diff_summary)
    write_json(compare_dir / "tracking_diff_summary.json", tracking_diff_summary)
    write_json(compare_dir / "observation_error_summary.json", top_level_summary["observation_error_summary"])
    write_json(checkpoints_dir / "summary.json", {"summary": checkpoint_summary, "items": checkpoint_comparisons})
    write_json(output_dir / "summary.json", top_level_summary)

    print(f"Output dir: {output_dir}")
    print(f"Frames analizados: {frame_count}")
    print(f"Offline tracking: {legacy_tracking_path}")
    print(f"Incremental tracking: {incremental_tracking_path}")
    print(f"Tracking diff: {tracking_diff_path}")
    print(f"Edge diff: {edge_diff_path}")
    print(f"Checkpoint summary: {checkpoints_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
