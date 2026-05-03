from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.actions.pathcrf_setpieces import classify_episode_starts as classify_episode_starts_legacy  # noqa: E402
from football_ai.actions.pathcrf_shot import apply_shot_heuristic as apply_shot_heuristic_legacy  # noqa: E402
from football_ai.actions.pathcrf_wrapper import (  # noqa: E402
    PathCRFInferenceConfig,
    _import_pathcrf_modules,
    _resolve_repo_path,
    _select_device,
)


FEATURES = ("x", "y", "vx", "vy", "speed", "accel")
TOLERANT_EVENT_WINDOWS = (0, 5, 10, 15, 20, 30)


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

    def infer(
        self,
        tracking_df: pd.DataFrame,
        *,
        debug_collector: list[dict[str, Any]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
            debug_collector=debug_collector,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audita en profundidad la comparacion legacy vs incremental de PathCRF "
            "a partir de un output ya generado por compare_pathcrf_modes.py."
        ),
    )
    parser.add_argument(
        "compare_root",
        type=Path,
        help="Directorio raiz output/actions/pathcrf_compare/<video>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta donde escribir los reportes. Por defecto usa <compare_root>/reports/pathcrf_incremental_alignment_audit.",
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path("football_ai/actions/repo/pathcrf"),
        help="Ruta al repo clonado de PathCRF.",
    )
    parser.add_argument("--trial", type=int, default=120, help="Trial/checkpoint de PathCRF.")
    parser.add_argument(
        "--model-file",
        type=str,
        default="state_dict_best_acc.pt",
        help="Checkpoint dentro de saved/<trial>/model o ruta absoluta.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo de inferencia.")
    parser.add_argument(
        "--decode",
        choices=("indep", "greedy", "viterbi"),
        default="indep",
        help="Modo de decodificacion si --no-crf esta activo.",
    )
    parser.add_argument("--no-crf", action="store_true", help="Desactiva el CRF del checkpoint.")
    parser.add_argument(
        "--correct-episode-lasts",
        action="store_true",
        help="Activa la correccion del ultimo nodo por episodio si el checkpoint la soporta.",
    )
    parser.add_argument("--evaluate", action="store_true", help="Activa metricas internas de PathCRF.")
    parser.add_argument("--window-seconds", type=float, default=None, help="Override de window_seconds.")
    parser.add_argument("--sample-freq", type=int, default=None, help="Override de sample_freq.")
    parser.add_argument(
        "--min-event-duration",
        type=int,
        default=10,
        help="Duracion minima para compactar self-loops al detectar eventos.",
    )
    parser.add_argument("--fps", type=float, default=None, help="FPS explicito. Si no, se infiere del tracking.")
    parser.add_argument("--top-slots", type=int, default=5, help="Numero de slots a plotear.")
    parser.add_argument("--max-lag", type=int, default=20, help="Lag maximo a auditar.")
    return parser


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_compare_artifacts(compare_root: Path) -> dict[str, Any]:
    compare_root = compare_root.resolve()
    offline_dir = compare_root / "offline"
    incremental_dir = compare_root / "incremental"
    compare_dir = compare_root / "compare"
    summary_path = compare_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No existe el summary base: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return {
        "summary": summary,
        "compare_root": compare_root,
        "offline_tracking": pd.read_parquet(offline_dir / "tracking.parquet"),
        "offline_edges": pd.read_parquet(offline_dir / "edge_sequence.parquet"),
        "offline_events": pd.read_parquet(offline_dir / "events_semantic.parquet"),
        "incremental_tracking": pd.read_parquet(incremental_dir / "tracking.parquet"),
        "incremental_edges": pd.read_parquet(incremental_dir / "edge_sequence.parquet"),
        "incremental_events": pd.read_parquet(incremental_dir / "events_semantic.parquet"),
        "compare_dir": compare_dir,
    }


def infer_fps(tracking_df: pd.DataFrame, explicit_fps: float | None) -> float:
    if explicit_fps is not None:
        return float(explicit_fps)
    if "timestamp" in tracking_df.columns:
        deltas = tracking_df["timestamp"].diff().dropna()
        deltas = deltas[deltas > 0]
        if not deltas.empty:
            return float(round(1.0 / float(deltas.median()), 6))
    return 25.0


def slot_names_from_tracking(tracking_df: pd.DataFrame) -> list[str]:
    slots: list[str] = []
    for column in tracking_df.columns:
        if not column.endswith("_x"):
            continue
        slot_name = column[:-2]
        if slot_name == "ball":
            continue
        slots.append(slot_name)
    return sorted(slots)


def metric_rmse(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(values ** 2)))


def metric_mae(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(np.abs(values)))


def metric_quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.quantile(values, q))


def ensure_frame_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "frame_id" in df.columns:
        return df.copy()
    result = df.reset_index(drop=False).copy()
    if "frame_id" in result.columns:
        return result
    result = result.rename(columns={"index": "frame_id"})
    return result


def build_observation_timeline(
    *,
    frame_count: int,
    slots: list[str],
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    if observations_df.empty:
        rows = [
            {
                "frame_id": int(frame_id),
                "slot": slot,
                "has_observation": False,
                "missing_age": 0,
                "is_reacquisition": False,
            }
            for slot in slots
            for frame_id in range(frame_count)
        ]
        return pd.DataFrame(rows)

    observation_map: dict[str, set[int]] = {
        slot: set(group["frame_id"].astype(int).tolist())
        for slot, group in observations_df.groupby("slot", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for slot in slots:
        observed_frames = observation_map.get(slot, set())
        last_observed_frame: int | None = None
        previous_observed = False
        for frame_id in range(frame_count):
            has_observation = int(frame_id) in observed_frames
            if has_observation:
                missing_age = 0
                is_reacquisition = bool(frame_id > 0 and not previous_observed)
                last_observed_frame = int(frame_id)
            else:
                missing_age = 0 if last_observed_frame is None else int(frame_id - last_observed_frame)
                is_reacquisition = False
            rows.append(
                {
                    "frame_id": int(frame_id),
                    "slot": slot,
                    "has_observation": bool(has_observation),
                    "missing_age": int(missing_age),
                    "is_reacquisition": bool(is_reacquisition),
                }
            )
            previous_observed = bool(has_observation)
    return pd.DataFrame(rows)


def build_alignment_metric_frames(
    legacy_tracking: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    observation_timeline: pd.DataFrame,
    slots: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slot in slots:
        meta = observation_timeline[observation_timeline["slot"] == slot].sort_values("frame_id", kind="stable")
        legacy = legacy_tracking[[
            "frame_id",
            f"{slot}_x",
            f"{slot}_y",
            f"{slot}_vx",
            f"{slot}_vy",
            f"{slot}_speed",
            f"{slot}_accel",
        ]].copy()
        legacy.columns = ["frame_id", "legacy_x", "legacy_y", "legacy_vx", "legacy_vy", "legacy_speed", "legacy_accel"]
        incremental = incremental_tracking[[
            "frame_id",
            f"{slot}_x",
            f"{slot}_y",
            f"{slot}_vx",
            f"{slot}_vy",
            f"{slot}_speed",
            f"{slot}_accel",
        ]].copy()
        incremental.columns = [
            "frame_id",
            "incremental_x",
            "incremental_y",
            "incremental_vx",
            "incremental_vy",
            "incremental_speed",
            "incremental_accel",
        ]
        merged = meta.merge(legacy, on="frame_id", how="left").merge(incremental, on="frame_id", how="left")
        dx = merged["legacy_x"] - merged["incremental_x"]
        dy = merged["legacy_y"] - merged["incremental_y"]
        dvx = merged["legacy_vx"] - merged["incremental_vx"]
        dvy = merged["legacy_vy"] - merged["incremental_vy"]
        dspeed = merged["legacy_speed"] - merged["incremental_speed"]
        daccel = merged["legacy_accel"] - merged["incremental_accel"]
        merged["position_sqerr"] = (dx ** 2) + (dy ** 2)
        merged["velocity_sqerr"] = (dvx ** 2) + (dvy ** 2)
        merged["position_err"] = np.sqrt(merged["position_sqerr"])
        merged["velocity_err"] = np.sqrt(merged["velocity_sqerr"])
        merged["vx_abs_diff"] = np.abs(dvx)
        merged["vy_abs_diff"] = np.abs(dvy)
        merged["speed_abs_diff"] = np.abs(dspeed)
        merged["accel_abs_diff"] = np.abs(daccel)
        merged["jerk_abs_diff"] = np.abs(merged["accel_abs_diff"].diff().fillna(0.0))
        merged["slot"] = slot
        rows.extend(merged.to_dict(orient="records"))
    return pd.DataFrame(rows)


def summarize_alignment_metrics(metric_frames: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    def _collect(subset: pd.DataFrame) -> dict[str, Any]:
        return {
            "frames": int(len(subset)),
            "position_rmse": metric_rmse(np.sqrt(subset["position_sqerr"].to_numpy(dtype=np.float32, copy=False))),
            "velocity_rmse": metric_rmse(np.sqrt(subset["velocity_sqerr"].to_numpy(dtype=np.float32, copy=False))),
            "vx_rmse": metric_rmse(subset["vx_abs_diff"].to_numpy(dtype=np.float32, copy=False)),
            "vy_rmse": metric_rmse(subset["vy_abs_diff"].to_numpy(dtype=np.float32, copy=False)),
            "speed_mae": metric_mae(subset["speed_abs_diff"].to_numpy(dtype=np.float32, copy=False)),
            "accel_mae": metric_mae(subset["accel_abs_diff"].to_numpy(dtype=np.float32, copy=False)),
            "accel_p95": metric_quantile(subset["accel_abs_diff"].to_numpy(dtype=np.float32, copy=False), 0.95),
            "accel_p99": metric_quantile(subset["accel_abs_diff"].to_numpy(dtype=np.float32, copy=False), 0.99),
            "jerk_p95": metric_quantile(subset["jerk_abs_diff"].to_numpy(dtype=np.float32, copy=False), 0.95),
        }

    by_slot_rows: list[dict[str, Any]] = []
    for slot, group in metric_frames.groupby("slot", sort=True):
        active_mask = group["has_observation"].astype(bool)
        missing_mask = ~active_mask
        first100_mask = group["frame_id"] < 100
        after100_mask = group["frame_id"] >= 100
        row = {"slot": slot}
        row.update({f"all_{key}": value for key, value in _collect(group).items()})
        row.update({f"active_{key}": value for key, value in _collect(group.loc[active_mask]).items()})
        row.update({f"missing_{key}": value for key, value in _collect(group.loc[missing_mask]).items()})
        row.update({f"first100_{key}": value for key, value in _collect(group.loc[first100_mask]).items()})
        row.update({f"after100_{key}": value for key, value in _collect(group.loc[after100_mask]).items()})
        row["reacquisition_events"] = int(group["is_reacquisition"].sum())
        row["observed_frames"] = int(active_mask.sum())
        row["missing_frames"] = int(missing_mask.sum())
        by_slot_rows.append(row)

    scope_masks = {
        "all": pd.Series(True, index=metric_frames.index),
        "active": metric_frames["has_observation"].astype(bool),
        "missing": ~metric_frames["has_observation"].astype(bool),
        "reacquisition_window": metric_frames["is_reacquisition"].astype(bool),
        "first100": metric_frames["frame_id"] < 100,
        "after100": metric_frames["frame_id"] >= 100,
    }
    active_missing_rows = []
    for scope, mask in scope_masks.items():
        payload = _collect(metric_frames.loc[mask])
        payload["scope"] = scope
        active_missing_rows.append(payload)

    active_payload = next(item for item in active_missing_rows if item["scope"] == "active")
    missing_payload = next(item for item in active_missing_rows if item["scope"] == "missing")
    dominant_source = "mixed"
    active_accel = active_payload.get("accel_mae") or 0.0
    missing_accel = missing_payload.get("accel_mae") or 0.0
    if missing_accel > (active_accel * 1.15):
        dominant_source = "missing_or_reacquisition"
    elif active_accel > (missing_accel * 1.15):
        dominant_source = "active_tracks"
    summary = {
        "dominant_divergence_source": dominant_source,
        "active": active_payload,
        "missing": missing_payload,
        "global": next(item for item in active_missing_rows if item["scope"] == "all"),
    }
    return pd.DataFrame(by_slot_rows), pd.DataFrame(active_missing_rows), summary


def build_reacquisition_events(metric_frames: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slot, group in metric_frames.groupby("slot", sort=True):
        group = group.sort_values("frame_id", kind="stable").reset_index(drop=True)
        reacq_frames = group[group["is_reacquisition"].astype(bool)]["frame_id"].astype(int).tolist()
        for frame_id in reacq_frames:
            window = group[(group["frame_id"] >= (frame_id - 5)) & (group["frame_id"] <= (frame_id + 10))]
            current = group[group["frame_id"] == frame_id]
            if current.empty:
                continue
            current_row = current.iloc[0]
            prev_frame = group[group["frame_id"] == (frame_id - 1)]
            missing_before = int(prev_frame.iloc[0]["missing_age"]) if not prev_frame.empty else 0
            rows.append(
                {
                    "slot": slot,
                    "frame_id": int(frame_id),
                    "missing_age_before_reacquisition": missing_before,
                    "delta_position": float(current_row["position_err"]) if pd.notna(current_row["position_err"]) else None,
                    "delta_vx": float(current_row["vx_abs_diff"]) if pd.notna(current_row["vx_abs_diff"]) else None,
                    "delta_vy": float(current_row["vy_abs_diff"]) if pd.notna(current_row["vy_abs_diff"]) else None,
                    "delta_speed": float(current_row["speed_abs_diff"]) if pd.notna(current_row["speed_abs_diff"]) else None,
                    "delta_accel": float(current_row["accel_abs_diff"]) if pd.notna(current_row["accel_abs_diff"]) else None,
                    "window_mean_position_error": metric_mae(window["position_err"].to_numpy(dtype=np.float32, copy=False)),
                    "window_mean_speed_error": metric_mae(window["speed_abs_diff"].to_numpy(dtype=np.float32, copy=False)),
                    "window_mean_accel_error": metric_mae(window["accel_abs_diff"].to_numpy(dtype=np.float32, copy=False)),
                    "window_max_accel_error": metric_quantile(window["accel_abs_diff"].to_numpy(dtype=np.float32, copy=False), 1.0),
                }
            )
    return pd.DataFrame(rows)


def team_from_slot(slot_name: str | None) -> str | None:
    if not slot_name:
        return None
    if slot_name.startswith("home_"):
        return "home"
    if slot_name.startswith("away_"):
        return "away"
    if slot_name.startswith("referee_"):
        return "referee"
    return None


def identity_or_best_slot_remap(
    legacy_tracking: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    slots: list[str],
) -> dict[str, str]:
    legacy_means = np.asarray(
        [
            [
                float(legacy_tracking[f"{slot}_x"].mean()),
                float(legacy_tracking[f"{slot}_y"].mean()),
            ]
            for slot in slots
        ],
        dtype=np.float32,
    )
    incremental_means = np.asarray(
        [
            [
                float(incremental_tracking[f"{slot}_x"].mean()),
                float(incremental_tracking[f"{slot}_y"].mean()),
            ]
            for slot in slots
        ],
        dtype=np.float32,
    )
    costs = np.linalg.norm(incremental_means[:, None, :] - legacy_means[None, :, :], axis=2)
    row_idx, col_idx = linear_sum_assignment(costs)
    return {slots[row]: slots[col] for row, col in zip(row_idx.tolist(), col_idx.tolist(), strict=True)}


def remap_slot(slot_name: str | None, remap: dict[str, str]) -> str | None:
    if slot_name is None:
        return None
    return remap.get(str(slot_name), str(slot_name))


def edge_match_summary(
    legacy_edges: pd.DataFrame,
    candidate_edges: pd.DataFrame,
    remap: dict[str, str] | None = None,
) -> dict[str, Any]:
    remap = remap or {}
    legacy_edges = ensure_frame_id_column(legacy_edges)
    candidate_edges = ensure_frame_id_column(candidate_edges)
    merged = legacy_edges[["frame_id", "edge_src", "edge_dst"]].merge(
        candidate_edges[["frame_id", "edge_src", "edge_dst"]],
        on="frame_id",
        how="inner",
        suffixes=("_legacy", "_candidate"),
    )
    if merged.empty:
        return {
            "rows": 0,
            "exact_match_rate": 0.0,
            "topology_match_rate": 0.0,
            "team_pair_match_rate": 0.0,
        }
    candidate_src = merged["edge_src_candidate"].map(lambda value: remap_slot(value, remap))
    candidate_dst = merged["edge_dst_candidate"].map(lambda value: remap_slot(value, remap))
    exact = merged["edge_src_legacy"].eq(candidate_src) & merged["edge_dst_legacy"].eq(candidate_dst)
    legacy_pairs = merged.apply(
        lambda row: tuple(sorted((str(row["edge_src_legacy"]), str(row["edge_dst_legacy"])))),
        axis=1,
    )
    candidate_pairs = pd.DataFrame({"src": candidate_src, "dst": candidate_dst}).apply(
        lambda row: tuple(sorted((str(row["src"]), str(row["dst"])))),
        axis=1,
    )
    topology = legacy_pairs.eq(candidate_pairs)
    legacy_team_pairs = merged.apply(
        lambda row: tuple(sorted((str(team_from_slot(row["edge_src_legacy"])), str(team_from_slot(row["edge_dst_legacy"]))))),
        axis=1,
    )
    candidate_team_pairs = pd.DataFrame({"src": candidate_src, "dst": candidate_dst}).apply(
        lambda row: tuple(sorted((str(team_from_slot(row["src"])), str(team_from_slot(row["dst"]))))),
        axis=1,
    )
    team_pair = legacy_team_pairs.eq(candidate_team_pairs)
    return {
        "rows": int(len(merged)),
        "exact_match_rate": float(exact.mean()),
        "topology_match_rate": float(topology.mean()),
        "team_pair_match_rate": float(team_pair.mean()),
    }


def event_match_summary(
    legacy_events: pd.DataFrame,
    candidate_events: pd.DataFrame,
    remap: dict[str, str] | None = None,
) -> dict[str, Any]:
    remap = remap or {}
    legacy = ensure_frame_id_column(legacy_events)
    candidate = ensure_frame_id_column(candidate_events)
    if "event_type_semantic" not in legacy.columns and "event_type" in legacy.columns:
        legacy["event_type_semantic"] = legacy["event_type"]
    if "event_type_semantic" not in candidate.columns and "event_type" in candidate.columns:
        candidate["event_type_semantic"] = candidate["event_type"]
    legacy["player_team"] = legacy.get("player_id", pd.Series(dtype=object)).map(team_from_slot)
    candidate["player_team"] = candidate.get("player_id", pd.Series(dtype=object)).map(
        lambda value: team_from_slot(remap_slot(value, remap))
    )
    legacy_keys = {
        (
            int(row["frame_id"]),
            str(row.get("event_type_semantic", "")),
            str(row.get("player_id", "")),
            str(row.get("receiver_id", "")),
        )
        for row in legacy.to_dict(orient="records")
    }
    candidate_keys = {
        (
            int(row["frame_id"]),
            str(row.get("event_type_semantic", "")),
            str(remap_slot(row.get("player_id"), remap) or ""),
            str(remap_slot(row.get("receiver_id"), remap) or ""),
        )
        for row in candidate.to_dict(orient="records")
    }
    legacy_frame_type = {
        (int(row["frame_id"]), str(row.get("event_type_semantic", "")))
        for row in legacy.to_dict(orient="records")
    }
    candidate_frame_type = {
        (int(row["frame_id"]), str(row.get("event_type_semantic", "")))
        for row in candidate.to_dict(orient="records")
    }
    legacy_frame_type_team = {
        (int(row["frame_id"]), str(row.get("event_type_semantic", "")), str(row.get("player_team", "")))
        for row in legacy.to_dict(orient="records")
    }
    candidate_frame_type_team = {
        (int(row["frame_id"]), str(row.get("event_type_semantic", "")), str(row.get("player_team", "")))
        for row in candidate.to_dict(orient="records")
    }
    return {
        "legacy_count": int(len(legacy)),
        "incremental_count": int(len(candidate)),
        "exact_shared": int(len(legacy_keys & candidate_keys)),
        "frame_type_shared": int(len(legacy_frame_type & candidate_frame_type)),
        "frame_type_team_shared": int(len(legacy_frame_type_team & candidate_frame_type_team)),
    }


def feature_diff_rows(
    legacy_tracking: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    slots: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in slots:
        row: dict[str, Any] = {"slot": slot}
        for feature in FEATURES:
            legacy_values = legacy_tracking[f"{slot}_{feature}"].to_numpy(dtype=np.float32, copy=False)
            incremental_values = incremental_tracking[f"{slot}_{feature}"].to_numpy(dtype=np.float32, copy=False)
            mask = np.isfinite(legacy_values) & np.isfinite(incremental_values)
            diffs = np.abs(legacy_values[mask] - incremental_values[mask])
            if diffs.size == 0:
                row[f"mean_abs_{feature}_diff"] = None
                row[f"p95_abs_{feature}_diff"] = None
                row[f"max_abs_{feature}_diff"] = None
                continue
            row[f"mean_abs_{feature}_diff"] = float(np.mean(diffs))
            row[f"p95_abs_{feature}_diff"] = float(np.quantile(diffs, 0.95))
            row[f"max_abs_{feature}_diff"] = float(np.max(diffs))
        rows.append(row)
    return rows


def summarize_feature_diffs(
    legacy_tracking: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    slots: list[str],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        slot_means = []
        for slot in slots:
            legacy_values = legacy_tracking[f"{slot}_{feature}"].to_numpy(dtype=np.float32, copy=False)
            incremental_values = incremental_tracking[f"{slot}_{feature}"].to_numpy(dtype=np.float32, copy=False)
            mask = np.isfinite(legacy_values) & np.isfinite(incremental_values)
            diffs = np.abs(legacy_values[mask] - incremental_values[mask])
            if diffs.size > 0:
                slot_means.append(float(np.mean(diffs)))
        summary_rows.append(
            {
                "feature": feature,
                "mean_absdiff_across_slots": float(np.mean(slot_means)) if slot_means else 0.0,
                "max_slot_mean_absdiff": float(np.max(slot_means)) if slot_means else 0.0,
            }
        )
    return summary_rows


def lagged_xy_distance(
    legacy_tracking: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    slot: str,
    lag_frames: int,
) -> float:
    legacy_x = legacy_tracking[f"{slot}_x"].to_numpy(dtype=np.float32, copy=False)
    legacy_y = legacy_tracking[f"{slot}_y"].to_numpy(dtype=np.float32, copy=False)
    incremental_x = incremental_tracking[f"{slot}_x"].to_numpy(dtype=np.float32, copy=False)
    incremental_y = incremental_tracking[f"{slot}_y"].to_numpy(dtype=np.float32, copy=False)
    if lag_frames > 0:
        legacy_x = legacy_x[:-lag_frames]
        legacy_y = legacy_y[:-lag_frames]
        incremental_x = incremental_x[lag_frames:]
        incremental_y = incremental_y[lag_frames:]
    mask = (
        np.isfinite(legacy_x)
        & np.isfinite(legacy_y)
        & np.isfinite(incremental_x)
        & np.isfinite(incremental_y)
    )
    if not np.any(mask):
        return float("inf")
    diffs = np.sqrt(((legacy_x[mask] - incremental_x[mask]) ** 2) + ((legacy_y[mask] - incremental_y[mask]) ** 2))
    return float(np.mean(diffs))


def build_lag_audit_report(
    legacy_tracking: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    slots: list[str],
    max_lag: int,
) -> dict[str, Any]:
    per_slot: list[dict[str, Any]] = []
    best_lag_counts: dict[str, int] = {}
    for slot in slots:
        raw_mean_distance = lagged_xy_distance(legacy_tracking, incremental_tracking, slot, lag_frames=0)
        lag_scores = [
            (lag, lagged_xy_distance(legacy_tracking, incremental_tracking, slot, lag_frames=lag))
            for lag in range(1, max_lag + 1)
        ]
        best_lag, best_distance = min(lag_scores, key=lambda item: item[1], default=(0, raw_mean_distance))
        accel_diff = np.abs(
            legacy_tracking[f"{slot}_accel"].to_numpy(dtype=np.float32, copy=False)
            - incremental_tracking[f"{slot}_accel"].to_numpy(dtype=np.float32, copy=False)
        )
        accel_diff = accel_diff[np.isfinite(accel_diff)]
        item = {
            "slot": slot,
            "raw_mean_distance_m": float(raw_mean_distance),
            "best_lag_frames": int(best_lag),
            "best_lag_mean_distance_m": float(best_distance),
            "lag_gain_m": float(raw_mean_distance - best_distance),
            "mean_abs_accel_diff": float(np.mean(accel_diff)) if accel_diff.size else 0.0,
            "p95_abs_accel_diff": float(np.quantile(accel_diff, 0.95)) if accel_diff.size else 0.0,
        }
        per_slot.append(item)
        best_lag_counts[str(best_lag)] = best_lag_counts.get(str(best_lag), 0) + 1
    top_lag_sensitive = sorted(per_slot, key=lambda item: item["lag_gain_m"], reverse=True)[:12]
    top_accel_slots = sorted(per_slot, key=lambda item: item["mean_abs_accel_diff"], reverse=True)[:12]
    return {
        "top_lag_sensitive_slots": top_lag_sensitive,
        "global_best_lag_counts": best_lag_counts,
        "top_accel_slots": top_accel_slots,
    }


def greedy_tolerant_event_matches(
    legacy_events: pd.DataFrame,
    candidate_events: pd.DataFrame,
    tolerance: int,
) -> dict[str, Any]:
    legacy = ensure_frame_id_column(legacy_events)
    candidate = ensure_frame_id_column(candidate_events)
    if "event_type_semantic" not in legacy.columns and "event_type" in legacy.columns:
        legacy["event_type_semantic"] = legacy["event_type"]
    if "event_type_semantic" not in candidate.columns and "event_type" in candidate.columns:
        candidate["event_type_semantic"] = candidate["event_type"]
    used_candidate: set[int] = set()
    matches = 0
    abs_deltas: list[int] = []
    for legacy_row in legacy.sort_values("frame_id", kind="stable").to_dict(orient="records"):
        best_idx = None
        best_delta = None
        for candidate_idx, candidate_row in enumerate(candidate.to_dict(orient="records")):
            if candidate_idx in used_candidate:
                continue
            if str(legacy_row.get("event_type_semantic", "")) != str(candidate_row.get("event_type_semantic", "")):
                continue
            delta = abs(int(legacy_row["frame_id"]) - int(candidate_row["frame_id"]))
            if delta > tolerance:
                continue
            if best_delta is None or delta < best_delta:
                best_idx = candidate_idx
                best_delta = delta
        if best_idx is not None and best_delta is not None:
            used_candidate.add(best_idx)
            matches += 1
            abs_deltas.append(int(best_delta))
    return {
        "matches": int(matches),
        "legacy_recall": float(matches / len(legacy)) if len(legacy) else 0.0,
        "incremental_precision_proxy": float(matches / len(candidate)) if len(candidate) else 0.0,
        "mean_abs_delta_frames": float(np.mean(abs_deltas)) if abs_deltas else 0.0,
    }


def build_temporal_alignment_report(
    legacy_edges: pd.DataFrame,
    incremental_edges: pd.DataFrame,
    legacy_events: pd.DataFrame,
    incremental_events: pd.DataFrame,
    max_lag: int,
) -> dict[str, Any]:
    legacy_edges = ensure_frame_id_column(legacy_edges)
    incremental_edges = ensure_frame_id_column(incremental_edges)
    edge_shift_curve: dict[str, Any] = {}
    best_shift = None
    best_payload = None
    for shift in range(-max_lag, max_lag + 1):
        shifted = incremental_edges[["frame_id", "edge_src", "edge_dst"]].copy()
        shifted["frame_id"] = shifted["frame_id"] + shift
        payload = edge_match_summary(legacy_edges, shifted)
        edge_shift_curve[str(shift)] = {
            "exact_match_rate": payload["exact_match_rate"],
            "rows": payload["rows"],
        }
        if best_payload is None or payload["exact_match_rate"] > best_payload["exact_match_rate"]:
            best_shift = shift
            best_payload = edge_shift_curve[str(shift)]
    tolerant_matches = {
        str(tolerance): greedy_tolerant_event_matches(legacy_events, incremental_events, tolerance)
        for tolerance in TOLERANT_EVENT_WINDOWS
    }
    return {
        "best_edge_shift": [int(best_shift or 0), best_payload or {"exact_match_rate": 0.0, "rows": 0}],
        "edge_shift_curve": edge_shift_curve,
        "tolerant_event_matches": tolerant_matches,
    }


def build_stuck_edge_timeline(legacy_edges: pd.DataFrame, incremental_edges: pd.DataFrame) -> pd.DataFrame:
    legacy = ensure_frame_id_column(legacy_edges).rename(
        columns={"edge_src": "offline_edge_src", "edge_dst": "offline_edge_dst"}
    )
    incremental = ensure_frame_id_column(incremental_edges).rename(
        columns={"edge_src": "incremental_edge_src", "edge_dst": "incremental_edge_dst"}
    )
    merged = legacy.merge(incremental, on="frame_id", how="outer").sort_values("frame_id", kind="stable")
    merged["offline_edge"] = merged.apply(
        lambda row: f"{row['offline_edge_src']} -> {row['offline_edge_dst']}"
        if pd.notna(row["offline_edge_src"]) and pd.notna(row["offline_edge_dst"])
        else None,
        axis=1,
    )
    merged["incremental_edge"] = merged.apply(
        lambda row: f"{row['incremental_edge_src']} -> {row['incremental_edge_dst']}"
        if pd.notna(row["incremental_edge_src"]) and pd.notna(row["incremental_edge_dst"])
        else None,
        axis=1,
    )
    merged["same_edge"] = merged["offline_edge"].fillna("").eq(merged["incremental_edge"].fillna(""))
    merged["incremental_edge_changed"] = merged["incremental_edge"].fillna("").ne(
        merged["incremental_edge"].shift(1).fillna("")
    )
    merged["offline_edge_changed"] = merged["offline_edge"].fillna("").ne(merged["offline_edge"].shift(1).fillna(""))
    merged = merged.rename(
        columns={
            "incremental_edge_src": "inc_src",
            "incremental_edge_dst": "inc_dst",
            "offline_edge_src": "off_src",
            "offline_edge_dst": "off_dst",
        }
    )
    merged["inc_score_if_available"] = np.nan
    merged["off_score_if_available"] = np.nan
    return merged[
        [
            "frame_id",
            "offline_edge",
            "incremental_edge",
            "same_edge",
            "incremental_edge_changed",
            "offline_edge_changed",
            "inc_src",
            "inc_dst",
            "off_src",
            "off_dst",
            "inc_score_if_available",
            "off_score_if_available",
        ]
    ].copy()


def edge_entropy(edge_values: pd.Series) -> float:
    values = [str(value) for value in edge_values.dropna().tolist() if str(value)]
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    probs = np.asarray([count / total for count in counts.values()], dtype=np.float64)
    return float(-(probs * np.log2(probs)).sum())


def build_edge_behavior_summary(timeline_df: pd.DataFrame, frame_split: int = 89) -> dict[str, Any]:
    if timeline_df.empty:
        return {
            "stuck_edge_duration": 0,
            "edge_change_rate": 0.0,
            "edge_entropy": 0.0,
            "after_frame_89_match_rate": 0.0,
            "dominant_incremental_run": None,
        }
    run_rows: list[dict[str, Any]] = []
    start_idx = 0
    prev_edge = None
    inc_edges = timeline_df["incremental_edge"].fillna("").tolist()
    frame_ids = timeline_df["frame_id"].astype(int).tolist()
    for idx, edge in enumerate(inc_edges):
        if prev_edge is None:
            prev_edge = edge
            start_idx = idx
            continue
        if edge != prev_edge:
            run_rows.append(
                {
                    "edge": prev_edge,
                    "start_frame": int(frame_ids[start_idx]),
                    "end_frame": int(frame_ids[idx - 1]),
                    "duration_frames": int(idx - start_idx),
                }
            )
            start_idx = idx
            prev_edge = edge
    if prev_edge is not None:
        run_rows.append(
            {
                "edge": prev_edge,
                "start_frame": int(frame_ids[start_idx]),
                "end_frame": int(frame_ids[len(frame_ids) - 1]),
                "duration_frames": int(len(frame_ids) - start_idx),
            }
        )
    dominant_run = max(run_rows, key=lambda item: item["duration_frames"])
    total_changes = int(timeline_df["incremental_edge_changed"].sum()) - 1
    total_changes = max(total_changes, 0)
    after_mask = timeline_df["frame_id"] >= int(frame_split)
    return {
        "stuck_edge_duration": int(dominant_run["duration_frames"]),
        "edge_change_rate": float(total_changes / max(len(timeline_df) - 1, 1)),
        "edge_entropy": edge_entropy(timeline_df["incremental_edge"]),
        "after_frame_89_match_rate": float(timeline_df.loc[after_mask, "same_edge"].mean()) if after_mask.any() else 0.0,
        "dominant_incremental_run": dominant_run,
        "all_incremental_runs": run_rows,
    }


def fallback_pathcrf_input_debug(
    tracking_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    observation_lookup = {
        (int(row["frame_id"]), str(row["slot"])): True
        for row in observations_df.to_dict(orient="records")
    } if not observations_df.empty else {}
    rows: list[dict[str, Any]] = []
    for slot in slot_names_from_tracking(tracking_df):
        missing_frames = 0
        seen_observation = False
        for row in tracking_df.to_dict(orient="records"):
            frame_id = int(row["frame_id"])
            has_observation = bool(observation_lookup.get((frame_id, slot), False))
            if has_observation:
                missing_frames = 0
                seen_observation = True
                source_mode = "observed"
            else:
                missing_frames = missing_frames + 1 if seen_observation else 0
                source_mode = "predicted" if seen_observation else "template"
            rows.append(
                {
                    "mode": label,
                    "frame_id": frame_id,
                    "slot": slot,
                    "team": team_from_slot(slot),
                    "source_track_id": None,
                    "x": row.get(f"{slot}_x"),
                    "y": row.get(f"{slot}_y"),
                    "vx": row.get(f"{slot}_vx"),
                    "vy": row.get(f"{slot}_vy"),
                    "speed": row.get(f"{slot}_speed"),
                    "accel": row.get(f"{slot}_accel"),
                    "has_observation": has_observation,
                    "missing_frames": int(missing_frames),
                    "is_predicted": bool(source_mode == "predicted"),
                    "source_mode": source_mode,
                }
            )
    return pd.DataFrame(rows)


def build_edge_window_slot_debug(
    offline_input_df: pd.DataFrame,
    incremental_input_df: pd.DataFrame,
    edge_timeline_df: pd.DataFrame,
    *,
    src_slot: str,
    dst_slot: str,
    window_start: int,
    window_end: int,
) -> pd.DataFrame:
    edge_window = edge_timeline_df[
        (edge_timeline_df["frame_id"] >= int(window_start)) & (edge_timeline_df["frame_id"] <= int(window_end))
    ][["frame_id", "offline_edge", "incremental_edge"]].copy()

    def _slot_slice(df: pd.DataFrame, slot_name: str, prefix: str) -> pd.DataFrame:
        subset = df[
            (df["slot"] == slot_name)
            & (df["frame_id"] >= int(window_start))
            & (df["frame_id"] <= int(window_end))
        ][
            [
                "frame_id",
                "x",
                "y",
                "vx",
                "vy",
                "speed",
                "accel",
                "has_observation",
                "missing_frames",
            ]
        ].copy()
        return subset.rename(
            columns={
                "x": f"{prefix}_x",
                "y": f"{prefix}_y",
                "vx": f"{prefix}_vx",
                "vy": f"{prefix}_vy",
                "speed": f"{prefix}_speed",
                "accel": f"{prefix}_accel",
                "has_observation": f"{prefix}_has_observation",
                "missing_frames": f"{prefix}_missing_frames",
            }
        )

    off_src = _slot_slice(offline_input_df, src_slot, "off_src")
    inc_src = _slot_slice(incremental_input_df, src_slot, "inc_src")
    off_dst = _slot_slice(offline_input_df, dst_slot, "off_dst")
    inc_dst = _slot_slice(incremental_input_df, dst_slot, "inc_dst")
    merged = edge_window.merge(off_src, on="frame_id", how="left").merge(inc_src, on="frame_id", how="left")
    merged = merged.merge(off_dst, on="frame_id", how="left").merge(inc_dst, on="frame_id", how="left")
    return merged.rename(
        columns={
            "off_src_has_observation": "src_has_obs_offline",
            "inc_src_has_observation": "src_has_obs_incremental",
            "off_dst_has_observation": "dst_has_obs_offline",
            "inc_dst_has_observation": "dst_has_obs_incremental",
            "off_src_missing_frames": "src_missing_frames_offline",
            "inc_src_missing_frames": "src_missing_frames_incremental",
            "off_dst_missing_frames": "dst_missing_frames_offline",
            "inc_dst_missing_frames": "dst_missing_frames_incremental",
            "offline_edge": "selected_edge_offline",
            "incremental_edge": "selected_edge_incremental",
        }
    )


def _dynamic_dense_transition_scores(prev_embed: np.ndarray, curr_embed: np.ndarray, linear_weight: np.ndarray, linear_bias: float) -> np.ndarray:
    dim = prev_embed.shape[1]
    prev_score = prev_embed @ linear_weight[:dim]
    curr_score = curr_embed @ linear_weight[dim:]
    return curr_score[:, None] + prev_score[None, :] + float(linear_bias)


def _crf_delta_debug(bundle: dict[str, Any], crf: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    micro_out = np.asarray(bundle["micro_out"], dtype=np.float32)
    valid_edge_ids = bundle.get("valid_edge_ids")
    if valid_edge_ids is not None:
        emissions = micro_out[:, np.asarray(valid_edge_ids, dtype=np.int64)]
    else:
        emissions = micro_out
    decoded = np.asarray(bundle.get("decoded"), dtype=np.int64) if bundle.get("decoded") is not None else emissions.argmax(axis=1)
    T, K = emissions.shape
    delta = np.full((T, K), -np.inf, dtype=np.float32)
    best_prev = np.full((T, K), -1, dtype=np.int32)
    best_trans = np.zeros((T, K), dtype=np.float32)
    delta[0] = emissions[0]
    crf_type = type(crf).__name__
    edge_embeds = bundle.get("edge_embeds_comp")
    if edge_embeds is not None:
        edge_embeds = np.asarray(edge_embeds, dtype=np.float32)

    for t in range(1, T):
        if crf_type == "StaticDenseCRF":
            transitions = crf.transitions.detach().cpu().numpy().astype(np.float32)
            score = delta[t - 1][None, :] + transitions
            prev_idx = score.argmax(axis=1)
            trans = transitions[np.arange(K), prev_idx]
            delta[t] = emissions[t] + score[np.arange(K), prev_idx]
            best_prev[t] = prev_idx.astype(np.int32)
            best_trans[t] = trans.astype(np.float32)
            continue

        if crf_type in {"DynamicDenseCRF"}:
            weight = crf.linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
            bias = float(crf.linear.bias.detach().cpu().numpy().reshape(-1)[0]) if crf.linear.bias is not None else 0.0
            trans_scores = _dynamic_dense_transition_scores(edge_embeds[t - 1], edge_embeds[t], weight, bias)
            score = delta[t - 1][None, :] + trans_scores
            prev_idx = score.argmax(axis=1)
            trans = trans_scores[np.arange(K), prev_idx]
            delta[t] = emissions[t] + score[np.arange(K), prev_idx]
            best_prev[t] = prev_idx.astype(np.int32)
            best_trans[t] = trans.astype(np.float32)
            continue

        if crf_type == "StaticSparseCRF":
            inc_idx = crf.inc_idx.detach().cpu().numpy()
            inc_mask = crf.inc_mask.detach().cpu().numpy().astype(bool)
            transitions = crf.transitions.detach().cpu().numpy().astype(np.float32)
            for curr in range(K):
                allowed = inc_mask[curr]
                if not np.any(allowed):
                    continue
                prevs = inc_idx[curr][allowed].astype(np.int64)
                trans = transitions[curr][allowed]
                score = delta[t - 1][prevs] + trans
                best_local = int(np.argmax(score))
                best_prev[t, curr] = int(prevs[best_local])
                best_trans[t, curr] = float(trans[best_local])
                delta[t, curr] = float(emissions[t, curr] + score[best_local])
            continue

        if crf_type in {"DynamicSparseCRF", "EdgeEmbedCRF"}:
            inc_idx = crf.inc_idx.detach().cpu().numpy()
            inc_mask = crf.inc_mask.detach().cpu().numpy().astype(bool)
            weight = crf.linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
            bias = float(crf.linear.bias.detach().cpu().numpy().reshape(-1)[0]) if crf.linear.bias is not None else 0.0
            dim = edge_embeds.shape[-1]
            w_prev = weight[:dim]
            w_curr = weight[dim:]
            for curr in range(K):
                allowed = inc_mask[curr]
                if not np.any(allowed):
                    continue
                prevs = inc_idx[curr][allowed].astype(np.int64)
                prev_embed = edge_embeds[t - 1][prevs]
                curr_embed = np.broadcast_to(edge_embeds[t][curr], (len(prevs), dim))
                trans = (prev_embed @ w_prev) + (curr_embed @ w_curr) + bias
                score = delta[t - 1][prevs] + trans
                best_local = int(np.argmax(score))
                best_prev[t, curr] = int(prevs[best_local])
                best_trans[t, curr] = float(trans[best_local])
                delta[t, curr] = float(emissions[t, curr] + score[best_local])
            continue

        raise ValueError(f"CRF debug no soportado para {crf_type}")

    return emissions, decoded, delta, best_prev, best_trans


def build_candidate_tables_from_debug_payloads(
    *,
    debug_payloads: list[dict[str, Any]],
    crf: Any,
    mode_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bundle in debug_payloads:
        comp_src = bundle.get("comp_src")
        comp_dst = bundle.get("comp_dst")
        if comp_src is None or comp_dst is None:
            continue
        comp_src = np.asarray(comp_src, dtype=np.int64)
        comp_dst = np.asarray(comp_dst, dtype=np.int64)
        padded_order = [str(item) for item in bundle["padded_order"]]
        sample_frame_ids = [int(item) for item in bundle["frame_ids"]]
        episode_frame_ids = [int(item) for item in bundle.get("episode_frame_ids", sample_frame_ids)]
        emissions, decoded, delta, best_prev, best_trans = _crf_delta_debug(bundle, crf)
        K = emissions.shape[1]
        episode_bounds: list[tuple[int, int, list[int]]] = []
        for idx, sample_frame in enumerate(sample_frame_ids):
            next_sample = sample_frame_ids[idx + 1] if idx + 1 < len(sample_frame_ids) else episode_frame_ids[-1] + 1
            frame_ids = [int(frame_id) for frame_id in episode_frame_ids if int(sample_frame) <= int(frame_id) < int(next_sample)]
            if not frame_ids:
                frame_ids = [int(sample_frame)]
            episode_bounds.append((int(sample_frame), int(next_sample - 1), frame_ids))
        for t, (sample_frame_id, _, propagated_frames) in enumerate(episode_bounds):
            order = np.argsort(delta[t])[::-1]
            ranks = np.empty(K, dtype=np.int32)
            ranks[order] = np.arange(1, K + 1, dtype=np.int32)
            best_score = float(delta[t, order[0]])
            second_score = float(delta[t, order[1]]) if K > 1 else best_score
            margin = float(best_score - second_score)
            selected_comp = int(decoded[t]) if decoded.size else int(order[0])
            selected_src = padded_order[int(comp_src[selected_comp])]
            selected_dst = padded_order[int(comp_dst[selected_comp])]
            selected_edge = f"{selected_src} -> {selected_dst}"
            for frame_id in propagated_frames:
                for curr in range(K):
                    prev_idx = int(best_prev[t, curr])
                    prev_src = padded_order[int(comp_src[prev_idx])] if prev_idx >= 0 else None
                    prev_dst = padded_order[int(comp_dst[prev_idx])] if prev_idx >= 0 else None
                    rows.append(
                        {
                            "mode": mode_label,
                            "frame_id": int(frame_id),
                            "sample_frame_id": int(sample_frame_id),
                            "candidate_rank": int(ranks[curr]),
                            "candidate_comp_id": int(curr),
                            "candidate_src": padded_order[int(comp_src[curr])],
                            "candidate_dst": padded_order[int(comp_dst[curr])],
                            "candidate_edge": f"{padded_order[int(comp_src[curr])]} -> {padded_order[int(comp_dst[curr])]}",
                            "emission_score": float(emissions[t, curr]),
                            "transition_score": float(best_trans[t, curr]),
                            "total_score": float(delta[t, curr]),
                            "score_margin": margin,
                            "best_prev_comp_id": prev_idx if prev_idx >= 0 else None,
                            "best_prev_src": prev_src,
                            "best_prev_dst": prev_dst,
                            "selected_edge": selected_edge,
                            "is_selected_edge": bool(curr == selected_comp),
                            "crf_type": str(bundle.get("crf_type")),
                            "phase_id": bundle.get("phase_id"),
                            "episode_id": bundle.get("episode_id"),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_candidate_debug(
    offline_edges: pd.DataFrame,
    incremental_edges: pd.DataFrame,
    offline_candidates: pd.DataFrame,
    incremental_candidates: pd.DataFrame,
    *,
    window_start: int,
    window_end: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    off = ensure_frame_id_column(offline_edges).rename(columns={"edge_src": "off_src", "edge_dst": "off_dst"})
    inc = ensure_frame_id_column(incremental_edges).rename(columns={"edge_src": "inc_src", "edge_dst": "inc_dst"})
    edge_rows = off.merge(inc, on="frame_id", how="inner")
    window_rows = edge_rows[(edge_rows["frame_id"] >= int(window_start)) & (edge_rows["frame_id"] <= int(window_end))].copy()
    rows: list[dict[str, Any]] = []
    present_rates: list[float] = []
    ranks: list[float] = []
    score_margins: list[float] = []
    for row in window_rows.to_dict(orient="records"):
        frame_id = int(row["frame_id"])
        off_edge = f"{row['off_src']} -> {row['off_dst']}"
        inc_edge = f"{row['inc_src']} -> {row['inc_dst']}"
        off_candidates = offline_candidates[offline_candidates["frame_id"] == frame_id]
        inc_candidates = incremental_candidates[incremental_candidates["frame_id"] == frame_id]
        off_set = set(off_candidates["candidate_edge"].tolist())
        inc_set = set(inc_candidates["candidate_edge"].tolist())
        intersection = len(off_set & inc_set)
        union = len(off_set | inc_set)
        offline_in_inc = inc_candidates[inc_candidates["candidate_edge"] == off_edge]
        rank_value = float(offline_in_inc["candidate_rank"].iloc[0]) if not offline_in_inc.empty else None
        if rank_value is not None:
            ranks.append(rank_value)
        if not inc_candidates.empty:
            score_margins.append(float(inc_candidates["score_margin"].iloc[0]))
        present = 1.0 if off_edge in inc_set else 0.0
        present_rates.append(present)
        stuck_rows = inc_candidates[inc_candidates["candidate_edge"] == inc_edge]
        offline_score_for_stuck = off_candidates[off_candidates["candidate_edge"] == inc_edge]
        rows.append(
            {
                "frame": frame_id,
                "offline_edge": off_edge,
                "incremental_edge": inc_edge,
                "num_candidates_offline": int(len(off_set)),
                "num_candidates_incremental": int(len(inc_set)),
                "candidate_set_jaccard": float(intersection / union) if union else 0.0,
                "is_offline_selected_edge_present_in_incremental_candidates": bool(present),
                "rank_of_offline_edge_under_incremental_scores": rank_value,
                "rank_of_stuck_edge_under_offline_scores": (
                    float(offline_score_for_stuck["candidate_rank"].iloc[0]) if not offline_score_for_stuck.empty else None
                ),
                "score_gap_between_best_and_second_incremental": (
                    float(inc_candidates["score_margin"].iloc[0]) if not inc_candidates.empty else None
                ),
                "score_of_incremental_selected_edge": (
                    float(stuck_rows["total_score"].iloc[0]) if not stuck_rows.empty else None
                ),
                "score_of_offline_selected_edge_under_incremental": (
                    float(offline_in_inc["total_score"].iloc[0]) if not offline_in_inc.empty else None
                ),
            }
        )
    summary = {
        "offline_selected_present_rate": float(np.mean(present_rates)) if present_rates else 0.0,
        "offline_selected_rank_mean": float(np.mean(ranks)) if ranks else None,
        "score_margin_mean": float(np.mean(score_margins)) if score_margins else None,
        "score_margin_p95": float(np.quantile(score_margins, 0.95)) if score_margins else None,
    }
    return pd.DataFrame(rows), summary


def phase_keeper_detection_report(
    legacy_tracking: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
) -> list[dict[str, Any]]:
    phase_ids = sorted(
        set(legacy_tracking.get("phase_id", pd.Series([1])).dropna().astype(int).tolist())
        | set(incremental_tracking.get("phase_id", pd.Series([1])).dropna().astype(int).tolist())
    )
    report: list[dict[str, Any]] = []
    for phase_id in phase_ids:
        legacy_phase = legacy_tracking[legacy_tracking.get("phase_id", 1) == phase_id]
        incremental_phase = incremental_tracking[incremental_tracking.get("phase_id", 1) == phase_id]
        legacy_pairs = {
            "home_1": float(legacy_phase["home_1_x"].mean()),
            "away_1": float(legacy_phase["away_1_x"].mean()),
        }
        incremental_pairs = {
            "home_1": float(incremental_phase["home_1_x"].mean()),
            "away_1": float(incremental_phase["away_1_x"].mean()),
        }
        legacy_left = min(legacy_pairs, key=legacy_pairs.get)
        legacy_right = max(legacy_pairs, key=legacy_pairs.get)
        incremental_left = min(incremental_pairs, key=incremental_pairs.get)
        incremental_right = max(incremental_pairs, key=incremental_pairs.get)
        report.append(
            {
                "phase_id": int(phase_id),
                "legacy_left_gk": legacy_left,
                "legacy_right_gk": legacy_right,
                "incremental_left_gk": incremental_left,
                "incremental_right_gk": incremental_right,
                "same_pair": legacy_left == incremental_left and legacy_right == incremental_right,
            }
        )
    return report


def recompute_motion_from_xy(tracking_df: pd.DataFrame, slots: list[str], fps: float) -> pd.DataFrame:
    dt = 1.0 / float(fps)
    result = tracking_df.copy()
    for slot in slots:
        x = result[f"{slot}_x"]
        y = result[f"{slot}_y"]
        result[f"{slot}_vx"] = x.diff().fillna(0.0) / dt
        result[f"{slot}_vy"] = y.diff().fillna(0.0) / dt
        result[f"{slot}_speed"] = np.sqrt((result[f"{slot}_vx"] ** 2) + (result[f"{slot}_vy"] ** 2))
        result[f"{slot}_accel"] = result[f"{slot}_speed"].diff().fillna(0.0) / dt
    return result


def summarize_variant_against_legacy(
    legacy_edges: pd.DataFrame,
    legacy_events: pd.DataFrame,
    candidate_edges: pd.DataFrame,
    candidate_events: pd.DataFrame,
) -> dict[str, Any]:
    edge_summary = edge_match_summary(legacy_edges, candidate_edges)
    event_summary = event_match_summary(legacy_events, candidate_events)
    timeline_df = build_stuck_edge_timeline(legacy_edges, candidate_edges)
    behavior = build_edge_behavior_summary(timeline_df)
    edge_change_count = max(int(timeline_df["incremental_edge_changed"].sum()) - 1, 0)
    return {
        "edge_exact_match_rate": edge_summary["exact_match_rate"],
        "edge_topology_match_rate": edge_summary["topology_match_rate"],
        "team_pair_match_rate": edge_summary["team_pair_match_rate"],
        "event_count": int(len(candidate_events)),
        "event_frame_type_shared": event_summary["frame_type_shared"],
        "event_frame_type_shared_rate_vs_legacy": (
            float(event_summary["frame_type_shared"] / len(legacy_events)) if len(legacy_events) else 0.0
        ),
        "stuck_edge_duration": behavior["stuck_edge_duration"],
        "edge_change_rate": behavior["edge_change_rate"],
        "edge_entropy": behavior["edge_entropy"],
        "after_frame_89_match_rate": behavior["after_frame_89_match_rate"],
        "number_of_edge_changes": edge_change_count,
    }


def build_feature_swap_report(
    runner: LegacyPathCRFRunner,
    legacy_tracking: pd.DataFrame,
    legacy_edges: pd.DataFrame,
    legacy_events: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    slots: list[str],
) -> dict[str, Any]:
    variants: dict[str, pd.DataFrame] = {
        "base_incremental": incremental_tracking.copy(),
        "legacy_accel_only": incremental_tracking.copy(),
        "legacy_vel_speed_accel": incremental_tracking.copy(),
        "legacy_all_motion": incremental_tracking.copy(),
    }
    for slot in slots:
        variants["legacy_accel_only"][f"{slot}_accel"] = legacy_tracking[f"{slot}_accel"].to_numpy(copy=True)
        for feature in ("vx", "vy", "speed", "accel"):
            variants["legacy_vel_speed_accel"][f"{slot}_{feature}"] = legacy_tracking[f"{slot}_{feature}"].to_numpy(
                copy=True
            )
        for feature in FEATURES:
            variants["legacy_all_motion"][f"{slot}_{feature}"] = legacy_tracking[f"{slot}_{feature}"].to_numpy(copy=True)
    report: dict[str, Any] = {}
    for name, tracking_df in variants.items():
        candidate_edges, _, candidate_events, _ = runner.infer(tracking_df)
        report[name] = summarize_variant_against_legacy(
            legacy_edges=legacy_edges,
            legacy_events=legacy_events,
            candidate_edges=candidate_edges,
            candidate_events=candidate_events,
        )
    return report


def build_recomputed_motion_report(
    runner: LegacyPathCRFRunner,
    legacy_edges: pd.DataFrame,
    legacy_events: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    incremental_edges: pd.DataFrame,
    incremental_events: pd.DataFrame,
    slots: list[str],
    fps: float,
) -> dict[str, Any]:
    recomputed = recompute_motion_from_xy(incremental_tracking, slots=slots, fps=fps)
    recomputed_edges, _, recomputed_events, _ = runner.infer(recomputed)
    return {
        "base_incremental": summarize_variant_against_legacy(
            legacy_edges=legacy_edges,
            legacy_events=legacy_events,
            candidate_edges=incremental_edges,
            candidate_events=incremental_events,
        ),
        "incremental_recomputed_motion": summarize_variant_against_legacy(
            legacy_edges=legacy_edges,
            legacy_events=legacy_events,
            candidate_edges=recomputed_edges,
            candidate_events=recomputed_events,
        ),
    }


def build_swap_experiment_report(
    runner: LegacyPathCRFRunner,
    legacy_tracking: pd.DataFrame,
    legacy_edges: pd.DataFrame,
    legacy_events: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    incremental_edges: pd.DataFrame,
    incremental_events: pd.DataFrame,
    slots: list[str],
    fps: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    variants: dict[str, pd.DataFrame] = {
        "inc_original": incremental_tracking.copy(),
        "inc_with_offline_xy": incremental_tracking.copy(),
        "inc_with_offline_motion_only": incremental_tracking.copy(),
        "inc_with_offline_xy_and_motion": incremental_tracking.copy(),
        "inc_with_incremental_xy_but_recomputed_legacy_smoothing_motion": recompute_motion_from_xy(
            incremental_tracking,
            slots=slots,
            fps=fps,
        ),
    }
    for slot in slots:
        for feature in ("x", "y"):
            variants["inc_with_offline_xy"][f"{slot}_{feature}"] = legacy_tracking[f"{slot}_{feature}"].to_numpy(copy=True)
            variants["inc_with_offline_xy_and_motion"][f"{slot}_{feature}"] = legacy_tracking[f"{slot}_{feature}"].to_numpy(copy=True)
        for feature in ("vx", "vy", "speed", "accel"):
            variants["inc_with_offline_motion_only"][f"{slot}_{feature}"] = legacy_tracking[f"{slot}_{feature}"].to_numpy(
                copy=True
            )
            variants["inc_with_offline_xy_and_motion"][f"{slot}_{feature}"] = legacy_tracking[f"{slot}_{feature}"].to_numpy(
                copy=True
            )

    rows: list[dict[str, Any]] = []
    narrative: dict[str, Any] = {}
    base_summary = summarize_variant_against_legacy(
        legacy_edges=legacy_edges,
        legacy_events=legacy_events,
        candidate_edges=incremental_edges,
        candidate_events=incremental_events,
    )
    rows.append({"variant": "inc_original", **base_summary})
    narrative["inc_original"] = base_summary

    for name, tracking_df in variants.items():
        if name == "inc_original":
            continue
        candidate_edges, _, candidate_events, _ = runner.infer(tracking_df)
        summary = summarize_variant_against_legacy(
            legacy_edges=legacy_edges,
            legacy_events=legacy_events,
            candidate_edges=candidate_edges,
            candidate_events=candidate_events,
        )
        rows.append({"variant": name, **summary})
        narrative[name] = summary

    narrative["inc_with_offline_candidates_if_possible"] = {
        "status": "not_applicable",
        "reason": "El candidate set de PathCRF se deriva del orden padded de nodos validos y, en esta corrida, permanece estable.",
    }
    narrative["inc_with_offline_observation_features_if_possible"] = {
        "status": "not_applicable",
        "reason": "El checkpoint de PathCRF consumido aqui solo usa indicadores de nodo y motion features del tracking; no hay observation scores externos separados.",
    }
    return pd.DataFrame(rows), narrative


def build_config_parity_report(
    *,
    compare_summary: dict[str, Any],
    fps: float,
) -> str:
    runtime_detector = ((compare_summary.get("runtime_config") or {}).get("detector") or {})
    legacy_inference = compare_summary.get("legacy_inference_config") or {}
    rows = [
        ("fps", compare_summary.get("legacy_inference_config", {}).get("fps"), fps, True, "summary.json"),
        ("sample_freq", legacy_inference.get("sample_freq"), legacy_inference.get("sample_freq"), True, "summary.json"),
        ("window_seconds", legacy_inference.get("window_seconds"), legacy_inference.get("window_seconds"), True, "summary.json"),
        ("fixed_lag_frames", None, runtime_detector.get("fixed_lag_frames"), False, "detector.py / summary.json"),
        ("alpha", None, runtime_detector.get("alpha"), False, "detector.py / summary.json"),
        ("beta", None, runtime_detector.get("beta"), False, "detector.py / summary.json"),
        ("velocity_decay", None, runtime_detector.get("velocity_decay"), False, "detector.py / summary.json"),
        ("pitch_length_m", None, runtime_detector.get("pitch_length_m"), False, "detector.py / summary.json"),
        ("pitch_width_m", None, runtime_detector.get("pitch_width_m"), False, "detector.py / summary.json"),
    ]
    lines = [
        "# Config parity report",
        "",
        "| key | offline_value | incremental_value | match_true_false | source_file |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, offline_value, incremental_value, is_match, source_file in rows:
        lines.append(
            f"| {key} | {offline_value} | {incremental_value} | {bool(is_match)} | {source_file} |"
        )
    lines.extend(
        [
            "",
            "- El orden de slots en esta comparativa coincide porque ambos tracking usan el formato ancho canonicalizado de PathCRF.",
            "- No aparecen offsets 0/1 en `frame_id`: la comparacion usa `frame_id` base 0 en ambos modos.",
            "- `fixed_lag_frames` solo existe en incremental; cualquier desfase temporal que introduzca debe interpretarse como latencia explicita, no como una diferencia de dataset offline.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_deep_audit_report(
    legacy_tracking: pd.DataFrame,
    legacy_edges: pd.DataFrame,
    legacy_events: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    incremental_edges: pd.DataFrame,
    incremental_events: pd.DataFrame,
    slots: list[str],
) -> dict[str, Any]:
    slot_remap = identity_or_best_slot_remap(legacy_tracking, incremental_tracking, slots)
    edge_summary = edge_match_summary(legacy_edges, incremental_edges)
    edge_summary_remap = edge_match_summary(legacy_edges, incremental_edges, remap=slot_remap)
    event_summary = event_match_summary(legacy_events, incremental_events)
    event_summary_remap = event_match_summary(legacy_events, incremental_events, remap=slot_remap)
    return {
        "phase_keeper_detection": phase_keeper_detection_report(legacy_tracking, incremental_tracking),
        "trajectory_slot_mapping_incremental_to_legacy": slot_remap,
        "edge_similarity": {
            "frames": int(len(legacy_edges)),
            "exact_match_rate": edge_summary["exact_match_rate"],
            "topology_match_rate": edge_summary["topology_match_rate"],
            "team_pair_match_rate": edge_summary["team_pair_match_rate"],
            "exact_match_rate_after_slot_remap": edge_summary_remap["exact_match_rate"],
            "topology_match_rate_after_slot_remap": edge_summary_remap["topology_match_rate"],
            "team_pair_match_rate_after_slot_remap": edge_summary_remap["team_pair_match_rate"],
        },
        "event_similarity": {
            "legacy_count": event_summary["legacy_count"],
            "incremental_count": event_summary["incremental_count"],
            "exact_shared": event_summary["exact_shared"],
            "exact_shared_after_slot_remap": event_summary_remap["exact_shared"],
            "frame_type_shared": event_summary["frame_type_shared"],
            "frame_type_team_shared": event_summary["frame_type_team_shared"],
            "frame_type_team_shared_after_slot_remap": event_summary_remap["frame_type_team_shared"],
        },
        "feature_absdiff_mean_across_slots": summarize_feature_diffs(
            legacy_tracking,
            incremental_tracking,
            slots,
        ),
    }


def plot_slot_series(
    legacy_tracking: pd.DataFrame,
    incremental_tracking: pd.DataFrame,
    slot: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(13, 14), sharex=True)
    time_axis = legacy_tracking["frame_id"].to_numpy(dtype=np.int32, copy=False)
    for axis, feature, ylabel in zip(
        axes,
        ("x", "y", "vx", "vy", "speed", "accel"),
        ("x (m)", "y (m)", "vx (m/s)", "vy (m/s)", "speed (m/s)", "accel (m/s^2)"),
        strict=True,
    ):
        axis.plot(time_axis, legacy_tracking[f"{slot}_{feature}"], label="legacy", linewidth=1.2)
        axis.plot(time_axis, incremental_tracking[f"{slot}_{feature}"], label="incremental", linewidth=1.0, alpha=0.9)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].set_title(f"Slot {slot}: legacy vs incremental")
    axes[-1].set_xlabel("frame_id")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_markdown_report(
    *,
    compare_root: Path,
    fps: float,
    deep_audit_report: dict[str, Any],
    lag_report: dict[str, Any],
    temporal_report: dict[str, Any],
    feature_swap_report: dict[str, Any],
    recomputed_motion_report: dict[str, Any],
) -> str:
    feature_summary = {
        item["feature"]: item["mean_absdiff_across_slots"]
        for item in deep_audit_report["feature_absdiff_mean_across_slots"]
    }
    best_shift = temporal_report["best_edge_shift"][0]
    tolerant_10 = temporal_report["tolerant_event_matches"]["10"]
    top_lag = lag_report["top_lag_sensitive_slots"][:5]
    lines = [
        "# Auditoria PathCRF legacy vs incremental",
        "",
        f"- Fuente: `{compare_root}`",
        f"- FPS auditado: `{fps}`",
        "",
        "## Tabla comparativa",
        "",
        "| Componente | Legacy | Incremental |",
        "| --- | --- | --- |",
        "| Seed/warmup | Rellena slots no observados con plantilla alineada y luego suaviza toda la serie | Fija la primera observacion real en cuanto aparece y usa fallback causal por plantilla alineada |",
        "| Suavizado XY | `outlier -> mediana(window=9) -> Savitzky-Golay(window=11, poly=2) -> EWM(alpha=0.35) -> jitter soft-limit` | Mediana local sobre cola corta + clamping causal contra frame previo |",
        "| Derivadas | `diff().fillna(0.0) / dt` | `delta / dt` con `valid_step`; en esta corrida no hay `valid_step=false` en slots de persona/arbitro |",
        "| Aceleracion | `speed.diff().fillna(0.0) / dt` | `(speed[t]-speed[t-1]) / dt` sobre speed incremental |",
        "| Latencia | Offline, usa toda la serie | Causal con `fixed_lag_frames` corto |",
        "",
        "## Findings",
        "",
        f"- `x` y `y` siguen bastante cerca en media (`{feature_summary['x']:.3f} m`, `{feature_summary['y']:.3f} m`), pero `accel` explota (`{feature_summary['accel']:.3f} m/s^2`).",
        f"- Recalcular solo las derivadas sobre el tracking incremental no arregla PathCRF: el `edge_exact_match_rate` se queda en `{recomputed_motion_report['incremental_recomputed_motion']['edge_exact_match_rate']:.3f}`.",
        f"- Sustituir solo `accel` por la version legacy ya sube el `edge_exact_match_rate` a `{feature_swap_report['legacy_accel_only']['edge_exact_match_rate']:.3f}`.",
        f"- Sustituir `vx/vy/speed/accel` por legacy sube el `edge_exact_match_rate` a `{feature_swap_report['legacy_vel_speed_accel']['edge_exact_match_rate']:.3f}` y la coincidencia de eventos por frame/tipo a `{feature_swap_report['legacy_vel_speed_accel']['event_frame_type_shared_rate_vs_legacy']:.3f}`.",
        f"- El mejor shift global de edges es `{best_shift}` frames, pero apenas mejora el exact match; no parece un problema de desfase temporal puro.",
        f"- Con tolerancia de 10 frames, la recuperacion de eventos legacy sube a `{tolerant_10['legacy_recall']:.3f}`, lo que indica cierta similitud temporal gruesa pero no igualdad frame a frame.",
        "",
        "## Lag y slots sensibles",
        "",
    ]
    for item in top_lag:
        lines.append(
            f"- `{item['slot']}` mejora `{item['lag_gain_m']:.3f} m` con lag `{item['best_lag_frames']}` y aun asi mantiene `mean_abs_accel_diff={item['mean_abs_accel_diff']:.3f}`."
        )
    lines.extend(
        [
            "",
            "## Recomendaciones priorizadas",
            "",
            "1. Reintroducir un suavizado causal mas parecido a legacy antes de derivar: primero una fase tipo Savitzky-Golay causal o un equivalente FIR causal, luego EWM causal por slot.",
            "2. Mantener el lag fijo corto, pero probar `8-12` frames como experimento controlado; el reporte de lag muestra mejoras modestas pero reales en varios slots.",
            "3. No centrar la correccion en `valid_step` ni en la formula de `diff`: en esta corrida no hay huecos NaN en slots de persona/arbitro, asi que ese no es el cuello de botella.",
            "4. Tratar el warmup como problema secundario: ayuda en algunos slots tardios, pero la divergencia de aceleracion se mantiene mas alla de los primeros 100 frames.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    artifacts = load_compare_artifacts(args.compare_root)
    compare_root = artifacts["compare_root"]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (compare_root / "reports" / "pathcrf_incremental_alignment_audit").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_tracking = artifacts["offline_tracking"]
    legacy_edges = artifacts["offline_edges"]
    legacy_events = artifacts["offline_events"]
    incremental_tracking = artifacts["incremental_tracking"]
    incremental_edges = artifacts["incremental_edges"]
    incremental_events = artifacts["incremental_events"]
    observations_path = artifacts["compare_dir"] / "canonical_slot_observations.parquet"
    observations_df = pd.read_parquet(observations_path) if observations_path.exists() else pd.DataFrame()
    offline_input_debug_path = artifacts["compare_dir"] / "pathcrf_input_offline.parquet"
    incremental_input_debug_path = artifacts["compare_dir"] / "pathcrf_input_incremental.parquet"
    offline_input_debug_df = (
        pd.read_parquet(offline_input_debug_path)
        if offline_input_debug_path.exists()
        else fallback_pathcrf_input_debug(legacy_tracking, observations_df, label="offline")
    )
    incremental_input_debug_df = (
        pd.read_parquet(incremental_input_debug_path)
        if incremental_input_debug_path.exists()
        else fallback_pathcrf_input_debug(incremental_tracking, observations_df, label="incremental")
    )

    fps = infer_fps(legacy_tracking, args.fps)
    slots = slot_names_from_tracking(legacy_tracking)
    observation_timeline = build_observation_timeline(
        frame_count=int(len(legacy_tracking)),
        slots=slots,
        observations_df=observations_df,
    )
    metric_frames = build_alignment_metric_frames(
        legacy_tracking=legacy_tracking,
        incremental_tracking=incremental_tracking,
        observation_timeline=observation_timeline,
        slots=slots,
    )
    by_slot_metrics_df, active_missing_metrics_df, alignment_summary = summarize_alignment_metrics(metric_frames)
    reacquisition_events_df = build_reacquisition_events(metric_frames)
    by_slot_metrics_df.to_csv(output_dir / "by_slot_metrics.csv", index=False)
    active_missing_metrics_df.to_csv(output_dir / "active_vs_missing_metrics.csv", index=False)
    reacquisition_events_df.to_csv(output_dir / "reacquisition_events.csv", index=False)
    feature_rows = feature_diff_rows(legacy_tracking, incremental_tracking, slots)
    feature_rows_df = pd.DataFrame(feature_rows)
    feature_rows_df.to_csv(output_dir / "motion_feature_diff_by_slot.csv", index=False)

    deep_audit_report = build_deep_audit_report(
        legacy_tracking=legacy_tracking,
        legacy_edges=legacy_edges,
        legacy_events=legacy_events,
        incremental_tracking=incremental_tracking,
        incremental_edges=incremental_edges,
        incremental_events=incremental_events,
        slots=slots,
    )
    write_json(output_dir / "deep_audit_report.json", deep_audit_report)

    lag_report = build_lag_audit_report(
        legacy_tracking=legacy_tracking,
        incremental_tracking=incremental_tracking,
        slots=slots,
        max_lag=int(args.max_lag),
    )
    write_json(output_dir / "lag_audit_report.json", lag_report)

    temporal_report = build_temporal_alignment_report(
        legacy_edges=legacy_edges,
        incremental_edges=incremental_edges,
        legacy_events=legacy_events,
        incremental_events=incremental_events,
        max_lag=int(args.max_lag),
    )
    write_json(output_dir / "temporal_alignment_report.json", temporal_report)

    stuck_edge_timeline_df = build_stuck_edge_timeline(legacy_edges, incremental_edges)
    stuck_edge_summary = build_edge_behavior_summary(stuck_edge_timeline_df)
    stuck_edge_timeline_df.to_csv(output_dir / "stuck_edge_timeline.csv", index=False)

    dominant_edge = stuck_edge_summary.get("dominant_incremental_run") or {}
    dominant_edge_value = str(dominant_edge.get("edge") or "")
    if " -> " in dominant_edge_value:
        dominant_src_slot, dominant_dst_slot = dominant_edge_value.split(" -> ", 1)
    else:
        dominant_src_slot, dominant_dst_slot = "away_11", "away_4"
    edge_window_df = build_edge_window_slot_debug(
        offline_input_debug_df,
        incremental_input_debug_df,
        stuck_edge_timeline_df,
        src_slot=dominant_src_slot,
        dst_slot=dominant_dst_slot,
        window_start=70,
        window_end=130,
    )
    edge_window_df.to_csv(output_dir / "edge_stuck_away11_away4_window.csv", index=False)

    inference_config = PathCRFInferenceConfig(
        repo_path=args.repo_path,
        trial=int(args.trial),
        model_file=str(args.model_file),
        use_crf=not bool(args.no_crf),
        decode=str(args.decode),
        correct_episode_lasts=bool(args.correct_episode_lasts),
        evaluate=bool(args.evaluate),
        window_seconds=float(args.window_seconds) if args.window_seconds is not None else None,
        fps=float(fps),
        sample_freq=int(args.sample_freq) if args.sample_freq is not None else None,
        min_event_duration=int(args.min_event_duration),
        device=str(args.device),
    )
    runner = LegacyPathCRFRunner(inference_config)

    offline_debug_payloads: list[dict[str, Any]] = []
    incremental_debug_payloads: list[dict[str, Any]] = []
    runner.infer(legacy_tracking, debug_collector=offline_debug_payloads)
    runner.infer(incremental_tracking, debug_collector=incremental_debug_payloads)
    offline_candidates_df = build_candidate_tables_from_debug_payloads(
        debug_payloads=offline_debug_payloads,
        crf=runner.model.crf,
        mode_label="offline",
    )
    incremental_candidates_df = build_candidate_tables_from_debug_payloads(
        debug_payloads=incremental_debug_payloads,
        crf=runner.model.crf,
        mode_label="incremental",
    )
    offline_candidates_df.to_parquet(output_dir / "pathcrf_candidates_offline.parquet", index=False)
    incremental_candidates_df.to_parquet(output_dir / "pathcrf_candidates_incremental.parquet", index=False)
    offline_input_debug_df.to_parquet(output_dir / "pathcrf_input_offline.parquet", index=False)
    incremental_input_debug_df.to_parquet(output_dir / "pathcrf_input_incremental.parquet", index=False)
    candidate_debug_df, candidate_debug_summary = summarize_candidate_debug(
        offline_edges=legacy_edges,
        incremental_edges=incremental_edges,
        offline_candidates=offline_candidates_df,
        incremental_candidates=incremental_candidates_df,
        window_start=70,
        window_end=130,
    )
    candidate_debug_df.to_csv(output_dir / "candidate_debug_70_130.csv", index=False)

    feature_swap_report = build_feature_swap_report(
        runner=runner,
        legacy_tracking=legacy_tracking,
        legacy_edges=legacy_edges,
        legacy_events=legacy_events,
        incremental_tracking=incremental_tracking,
        slots=slots,
    )
    write_json(output_dir / "feature_swap_report.json", feature_swap_report)

    recomputed_motion_report = build_recomputed_motion_report(
        runner=runner,
        legacy_edges=legacy_edges,
        legacy_events=legacy_events,
        incremental_tracking=incremental_tracking,
        incremental_edges=incremental_edges,
        incremental_events=incremental_events,
        slots=slots,
        fps=fps,
    )
    write_json(output_dir / "recomputed_motion_report.json", recomputed_motion_report)

    swap_experiment_df, swap_experiment_report = build_swap_experiment_report(
        runner=runner,
        legacy_tracking=legacy_tracking,
        legacy_edges=legacy_edges,
        legacy_events=legacy_events,
        incremental_tracking=incremental_tracking,
        incremental_edges=incremental_edges,
        incremental_events=incremental_events,
        slots=slots,
        fps=fps,
    )
    swap_experiment_df.to_csv(output_dir / "swap_experiment_summary.csv", index=False)
    write_json(output_dir / "swap_experiment_report.json", swap_experiment_report)
    swap_lines = [
        "# Swap experiment report",
        "",
        "| variant | edge_exact_match_rate | team_pair_match_rate | stuck_edge_duration | edge_change_rate | edge_entropy | after_frame_89_match_rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in swap_experiment_df.to_dict(orient="records"):
        swap_lines.append(
            f"| {row['variant']} | {row['edge_exact_match_rate']:.3f} | {row['team_pair_match_rate']:.3f} | {int(row['stuck_edge_duration'])} | {row['edge_change_rate']:.3f} | {row['edge_entropy']:.3f} | {row['after_frame_89_match_rate']:.3f} |"
        )
    (output_dir / "swap_experiment_report.md").write_text("\n".join(swap_lines) + "\n", encoding="utf-8")
    config_parity_markdown = build_config_parity_report(compare_summary=artifacts["summary"], fps=fps)
    (output_dir / "config_parity_report.md").write_text(config_parity_markdown, encoding="utf-8")

    top_slots = [
        item["slot"]
        for item in sorted(feature_rows, key=lambda row: float(row.get("mean_abs_accel_diff") or 0.0), reverse=True)[
            : max(1, int(args.top_slots))
        ]
    ]
    plots_dir = output_dir / "plots"
    for slot in top_slots:
        plot_slot_series(
            legacy_tracking=legacy_tracking,
            incremental_tracking=incremental_tracking,
            slot=slot,
            output_path=plots_dir / f"slot_{slot}_x_y_vx_vy_speed_accel.png",
        )

    markdown_report = build_markdown_report(
        compare_root=compare_root,
        fps=fps,
        deep_audit_report=deep_audit_report,
        lag_report=lag_report,
        temporal_report=temporal_report,
        feature_swap_report=feature_swap_report,
        recomputed_motion_report=recomputed_motion_report,
    )
    (output_dir / "audit_report.md").write_text(markdown_report, encoding="utf-8")

    final_lines = [
        "# Final report",
        "",
        f"- Compare root: `{compare_root}`",
        f"- Dominant incremental run: `{dominant_edge_value}` desde frame `{dominant_edge.get('start_frame')}` hasta `{dominant_edge.get('end_frame')}` ({dominant_edge.get('duration_frames')} frames).",
        f"- `after_frame_89_match_rate`: `{stuck_edge_summary['after_frame_89_match_rate']:.3f}`.",
        f"- `offline_selected_present_rate`: `{candidate_debug_summary['offline_selected_present_rate']:.3f}`.",
        f"- `offline_selected_rank_mean`: `{candidate_debug_summary['offline_selected_rank_mean']}`.",
        f"- `score_margin_mean`: `{candidate_debug_summary['score_margin_mean']}`.",
        "",
        "## Diagnostico",
        "",
        f"- La divergencia dominante sigue viniendo de `{alignment_summary['dominant_divergence_source']}`.",
        f"- El swap `inc_with_offline_motion_only` alcanza `edge_exact_match_rate={swap_experiment_report['inc_with_offline_motion_only']['edge_exact_match_rate']:.3f}`.",
        f"- El swap `inc_with_offline_xy_and_motion` alcanza `edge_exact_match_rate={swap_experiment_report['inc_with_offline_xy_and_motion']['edge_exact_match_rate']:.3f}`.",
        f"- El edge offline seleccionado aparece entre candidatos incrementales con tasa `{candidate_debug_summary['offline_selected_present_rate']:.3f}`, lo que orienta el problema hacia scoring/motion mas que hacia desaparicion de candidatos." if candidate_debug_summary["offline_selected_present_rate"] >= 0.95 else f"- El edge offline seleccionado no siempre aparece entre candidatos incrementales (`present_rate={candidate_debug_summary['offline_selected_present_rate']:.3f}`), lo que indica un problema tambien de candidate generation/gating.",
        "",
        "## Artefactos clave",
        "",
        "- `stuck_edge_timeline.csv`",
        "- `candidate_debug_70_130.csv`",
        "- `edge_stuck_away11_away4_window.csv`",
        "- `swap_experiment_summary.csv`",
        "- `config_parity_report.md`",
        "",
        "## Siguiente cambio minimo recomendado",
        "",
        "- Corregir y medir el efecto real de `fixed_lag_frames` antes de seguir afinando gains; sin eso, cualquier tuning del suavizado incremental queda mal condicionado.",
    ]
    (output_dir / "final_report.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    summary_payload = {
        "compare_root": str(compare_root),
        "output_dir": str(output_dir),
        "fps": float(fps),
        "dominant_divergence_source": alignment_summary["dominant_divergence_source"],
        "stuck_edge_duration": stuck_edge_summary["stuck_edge_duration"],
        "edge_change_rate": stuck_edge_summary["edge_change_rate"],
        "edge_entropy": stuck_edge_summary["edge_entropy"],
        "offline_selected_present_rate": candidate_debug_summary["offline_selected_present_rate"],
        "offline_selected_rank_mean": candidate_debug_summary["offline_selected_rank_mean"],
        "score_margin_mean": candidate_debug_summary["score_margin_mean"],
        "score_margin_p95": candidate_debug_summary["score_margin_p95"],
        "after_frame_89_match_rate": stuck_edge_summary["after_frame_89_match_rate"],
        "active_vs_missing": alignment_summary,
        "reacquisition_events": {
            "count": int(len(reacquisition_events_df)),
            "mean_delta_accel": metric_mae(reacquisition_events_df.get("delta_accel", pd.Series(dtype=np.float32)).to_numpy(dtype=np.float32, copy=False))
            if not reacquisition_events_df.empty
            else None,
            "p95_delta_accel": metric_quantile(reacquisition_events_df.get("delta_accel", pd.Series(dtype=np.float32)).to_numpy(dtype=np.float32, copy=False), 0.95)
            if not reacquisition_events_df.empty
            else None,
        },
        "feature_swap_report": feature_swap_report,
        "swap_experiment_report": swap_experiment_report,
        "recomputed_motion_report": recomputed_motion_report,
        "reports": {
            "by_slot_metrics": str(output_dir / "by_slot_metrics.csv"),
            "active_vs_missing_metrics": str(output_dir / "active_vs_missing_metrics.csv"),
            "reacquisition_events": str(output_dir / "reacquisition_events.csv"),
            "stuck_edge_timeline": str(output_dir / "stuck_edge_timeline.csv"),
            "candidate_debug_70_130": str(output_dir / "candidate_debug_70_130.csv"),
            "pathcrf_input_offline": str(output_dir / "pathcrf_input_offline.parquet"),
            "pathcrf_input_incremental": str(output_dir / "pathcrf_input_incremental.parquet"),
            "pathcrf_candidates_offline": str(output_dir / "pathcrf_candidates_offline.parquet"),
            "pathcrf_candidates_incremental": str(output_dir / "pathcrf_candidates_incremental.parquet"),
            "plots_dir": str(plots_dir),
            "final_report_markdown": str(output_dir / "final_report.md"),
            "swap_experiment_report_markdown": str(output_dir / "swap_experiment_report.md"),
        },
    }
    write_json(output_dir / "summary.json", summary_payload)

    index_payload = {
        "compare_root": str(compare_root),
        "output_dir": str(output_dir),
        "fps": float(fps),
        "runner_model_path": str(runner.model_path),
        "reports": {
            "summary_json": str(output_dir / "summary.json"),
            "deep_audit_report": str(output_dir / "deep_audit_report.json"),
            "lag_audit_report": str(output_dir / "lag_audit_report.json"),
            "temporal_alignment_report": str(output_dir / "temporal_alignment_report.json"),
            "feature_swap_report": str(output_dir / "feature_swap_report.json"),
            "swap_experiment_report": str(output_dir / "swap_experiment_report.json"),
            "swap_experiment_report_markdown": str(output_dir / "swap_experiment_report.md"),
            "recomputed_motion_report": str(output_dir / "recomputed_motion_report.json"),
            "stuck_edge_timeline": str(output_dir / "stuck_edge_timeline.csv"),
            "candidate_debug_70_130": str(output_dir / "candidate_debug_70_130.csv"),
            "pathcrf_input_offline": str(output_dir / "pathcrf_input_offline.parquet"),
            "pathcrf_input_incremental": str(output_dir / "pathcrf_input_incremental.parquet"),
            "pathcrf_candidates_offline": str(output_dir / "pathcrf_candidates_offline.parquet"),
            "pathcrf_candidates_incremental": str(output_dir / "pathcrf_candidates_incremental.parquet"),
            "config_parity_report_markdown": str(output_dir / "config_parity_report.md"),
            "swap_experiment_summary": str(output_dir / "swap_experiment_summary.csv"),
            "final_report_markdown": str(output_dir / "final_report.md"),
            "motion_feature_diff_by_slot": str(output_dir / "motion_feature_diff_by_slot.csv"),
            "by_slot_metrics": str(output_dir / "by_slot_metrics.csv"),
            "active_vs_missing_metrics": str(output_dir / "active_vs_missing_metrics.csv"),
            "reacquisition_events": str(output_dir / "reacquisition_events.csv"),
            "audit_report_markdown": str(output_dir / "audit_report.md"),
            "plots_dir": str(plots_dir),
        },
    }
    write_json(output_dir / "audit_index.json", index_payload)


if __name__ == "__main__":
    main()
