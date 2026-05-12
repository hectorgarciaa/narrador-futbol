from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn.cluster import KMeans

from football_ai.core.config import Config
from football_ai.tracking.tracker import Tracker
from football_ai.tracking.phases.detection.utils import normalize_detection_class_name
from football_ai.tracking.phases.identification.team_detector_utils import (
    bbox_area,
    extract_shirt_crop,
    field_position_to_tuple,
    goalkeeper_position_gate,
    person_axis_positions,
    referee_position_gate,
    serialize_color,
)


SUPPORTED_CLASSES = frozenset({"player", "goalkeeper", "referee", "ball"})


@dataclass
class RelabelAnalysisArtifacts:
    output_dir: Path
    detections_csv: Path
    frames_jsonl: Path
    summary_json: Path


def _coerce_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _ensure_serializable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_ensure_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _ensure_serializable(item) for key, item in value.items()}
    return value


def _lab_to_rgb_uint8(lab_triplet: list[float] | tuple[float, float, float] | np.ndarray | None) -> tuple[int, int, int]:
    if lab_triplet is None:
        return (160, 160, 160)

    # El proyecto guarda LAB en el rango OpenCV de imagen de 8 bits
    # (L,a,b en [0,255]). Para convertirlo bien a RGB/HEX, OpenCV espera
    # precisamente un píxel uint8 en ese mismo rango.
    arr = np.asarray(lab_triplet, dtype=np.float32).reshape(1, 1, 3)
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_LAB2BGR)[0, 0]
    rgb = bgr[::-1]
    rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def lab_to_hex(lab_triplet: list[float] | tuple[float, float, float] | np.ndarray | None) -> str:
    r, g, b = _lab_to_rgb_uint8(lab_triplet)
    return f"#{r:02x}{g:02x}{b:02x}"


def load_config(config_path: str | Path | None = None) -> Config:
    return Config.from_yaml(None if config_path is None else str(config_path))


def build_tracker_for_analysis(
    config_path: str | Path | None = None,
    model_path: str | Path | None = None,
    team_colors: dict[str, list[float]] | None = None,
) -> tuple[Config, Tracker]:
    config = load_config(config_path)
    detector_conf = dict(config.detection)
    team_detector_conf = dict(config.team_detector)
    bytetracker_conf = dict(config.bytetracker)
    ball_conf = dict(config.ball)
    tracker_conf = dict(config.tracking)
    projector_conf = dict(config.projector)

    if team_colors is not None:
        team_detector_conf["team_colors"] = team_colors

    effective_model_path = (
        _coerce_path(model_path)
        if model_path is not None
        else config.get_path("paths", "models", "modelo_base")
    )
    tracker = Tracker(
        str(effective_model_path),
        detector_conf,
        team_detector_conf,
        bytetracker_conf,
        ball_conf,
        tracker_conf,
        projector_conf,
        config.project_root,
    )
    return config, tracker


def _filter_supported_detection_boxes(results) -> None:
    names = {
        key: normalize_detection_class_name(value)
        for key, value in results.names.items()
    }
    results.names = names
    keep_indexes: list[int] = []
    if results.boxes is None:
        return
    for index, cls_value in enumerate(results.boxes.cls.tolist()):
        class_name = names.get(int(cls_value))
        if class_name in SUPPORTED_CLASSES:
            keep_indexes.append(index)
    if len(keep_indexes) == len(results.boxes.cls):
        return
    results.boxes = results.boxes[keep_indexes]


def _build_distances_dict(team_detector, shirt_color: np.ndarray | None) -> dict[str, float | None] | None:
    if shirt_color is None:
        return None
    distances: dict[str, float | None] = {}
    for team_name, team_color in team_detector.team_colors.items():
        if team_color is None:
            distances[str(team_name)] = None
            continue
        distances[str(team_name)] = float(np.linalg.norm(shirt_color - team_color))
    return distances


def _should_accept_sample_for_bucket(team_detector, sample_bucket: str, confidence: float | None) -> bool:
    if sample_bucket not in team_detector.updated:
        return False
    conf_value = 0.0 if confidence is None else float(confidence)
    min_conf = float(team_detector.color_model.min_conf.get(sample_bucket, 0.0))
    return bool(conf_value >= min_conf or (not bool(team_detector.updated[sample_bucket]) and int(team_detector.n_frame) > 200))


def _analysis_team_detector(tracker: Tracker):
    return tracker.identification_phase.team_detector


def _crop_to_rgb_uint8(image_bgr: np.ndarray | None, max_side: int = 96) -> np.ndarray | None:
    if image_bgr is None or image_bgr.size <= 0:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        return None
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale < 1.0:
        image_rgb = cv2.resize(
            image_rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image_rgb


def _build_kmeans_prediction_image(crop_bgr: np.ndarray, n_clusters: int = 2) -> tuple[np.ndarray, np.ndarray] | None:
    if crop_bgr is None or crop_bgr.size <= 0:
        return None
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape(-1, 3).astype(np.float32)
    if len(pixels) < n_clusters:
        return None

    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, random_state=0)
    km.fit(pixels)
    labels = km.labels_.reshape(lab.shape[:2])
    centers_lab = np.asarray(km.cluster_centers_, dtype=np.float32)
    centers_lab_u8 = np.clip(np.round(centers_lab), 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    centers_bgr = cv2.cvtColor(centers_lab_u8, cv2.COLOR_LAB2BGR).reshape(-1, 3)

    clustered_bgr = np.zeros_like(crop_bgr)
    for cluster_idx in range(n_clusters):
        clustered_bgr[labels == cluster_idx] = centers_bgr[cluster_idx]
    return clustered_bgr, centers_lab


def _run_team_detector_with_debug(
    team_detector,
    frame_bgr: np.ndarray,
    filtering_clean: dict[str, Any],
    field_width_m: float,
    sideline_band_distance_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shirts_by_detection_idx: dict[int, np.ndarray | None] = {}
    colors_by_detection_idx: dict[int, np.ndarray | None] = {}
    color_candidate_indices: list[int] = []
    color_candidate_shirts: list[np.ndarray] = []
    bbox_xyxy = np.asarray(filtering_clean.get("bbox_xyxy", []), dtype=np.float32).reshape(-1, 4)
    confidences = list(filtering_clean.get("confidence", []))
    field_positions = np.asarray(filtering_clean.get("field_positions_m", []), dtype=np.float32).reshape(-1, 2)
    yolo_class_labels = [str(class_name) for class_name in filtering_clean.get("class_name", [])]
    x_positions = person_axis_positions(field_positions, yolo_class_labels, 0)
    y_positions = person_axis_positions(field_positions, yolo_class_labels, 1)

    for det_idx, (class_name, bbox) in enumerate(zip(yolo_class_labels, bbox_xyxy)):
        if class_name not in team_detector.candidate_classes:
            continue
        shirt = extract_shirt_crop(frame_bgr, bbox)
        shirts_by_detection_idx[det_idx] = shirt
        if shirt is None:
            continue
        color_candidate_indices.append(det_idx)
        color_candidate_shirts.append(shirt)

    if color_candidate_shirts:
        shirt_colors = team_detector.shirt_detector.get_color_kmeans_batch(color_candidate_shirts)
        for det_idx, shirt_color in zip(color_candidate_indices, shirt_colors):
            colors_by_detection_idx[det_idx] = None if shirt_color is None else np.asarray(shirt_color, dtype=np.float32)

    teams_of_detected_objects: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for det_idx, field_position in enumerate(field_positions):
        field_position = field_position_to_tuple(field_position)
        class_name = yolo_class_labels[det_idx]
        bbox_size = bbox_area(bbox_xyxy[det_idx])
        bootstrap_bucket = None
        bootstrap_index = None
        sample_accepted = False
        confidence = None
        try:
            confidence = float(confidences[det_idx])
        except Exception:
            confidence = None

        if class_name not in team_detector.candidate_classes:
            team_payload = {
                "class": class_name,
                "team": None,
                "shirt_color": None,
                "distances": None,
                "bbox_size": float(bbox_size),
            }
            teams_of_detected_objects.append(team_payload)
            debug_rows.append(
                {
                    "det_idx": int(det_idx),
                    "class_yolo": class_name,
                    "sample_bucket_initial": class_name,
                    "sample_bucket_after_new_possible": class_name,
                    "new_possible_class": None,
                    "effective_class_before_reassign": class_name,
                    "class_final": class_name,
                    "team_final": None,
                    "field_position_m": None if field_position is None else [float(field_position[0]), float(field_position[1])],
                    "shirt_color_lab": None,
                    "distances": None,
                    "bbox_size": float(bbox_size),
                    "confidence": confidence,
                    "can_be_ref": False,
                    "can_be_middle_ref": False,
                    "can_be_goalkeeper": False,
                    "nearest_outfield_team": None,
                    "reason_path": "non_candidate_class",
                    "player_stats_ready": bool(team_detector.updated["player"] and len(team_detector.outfield_team_distance_stats) >= 2),
                    "updated_player": bool(team_detector.updated["player"]),
                    "updated_referee": bool(team_detector.updated["referee"]),
                    "bootstrap_bucket": None,
                    "bootstrap_index": None,
                    "shirt_crop": None,
                }
            )
            continue

        shirt_color = colors_by_detection_idx.get(det_idx)
        sample_bucket = class_name
        effective_class = class_name
        new_possible_class = None
        distances_before = _build_distances_dict(team_detector, shirt_color)
        player_stats_ready = bool(team_detector.updated["player"] and len(team_detector.outfield_team_distance_stats) >= 2)
        within_team_bounds: dict[str, bool | None] = {}
        nearest_outfield_team = None

        if distances_before is not None:
            nearest_outfield_team = team_detector.color_model.nearest_outfield_team(distances_before)
            for team_name, distance_value in distances_before.items():
                if team_name == "referee":
                    continue
                stats = team_detector.outfield_team_distance_stats.get(team_name)
                within_team_bounds[team_name] = (
                    None
                    if distance_value is None or not isinstance(stats, dict)
                    else bool(float(distance_value) < float(stats.get("upper_bound", np.inf)))
                )

        if field_position is not None and shirt_color is not None and sample_bucket in team_detector.updated:
            if team_detector.updated[sample_bucket]:
                proposed_bucket, proposed_class, sample_decision_reason = team_detector.color_model.decide_sample_class(
                    shirt_color,
                    can_be_goalkeeper=goalkeeper_position_gate(
                        x_positions,
                        field_position,
                        field_width_m,
                        sideline_band_distance_m,
                    ),
                    can_be_middle_ref=referee_position_gate(
                        x_positions,
                        y_positions,
                        field_position,
                        field_width_m,
                        sideline_band_distance_m,
                    )[1],
                )
                sample_bucket = proposed_bucket if proposed_bucket is not None else sample_bucket
                effective_class = proposed_class if proposed_class is not None else effective_class
                new_possible_class = proposed_class

            if sample_bucket in team_detector.updated:
                previous_count = len(team_detector.class_samples[sample_bucket])
                sample_trace, _cluster_event = team_detector.color_model.maybe_add_sample(
                    sample_bucket,
                    shirt_color,
                    0.0 if confidence is None else float(confidence),
                    int(team_detector.n_frame),
                )
                sample_accepted = bool(sample_trace and sample_trace.get("accepted"))
                if sample_accepted:
                    bootstrap_bucket = sample_bucket
                    bootstrap_index = previous_count + 1

        can_be_ref = False
        can_be_middle_ref = False
        if field_position is not None:
            can_be_ref, can_be_middle_ref = referee_position_gate(
                x_positions,
                y_positions,
                field_position,
                field_width_m,
                sideline_band_distance_m,
            )
        can_be_goalkeeper = bool(
            field_position is not None
            and goalkeeper_position_gate(
                x_positions,
                field_position,
                field_width_m,
                sideline_band_distance_m,
            )
        )
        class_name_aux, team, distances, relabel_trace = team_detector._reassign_class(
            shirt_color,
            effective_class,
            field_position,
            can_be_ref,
            can_be_middle_ref,
            can_be_goalkeeper,
        )
        final_class_name = class_name_aux if field_position is not None else class_name
        team_payload = {
            "class": final_class_name,
            "team": team,
            "shirt_color": serialize_color(shirt_color),
            "distances": distances,
            "bbox_size": float(bbox_size),
        }
        teams_of_detected_objects.append(team_payload)

        relabel_reason = str(relabel_trace.get("reason", ""))
        if new_possible_class is not None:
            reason_path = f"new_possible:{new_possible_class}|reassign:{final_class_name}|reason:{relabel_reason}"
        elif final_class_name != class_name:
            reason_path = f"reassign_only:{class_name}->{final_class_name}|reason:{relabel_reason}"
        else:
            reason_path = f"kept_class|reason:{relabel_reason}"

        debug_rows.append(
            {
                "det_idx": int(det_idx),
                "class_yolo": class_name,
                "sample_bucket_initial": class_name,
                "sample_bucket_after_new_possible": sample_bucket,
                "new_possible_class": new_possible_class,
                "effective_class_before_reassign": effective_class,
                "class_final": final_class_name,
                "team_final": team,
                "field_position_m": None if field_position is None else [float(field_position[0]), float(field_position[1])],
                "shirt_color_lab": serialize_color(shirt_color),
                "distances": distances,
                "bbox_size": float(bbox_size),
                "confidence": confidence,
                "can_be_ref": can_be_ref,
                "can_be_middle_ref": bool(can_be_middle_ref),
                "can_be_goalkeeper": can_be_goalkeeper,
                "nearest_outfield_team": nearest_outfield_team,
                "reason_path": reason_path,
                "player_stats_ready": player_stats_ready,
                "updated_player": bool(team_detector.updated["player"]),
                "updated_referee": bool(team_detector.updated["referee"]),
                "bootstrap_bucket": bootstrap_bucket,
                "bootstrap_index": bootstrap_index,
                "sample_accepted_for_bucket": sample_accepted,
                "shirt_crop": shirts_by_detection_idx.get(det_idx),
                "within_team_bounds": within_team_bounds,
                "distances_before": distances_before,
                "relabel_reason": relabel_reason,
            }
        )

    return teams_of_detected_objects, debug_rows


def analyze_team_detector_video(
    config_path: str | Path | None = None,
    video_key: str = "video_prueba_medio",
    model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_frames: int | None = None,
    frame_step: int = 1,
    team_colors: dict[str, list[float]] | None = None,
    verbose: bool = True,
) -> RelabelAnalysisArtifacts:
    config, tracker = build_tracker_for_analysis(
        config_path=config_path,
        model_path=model_path,
        team_colors=team_colors,
    )
    video_path = config.get_path("paths", "data", video_key)
    resolved_output_dir = (
        _coerce_path(output_dir)
        if output_dir is not None
        else config.project_root / "output" / "analysis" / f"team_detector_{video_key}"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    detections_csv = resolved_output_dir / "detections.csv"
    frames_jsonl = resolved_output_dir / "frames.jsonl"
    summary_json = resolved_output_dir / "summary.json"
    bootstrap_samples_json = resolved_output_dir / "bootstrap_samples.json"
    bootstrap_samples_dir = resolved_output_dir / "bootstrap_samples"
    bootstrap_samples_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    team_detector = _analysis_team_detector(tracker)
    field_width_m = float(tracker.identification_phase.referee_field_width_m)
    sideline_band_distance_m = float(tracker.identification_phase.referee_sideline_band_distance_m)
    records: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    bootstrap_samples: dict[str, list[dict[str, Any]]] = {"player": [], "referee": []}
    bootstrap_closed_for_artifact: dict[str, bool] = {"player": False, "referee": False}
    processed_frames = 0
    seen_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index = seen_frames
        seen_frames += 1
        if frame_step > 1 and (frame_index % frame_step) != 0:
            continue
        if max_frames is not None and processed_frames >= max_frames:
            break

        frame_time_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        detection_packet, _detector_ms = tracker.detection_phase.process(
            frame,
            frame_index=frame_index,
            frame_time_ms=frame_time_ms,
        )
        reference_packet, _projection_ms = tracker.projection_phase.process(
            frame,
            detection_packet,
        )
        filtering_packet, _filter_ms = tracker.filtering_phase.process(
            reference_packet,
            active_track_boxes_xyxy=[],
            geometry=tracker.projection_phase.geometry,
        )
        filtering_clean = filtering_packet["clean"]
        team_detector.n_frame += 1
        teams_of_detected_objects, debug_rows = _run_team_detector_with_debug(
            team_detector,
            frame,
            filtering_clean,
            field_width_m,
            sideline_band_distance_m,
        )

        quality_diagnostics = dict(reference_packet.get("trace", {}).get("diagnostics", {}) or {})
        frame_detections_count = int(filtering_clean.get("num_detections", 0))
        field_positions = np.asarray(filtering_clean.get("field_positions_m", []), dtype=np.float32).reshape(-1, 2)
        ground_points_projected = np.asarray(
            filtering_clean.get("ground_points_image_original", []),
            dtype=np.float32,
        ).reshape(-1, 2)
        bbox_xyxy = np.asarray(filtering_clean.get("bbox_xyxy", []), dtype=np.float32).reshape(-1, 4)
        frame_row = {
            "frame_idx": int(frame_index),
            "num_detections": frame_detections_count,
            "homography_quality_status": None if quality_diagnostics is None else quality_diagnostics.get("quality_status"),
            "homography_quality_score": None if quality_diagnostics is None else quality_diagnostics.get("quality_score"),
            "field_positions_usable_for_tracking": bool(reference_packet["clean"].get("field_positions_usable_for_tracking")),
            "team_colors": {
                team_name: None if color is None else [float(x) for x in np.asarray(color, dtype=np.float32).reshape(-1).tolist()]
                for team_name, color in team_detector.team_colors.items()
            },
            "outfield_team_distance_stats": _ensure_serializable(team_detector.outfield_team_distance_stats),
            "referee_distance_stats": _ensure_serializable(team_detector.color_model.referee_distance_stats),
            "updated": {name: bool(value) for name, value in team_detector.updated.items()},
        }
        frame_rows.append(frame_row)

        for det_idx, (team_result, debug_row) in enumerate(zip(teams_of_detected_objects, debug_rows)):
            bbox = bbox_xyxy[det_idx].reshape(-1) if det_idx < len(bbox_xyxy) else np.zeros(4, dtype=np.float32)
            field_position = field_positions[det_idx] if det_idx < len(field_positions) else np.array([np.nan, np.nan], dtype=np.float32)
            ground_point = ground_points_projected[det_idx] if det_idx < len(ground_points_projected) else np.array([np.nan, np.nan], dtype=np.float32)
            distances = team_result.get("distances")
            distances_before = debug_row.get("distances_before")
            within_team_bounds = debug_row.get("within_team_bounds") or {}
            row = {
                "frame_idx": int(frame_index),
                "det_idx": int(det_idx),
                "bbox_x1": float(bbox[0]),
                "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]),
                "bbox_y2": float(bbox[3]),
                "bbox_w": float(max(0.0, bbox[2] - bbox[0])),
                "bbox_h": float(max(0.0, bbox[3] - bbox[1])),
                "confidence": debug_row.get("confidence"),
                "class_yolo": debug_row.get("class_yolo"),
                "class_final": team_result.get("class"),
                "team_final": team_result.get("team"),
                "new_possible_class": debug_row.get("new_possible_class"),
                "effective_class_before_reassign": debug_row.get("effective_class_before_reassign"),
                "sample_bucket_initial": debug_row.get("sample_bucket_initial"),
                "sample_bucket_after_new_possible": debug_row.get("sample_bucket_after_new_possible"),
                "reason_path": debug_row.get("reason_path"),
                "shirt_color_lab_l": None if team_result.get("shirt_color") is None else float(team_result["shirt_color"][0]),
                "shirt_color_lab_a": None if team_result.get("shirt_color") is None else float(team_result["shirt_color"][1]),
                "shirt_color_lab_b": None if team_result.get("shirt_color") is None else float(team_result["shirt_color"][2]),
                "shirt_color_hex": lab_to_hex(team_result.get("shirt_color")),
                "field_x_m": None if not np.all(np.isfinite(field_position)) else float(field_position[0]),
                "field_y_m": None if not np.all(np.isfinite(field_position)) else float(field_position[1]),
                "ground_x_px": None if not np.all(np.isfinite(ground_point)) else float(ground_point[0]),
                "ground_y_px": None if not np.all(np.isfinite(ground_point)) else float(ground_point[1]),
                "can_be_ref": bool(debug_row.get("can_be_ref")),
                "can_be_middle_ref": bool(debug_row.get("can_be_middle_ref")),
                "can_be_goalkeeper": bool(debug_row.get("can_be_goalkeeper")),
                "player_stats_ready": bool(debug_row.get("player_stats_ready")),
                "updated_player": bool(debug_row.get("updated_player")),
                "updated_referee": bool(debug_row.get("updated_referee")),
                "nearest_outfield_team": debug_row.get("nearest_outfield_team"),
                "distance_equipo_1": None if distances is None else distances.get("Equipo 1"),
                "distance_equipo_2": None if distances is None else distances.get("Equipo 2"),
                "distance_referee": None if distances is None else distances.get("referee"),
                "distance_before_equipo_1": None if distances_before is None else distances_before.get("Equipo 1"),
                "distance_before_equipo_2": None if distances_before is None else distances_before.get("Equipo 2"),
                "distance_before_referee": None if distances_before is None else distances_before.get("referee"),
                "within_bound_equipo_1": within_team_bounds.get("Equipo 1"),
                "within_bound_equipo_2": within_team_bounds.get("Equipo 2"),
                "relabelled": bool(debug_row.get("class_yolo") != team_result.get("class")),
                "relabeled_to_referee": bool(team_result.get("class") == "referee" and debug_row.get("class_yolo") != "referee"),
                "relabeled_to_goalkeeper": bool(team_result.get("class") == "goalkeeper" and debug_row.get("class_yolo") != "goalkeeper"),
                "homography_quality_status": frame_row["homography_quality_status"],
                "homography_quality_score": frame_row["homography_quality_score"],
            }
            team_colors = frame_row["team_colors"]
            outfield_stats = frame_row["outfield_team_distance_stats"]
            for team_name in ("Equipo 1", "Equipo 2", "referee"):
                color = team_colors.get(team_name)
                row[f"ref_{team_name}_l"] = None if color is None else float(color[0])
                row[f"ref_{team_name}_a"] = None if color is None else float(color[1])
                row[f"ref_{team_name}_b"] = None if color is None else float(color[2])
            for team_name in ("Equipo 1", "Equipo 2"):
                stats = outfield_stats.get(team_name) if isinstance(outfield_stats, dict) else None
                row[f"upper_bound_{team_name}"] = None if not isinstance(stats, dict) else stats.get("upper_bound")
                row[f"median_dist_{team_name}"] = None if not isinstance(stats, dict) else stats.get("median")
                row[f"iqr_dist_{team_name}"] = None if not isinstance(stats, dict) else stats.get("iqr")
            referee_stats = frame_row.get("referee_distance_stats")
            row["upper_bound_referee"] = None if not isinstance(referee_stats, dict) else referee_stats.get("upper_bound")
            row["median_dist_referee"] = None if not isinstance(referee_stats, dict) else referee_stats.get("median")
            row["iqr_dist_referee"] = None if not isinstance(referee_stats, dict) else referee_stats.get("iqr")
            records.append(row)

            bootstrap_bucket = debug_row.get("bootstrap_bucket")
            bootstrap_index = debug_row.get("bootstrap_index")
            shirt_crop = debug_row.get("shirt_crop")
            if (
                bootstrap_bucket in bootstrap_samples
                and bootstrap_index is not None
                and bool(debug_row.get("sample_accepted_for_bucket"))
                and not bootstrap_closed_for_artifact.get(str(bootstrap_bucket), False)
                and shirt_crop is not None
            ):
                rgb_crop = _crop_to_rgb_uint8(shirt_crop)
                if rgb_crop is not None:
                    crop_filename = f"{bootstrap_bucket}_{int(bootstrap_index):03d}_frame_{int(frame_index):05d}_det_{int(det_idx):03d}.png"
                    crop_path = bootstrap_samples_dir / crop_filename
                    cv2.imwrite(str(crop_path), cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
                    bootstrap_samples[bootstrap_bucket].append(
                        {
                            "bucket": str(bootstrap_bucket),
                            "index": int(bootstrap_index),
                            "frame_idx": int(frame_index),
                            "det_idx": int(det_idx),
                            "confidence": debug_row.get("confidence"),
                            "class_yolo": debug_row.get("class_yolo"),
                            "class_final": team_result.get("class"),
                            "reason_path": debug_row.get("reason_path"),
                            "shirt_color_lab": team_result.get("shirt_color"),
                            "shirt_color_hex": lab_to_hex(team_result.get("shirt_color")),
                            "crop_path": str(crop_path),
                        }
                    )
                    if bool(team_detector.updated.get(str(bootstrap_bucket), False)):
                        bootstrap_closed_for_artifact[str(bootstrap_bucket)] = True

        processed_frames += 1
        if verbose and (processed_frames == 1 or processed_frames % 10 == 0):
            print(f"Procesados {processed_frames} frames...", flush=True)

    cap.release()

    detections_df = pd.DataFrame.from_records(records)
    detections_df.to_csv(detections_csv, index=False)
    with frames_jsonl.open("w", encoding="utf-8") as handle:
        for row in frame_rows:
            handle.write(json.dumps(_ensure_serializable(row), ensure_ascii=False) + "\n")
    bootstrap_samples_json.write_text(
        json.dumps(_ensure_serializable(bootstrap_samples), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "video_path": str(video_path),
        "model_path": str(_coerce_path(model_path) if model_path is not None else config.get_path("paths", "models", "modelo_base")),
        "frames_processed": int(processed_frames),
        "detections_total": int(len(detections_df)),
        "relabelled_total": int(detections_df["relabelled"].sum()) if not detections_df.empty else 0,
        "relabelled_to_referee": int(detections_df["relabeled_to_referee"].sum()) if not detections_df.empty else 0,
        "relabelled_to_goalkeeper": int(detections_df["relabeled_to_goalkeeper"].sum()) if not detections_df.empty else 0,
        "bootstrap_min_samples": _ensure_serializable(team_detector.color_model.min_samples),
        "bootstrap_samples_used": {bucket: int(len(samples)) for bucket, samples in bootstrap_samples.items()},
        "bootstrap_samples_json": str(bootstrap_samples_json),
        "output_dir": str(resolved_output_dir),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return RelabelAnalysisArtifacts(
        output_dir=resolved_output_dir,
        detections_csv=detections_csv,
        frames_jsonl=frames_jsonl,
        summary_json=summary_json,
    )


def load_artifacts(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    resolved = _coerce_path(output_dir)
    detections_df = pd.read_csv(resolved / "detections.csv")
    frame_rows = []
    with (resolved / "frames.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            frame_rows.append(json.loads(line))
    frames_df = pd.DataFrame(frame_rows)
    summary = json.loads((resolved / "summary.json").read_text(encoding="utf-8"))
    return detections_df, frames_df, summary


def load_bootstrap_samples(output_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    resolved = _coerce_path(output_dir)
    bootstrap_path = resolved / "bootstrap_samples.json"
    if not bootstrap_path.exists():
        return {"player": [], "referee": []}
    return json.loads(bootstrap_path.read_text(encoding="utf-8"))


def build_bootstrap_bucket_figure(
    output_dir: str | Path,
    bucket: str,
    n_cols: int = 8,
    title: str | None = None,
) -> plt.Figure:
    bootstrap_samples = load_bootstrap_samples(output_dir)
    samples = bootstrap_samples.get(bucket, [])
    if not samples:
        raise ValueError(f"No hay muestras de bootstrap guardadas para '{bucket}'.")

    n_cols = max(1, int(n_cols))
    n_samples = len(samples)
    n_rows = int(np.ceil(n_samples / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols * 2,
        figsize=(max(16, n_cols * 3.0), max(3.8, n_rows * 3.0)),
        squeeze=False,
    )
    fig.suptitle(title or f"Bootstrap de {bucket}", fontsize=14)

    flat_axes = axes.reshape(-1)
    for ax in flat_axes:
        ax.axis("off")

    for sample_idx, sample in enumerate(samples):
        row_idx = sample_idx // n_cols
        col_idx = sample_idx % n_cols
        ax_orig = axes[row_idx][col_idx * 2]
        ax_pred = axes[row_idx][col_idx * 2 + 1]
        crop_path = Path(sample["crop_path"])
        if not crop_path.exists():
            continue

        crop_bgr = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop_bgr is None:
            continue

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pred_result = _build_kmeans_prediction_image(crop_bgr)
        pred_rgb = None if pred_result is None else cv2.cvtColor(pred_result[0], cv2.COLOR_BGR2RGB)

        ax_orig.imshow(crop_rgb)
        ax_orig.set_title(
            (
                f"#{int(sample['index'])} YOLO:{sample.get('class_yolo')}\n"
                f"f{int(sample['frame_idx'])} d{int(sample['det_idx'])} conf:{float(sample.get('confidence')):.3f}"
            ),
            fontsize=8,
        )
        ax_pred.imshow(pred_rgb if pred_rgb is not None else crop_rgb)
        ax_pred.set_title(
            f"Means -> {sample.get('class_final')}",
            fontsize=8,
        )

        border_color = str(sample.get("shirt_color_hex", "#888888"))
        for ax in (ax_orig, ax_pred):
            border = patches.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                fill=False,
                linewidth=3,
                edgecolor=border_color,
            )
            ax.add_patch(border)

    fig.tight_layout()
    return fig


def build_bootstrap_reference_figures(
    output_dir: str | Path,
    n_cols: int = 8,
) -> tuple[plt.Figure, plt.Figure]:
    return (
        build_bootstrap_bucket_figure(output_dir, bucket="player", n_cols=n_cols, title="Bootstrap de player"),
        build_bootstrap_bucket_figure(output_dir, bucket="referee", n_cols=n_cols, title="Bootstrap de referee"),
    )


def build_lab_scatter_3d(
    detections_df: pd.DataFrame,
    sample_limit: int | None = 8000,
    only_relabelled: bool = False,
    include_spheres: bool = True,
    include_reference_history: bool = True,
    reference_history_stride: int = 10,
    max_spheres_per_team: int = 10,
    title: str = "Espacio LAB de camisetas",
) -> go.Figure:
    df = detections_df.copy()
    df = df.dropna(subset=["shirt_color_lab_l", "shirt_color_lab_a", "shirt_color_lab_b"])
    if only_relabelled:
        df = df[df["relabelled"] == True]
    if sample_limit is not None and len(df) > sample_limit:
        df = df.sample(sample_limit, random_state=0)

    hover = (
        "frame=%{customdata[0]} det=%{customdata[1]}<br>"
        "YOLO=%{customdata[2]} final=%{customdata[3]}<br>"
        "team=%{customdata[4]}<br>"
        "new_possible=%{customdata[5]}<br>"
        "reason=%{customdata[6]}"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df["shirt_color_lab_l"],
            y=df["shirt_color_lab_a"],
            z=df["shirt_color_lab_b"],
            mode="markers",
            name="Detecciones",
            marker={
                "size": 4,
                "opacity": 0.65,
                "color": df["shirt_color_hex"],
                "line": {"width": 0},
                "symbol": np.where(df["relabelled"], "diamond", "circle"),
            },
            customdata=np.column_stack(
                [
                    df["frame_idx"],
                    df["det_idx"],
                    df["class_yolo"],
                    df["class_final"],
                    df["team_final"].fillna("-"),
                    df["new_possible_class"].fillna("-"),
                    df["reason_path"].fillna("-"),
                ]
            ),
            hovertemplate=hover,
        )
    )

    if not detections_df.empty:
        reference_rows = (
            detections_df.sort_values(["frame_idx", "det_idx"])
            .groupby("frame_idx", as_index=False)
            .first()
        )
        latest_refs: dict[str, list[float]] = {}
        for team_name in ("Equipo 1", "Equipo 2", "referee"):
            ref_cols = [f"ref_{team_name}_l", f"ref_{team_name}_a", f"ref_{team_name}_b"]
            ref_df = reference_rows.dropna(subset=ref_cols).copy()
            if ref_df.empty:
                continue

            if include_reference_history:
                history_df = ref_df.iloc[:: max(1, int(reference_history_stride))].copy()
                if history_df.iloc[-1]["frame_idx"] != ref_df.iloc[-1]["frame_idx"]:
                    history_df = pd.concat([history_df, ref_df.tail(1)], ignore_index=True)
                fig.add_trace(
                    go.Scatter3d(
                        x=history_df[ref_cols[0]],
                        y=history_df[ref_cols[1]],
                        z=history_df[ref_cols[2]],
                        mode="lines+markers",
                        name=f"Trayectoria ref {team_name}",
                        marker={
                            "size": 4,
                            "color": [lab_to_hex([l, a, b]) for l, a, b in history_df[ref_cols].to_numpy()],
                        },
                        line={"width": 4, "color": lab_to_hex(ref_df[ref_cols].iloc[-1].tolist())},
                        customdata=history_df[["frame_idx"]].to_numpy(),
                        hovertemplate=f"{team_name}<br>frame=%{{customdata[0]}}<extra></extra>",
                    )
                )

            latest_coords = ref_df[ref_cols].iloc[-1].tolist()
            latest_refs[team_name] = [float(value) for value in latest_coords]
            fig.add_trace(
                go.Scatter3d(
                    x=[latest_coords[0]],
                    y=[latest_coords[1]],
                    z=[latest_coords[2]],
                    mode="markers+text",
                    name=f"Referencia actual {team_name}",
                    text=[team_name],
                    textposition="top center",
                    marker={
                        "size": 10,
                        "symbol": "x",
                        "color": lab_to_hex(latest_coords),
                        "line": {"width": 3, "color": "#111111"},
                    },
                )
            )

            if include_spheres:
                stats_col = "upper_bound_referee" if team_name == "referee" else f"upper_bound_{team_name}"
                sphere_df = ref_df.dropna(subset=[stats_col]).copy()
                if sphere_df.empty:
                    continue
                if len(sphere_df) > max_spheres_per_team:
                    sphere_df = sphere_df.iloc[np.linspace(0, len(sphere_df) - 1, max_spheres_per_team, dtype=int)]
                for sphere_idx, (_, sphere_row) in enumerate(sphere_df.iterrows()):
                    center = [float(sphere_row[ref_cols[0]]), float(sphere_row[ref_cols[1]]), float(sphere_row[ref_cols[2]])]
                    radius = float(sphere_row[stats_col])
                    if not np.isfinite(radius) or radius <= 0.0:
                        continue
                    sphere_x, sphere_y, sphere_z = make_sphere_mesh(center, radius)
                    is_latest = int(sphere_row["frame_idx"]) == int(reference_rows["frame_idx"].max())
                    fig.add_trace(
                        go.Mesh3d(
                            x=sphere_x.ravel(),
                            y=sphere_y.ravel(),
                            z=sphere_z.ravel(),
                            alphahull=0,
                            opacity=0.12 if is_latest else 0.05,
                            color=lab_to_hex(center),
                            name=f"Esfera {'actual ' if is_latest else ''}{team_name}",
                            hoverinfo="skip",
                            showscale=False,
                            visible=True if is_latest else "legendonly",
                        )
                    )

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "L",
            "yaxis_title": "a",
            "zaxis_title": "b",
        },
        height=760,
        legend={"itemsizing": "constant"},
    )
    return fig


def make_sphere_mesh(center: list[float], radius: float, resolution: int = 22) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = float(center[0]) + radius * np.outer(np.cos(u), np.sin(v))
    y = float(center[1]) + radius * np.outer(np.sin(u), np.sin(v))
    z = float(center[2]) + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def build_field_relabel_map(
    detections_df: pd.DataFrame,
    sample_limit: int | None = 8000,
    title: str = "Posiciones en campo y relabel",
) -> go.Figure:
    df = detections_df.dropna(subset=["field_x_m", "field_y_m"]).copy()
    if sample_limit is not None and len(df) > sample_limit:
        df = df.sample(sample_limit, random_state=0)
    category = np.where(
        df["relabeled_to_referee"],
        "player/ref->referee",
        np.where(df["relabeled_to_goalkeeper"], "player/ref->goalkeeper", np.where(df["relabelled"], "otros relabel", "sin relabel")),
    )
    df["category"] = category

    color_map = {
        "sin relabel": "#7f8c8d",
        "player/ref->referee": "#e74c3c",
        "player/ref->goalkeeper": "#2980b9",
        "otros relabel": "#8e44ad",
    }
    fig = go.Figure()
    for cat_name, part in df.groupby("category", dropna=False):
        fig.add_trace(
            go.Scatter(
                x=part["field_x_m"],
                y=part["field_y_m"],
                mode="markers",
                name=str(cat_name),
                marker={
                    "size": 8,
                    "opacity": 0.65,
                    "color": color_map.get(str(cat_name), "#2c3e50"),
                    "symbol": "diamond" if "relabel" in str(cat_name) and str(cat_name) != "sin relabel" else "circle",
                },
                customdata=np.column_stack(
                    [
                        part["frame_idx"],
                        part["det_idx"],
                        part["class_yolo"],
                        part["class_final"],
                        part["reason_path"],
                    ]
                ),
                hovertemplate=(
                    "frame=%{customdata[0]} det=%{customdata[1]}<br>"
                    "YOLO=%{customdata[2]} final=%{customdata[3]}<br>"
                    "reason=%{customdata[4]}<br>"
                    "x=%{x:.2f}m y=%{y:.2f}m"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Longitud campo (m)",
        yaxis_title="Anchura campo (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        height=650,
    )
    return fig


def build_relabel_timeline(detections_df: pd.DataFrame) -> go.Figure:
    timeline_df = (
        detections_df.assign(
            relabel_to_ref=lambda df_: df_["relabeled_to_referee"].astype(int),
            relabel_to_gk=lambda df_: df_["relabeled_to_goalkeeper"].astype(int),
            relabel_any=lambda df_: df_["relabelled"].astype(int),
        )
        .groupby("frame_idx", as_index=False)[["relabel_any", "relabel_to_ref", "relabel_to_gk"]]
        .sum()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline_df["frame_idx"], y=timeline_df["relabel_any"], mode="lines", name="Relabel total"))
    fig.add_trace(go.Scatter(x=timeline_df["frame_idx"], y=timeline_df["relabel_to_ref"], mode="lines", name="A referee"))
    fig.add_trace(go.Scatter(x=timeline_df["frame_idx"], y=timeline_df["relabel_to_gk"], mode="lines", name="A goalkeeper"))
    fig.update_layout(
        title="Relabels por frame",
        xaxis_title="Frame",
        yaxis_title="Número de detecciones",
        height=420,
    )
    return fig


def build_distance_margin_plot(detections_df: pd.DataFrame) -> go.Figure:
    df = detections_df.copy()
    df = df.dropna(subset=["distance_before_equipo_1", "distance_before_equipo_2"])
    df["nearest_team_distance"] = df[["distance_before_equipo_1", "distance_before_equipo_2"]].min(axis=1)
    df["nearest_team_upper_bound"] = np.where(
        df["distance_before_equipo_1"] <= df["distance_before_equipo_2"],
        df["upper_bound_Equipo 1"],
        df["upper_bound_Equipo 2"],
    )
    df["distance_margin"] = df["nearest_team_distance"] - df["nearest_team_upper_bound"]
    fig = go.Figure()
    for relabel_state, part in df.groupby("relabelled", dropna=False):
        label = "Relabelled" if bool(relabel_state) else "Sin relabel"
        fig.add_trace(
            go.Histogram(
                x=part["distance_margin"],
                name=label,
                opacity=0.65,
                nbinsx=70,
            )
        )
    fig.update_layout(
        title="Margen a la frontera del cluster más cercano",
        xaxis_title="distancia_al_equipo_más_cercano - upper_bound",
        yaxis_title="Detecciones",
        barmode="overlay",
        height=420,
    )
    return fig


def show_frame_overlay(
    detections_df: pd.DataFrame,
    video_path: str | Path,
    frame_idx: int,
    line_width: int = 2,
    figsize: tuple[int, int] = (16, 9),
) -> plt.Figure:
    cap = cv2.VideoCapture(str(_coerce_path(video_path)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {frame_idx} de {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(frame_rgb)
    frame_rows = detections_df[detections_df["frame_idx"] == int(frame_idx)].copy()
    for _, row in frame_rows.iterrows():
        x1, y1, x2, y2 = row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]
        color = "#d35400" if row["relabelled"] else "#27ae60"
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=line_width)
        ax.add_patch(rect)
        label = f"{row['det_idx']} y:{row['class_yolo']} -> td:{row['class_final']}"
        ax.text(
            x1,
            max(0.0, y1 - 4.0),
            label,
            color="white",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.8, "pad": 1},
        )
    ax.set_title(f"Frame {frame_idx}")
    ax.axis("off")
    return fig
