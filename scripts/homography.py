#!/usr/bin/env python3

import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import convert_to_serializable
from football_ai.detection import Detector
from football_ai.filtering import filter_post_homography
from football_ai.reference_points import PnLCalibFieldProjector
from football_ai.reference_points.geometry import project_image_points
from football_ai.visualization.simple_drawer import (
    FieldPanel,
    VideoOutput,
    VideoPanel,
    panel_color,
)

from scripts._cli_common import (
    build_timestamp,
    build_video_model_parser,
    load_config,
    resolve_video_and_model_paths,
    slugify_model_name,
)

REJECTED_DETECTION_COLOR = (0, 0, 255)


def _result_to_detections(result):
    detections = []
    names = getattr(result, "names", {})
    for box in result.boxes:
        class_id = int(box.cls)
        detections.append(
            {
                "class": names.get(class_id, str(class_id)),
                "confidence": float(box.conf),
                "bbox": [int(value) for value in box.xyxy[0].tolist()],
            }
        )
    return detections


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _estimate_keypoints_field(estimate, homography_image_to_field):
    if estimate is None or homography_image_to_field is None:
        return []
    items = []
    for keypoint_id, point in estimate.keypoints_dict.items():
        image_point = np.asarray([[point["x"], point["y"]]], dtype=np.float32)
        field_point = project_image_points(image_point, homography_image_to_field)[0]
        items.append(
            {
                "id": int(keypoint_id),
                "confidence": _safe_float(point.get("p")),
                "image_position_px": [float(point["x"]), float(point["y"])],
                "field_position_m": field_point.astype(np.float32),
            }
        )
    return items


def _estimate_lines_field(estimate, homography_image_to_field):
    if estimate is None or homography_image_to_field is None:
        return []
    items = []
    for line_id, line in estimate.lines_dict.items():
        image_points = np.asarray(
            [[line["x_1"], line["y_1"]], [line["x_2"], line["y_2"]]],
            dtype=np.float32,
        )
        field_points = project_image_points(image_points, homography_image_to_field)
        items.append(
            {
                "id": int(line_id),
                "confidence_1": _safe_float(line.get("p_1")),
                "confidence_2": _safe_float(line.get("p_2")),
                "image_point_1_px": [float(line["x_1"]), float(line["y_1"])],
                "image_point_2_px": [float(line["x_2"]), float(line["y_2"])],
                "field_point_1_m": field_points[0].astype(np.float32),
                "field_point_2_m": field_points[1].astype(np.float32),
            }
        )
    return items


def _homography_text_lines(frame_record):
    homography = frame_record["homography"]
    selected_attempt = homography.get("quality_diagnostics", {}).get("selected_attempt") or {}
    return [
        f"frame {frame_record['frame_idx']}",
        f"estado: {homography.get('quality_status')}",
        f"score: {homography.get('quality_score')}",
        f"modo: {homography.get('estimation_mode')}",
        f"kp: {homography.get('visible_keypoints_count')}  lineas: {homography.get('visible_lines_count')}",
        f"intentos: {homography.get('attempt_count')}",
        f"sel: {selected_attempt.get('attempt_index', '-')}",
        f"reproj: {homography.get('reprojection_error')}",
        f"rechazo: {homography.get('rejection_type') or '-'}",
    ]


def _summary_lines(prefix, items, confidence_key):
    if not items:
        return [f"{prefix}: -"]
    text = []
    for item in items[:6]:
        confidence = item.get(confidence_key)
        if confidence is None:
            text.append(str(item["id"]))
        else:
            text.append(f'{item["id"]}:{confidence:.2f}')
    return [f"{prefix}: " + ", ".join(text)]


def _position_label(position):
    if position is None or len(position) < 2 or not np.all(np.isfinite(position[:2])):
        return None
    return f"{float(position[0]):.1f},{float(position[1]):.1f}"


def _colorize_detection(detection):
    passed_filter = bool(detection.get("passed_post_homography_filter", False))
    class_color = panel_color(detection["class"])
    detection["render_color_video"] = class_color if passed_filter else REJECTED_DETECTION_COLOR
    detection["render_color_field"] = class_color
    return detection


def _annotate_filtered_detections(detections, keep_mask, field_positions_all_m):
    keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
    field_positions_all_m = np.asarray(field_positions_all_m, dtype=np.float32).reshape(-1, 2)
    for idx, detection in enumerate(detections):
        passed_filter = bool(idx < len(keep_mask) and keep_mask[idx])
        field_position = field_positions_all_m[idx] if idx < len(field_positions_all_m) else np.asarray([np.nan, np.nan], dtype=np.float32)
        detection["passed_post_homography_filter"] = passed_filter
        detection["field_position_m"] = field_position
        _colorize_detection(detection)
    return detections


def _filter_summary(detections):
    total = len(detections)
    passed = sum(1 for detection in detections if detection.get("passed_post_homography_filter"))
    return {
        "total_detections_before_filter": int(total),
        "total_detections_passed_filter": int(passed),
        "total_detections_rejected_filter": int(total - passed),
    }


def _field_detections(detections):
    return [
        detection
        for detection in detections
        if detection.get("passed_post_homography_filter")
    ]


def _video_panel_item(detection):
    return {
        "bbox": detection["bbox"],
        "color": detection.get("render_color_video") or panel_color(detection["class"]),
        "class_name": detection["class"],
        "confidence": detection["confidence"],
        "field_position_m": detection.get("field_position_m"),
    }


def _field_entity(detection):
    return {
        "position_m": detection.get("field_position_m"),
        "color": detection.get("render_color_field") or panel_color(detection["class"]),
        "canonical_id": detection.get("canonical_id"),
        "position_label": _position_label(detection.get("field_position_m")),
        "radius": 4 if detection["class"] == "ball" else 8,
    }


def _field_point(item, color, radius):
    return {
        "position_m": item.get("field_position_m"),
        "color": color,
        "canonical_id": None,
        "position_label": None,
        "radius": radius,
    }


def _field_line(item):
    return {
        "point_1_m": item.get("field_point_1_m"),
        "point_2_m": item.get("field_point_2_m"),
        "color": (255, 255, 0),
        "thickness": 2,
    }


def main():
    parser = build_video_model_parser("Detección + homografía PnLCalib")
    args = parser.parse_args()

    config = load_config(args.config)
    video_path, model_path = resolve_video_and_model_paths(args, config)

    timestamp = build_timestamp()
    output_dir = PROJECT_ROOT / "output" / "homography" / slugify_model_name(model_path) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(str(model_path), **config.detection)
    projector_conf = dict(config.projector.get("constructor", {}))
    projector = PnLCalibFieldProjector(project_root=config.project_root, **projector_conf)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = VideoOutput(
        output_dir / "homography.mp4",
        fps,
        [[VideoPanel((height, width)), FieldPanel((height, width))]],
    )

    frames = {}
    results = detector.detect(str(video_path), stream=True)
    if args.max_frames is not None:
        results = itertools.islice(results, args.max_frames)
    for frame_index, result in enumerate(results):

        detections = _result_to_detections(result)
        class_names = [item["class"] for item in detections]
        boxes = np.asarray([item["bbox"] for item in detections], dtype=np.float32).reshape(-1, 4)
        projection_result, projection_meta = projector.project_detections_with_metadata(
            result.orig_img,
            boxes,
            class_names=class_names,
        )

        estimate = projection_meta.get("estimate")
        homography_image_to_field = projection_result.homography_image_to_field
        keep_mask, field_positions_all_m = filter_post_homography(
            projection_meta.get("field_positions_all_m"),
            has_homography=homography_image_to_field is not None,
            field_positions_usable_for_tracking=projection_result.field_positions_usable_for_tracking,
            geometry=projector.geometry,
            field_length_m=projector.geometry.field_length_m,
            field_width_m=projector.geometry.field_width_m,
            detection_boxes=boxes if len(boxes) > 0 else None,
            active_track_boxes=None,
        )
        keypoints_field = _estimate_keypoints_field(estimate, homography_image_to_field)
        lines_field = _estimate_lines_field(estimate, homography_image_to_field)
        detections = _annotate_filtered_detections(
            detections,
            keep_mask,
            field_positions_all_m,
        )
        field_detections = _field_detections(detections)
        filter_summary = _filter_summary(detections)

        frame_record = {
            "frame_idx": frame_index,
            "detections": detections,
            **filter_summary,
            "homography": {
                "has_homography": homography_image_to_field is not None,
                "homography_image_to_field": homography_image_to_field,
                "frame_shape_original": projection_result.frame_shape_original,
                "frame_shape_projected": projection_result.frame_shape_projected,
                "ground_points_image_original": projection_result.ground_points_image_original,
                "ground_points_image_projected": projection_result.ground_points_image_projected,
                "field_positions_all_m": field_positions_all_m,
                "field_positions_usable_for_tracking": projection_result.field_positions_usable_for_tracking,
                "estimation_mode": projection_result.estimation_mode,
                "reprojection_error": projection_result.reprojection_error,
                "visible_keypoints_count": projection_result.visible_keypoints_count,
                "visible_lines_count": projection_result.visible_lines_count,
                "keypoint_threshold_used": projection_result.keypoint_threshold_used,
                "line_threshold_used": projection_result.line_threshold_used,
                "quality_score": projection_result.homography_quality_score,
                "quality_status": projection_result.homography_quality_status,
                "quality_diagnostics": projection_result.quality_diagnostics or {},
                "attempt_count": len((projection_result.quality_diagnostics or {}).get("attempts", [])),
                "rejection_type": (projection_result.quality_diagnostics or {}).get("rejection_type"),
                "rejection_reasons": (projection_result.quality_diagnostics or {}).get("rejection_reasons", []),
                "keypoints": keypoints_field,
                "lines": lines_field,
            },
        }
        frames[str(frame_index)] = frame_record

        text_lines = _homography_text_lines(frame_record)
        if homography_image_to_field is None:
            text_lines.append("sin homografia valida")
        text_lines.extend(_summary_lines("kp ids", keypoints_field, "confidence"))
        text_lines.extend(_summary_lines("line ids", lines_field, "confidence_1"))
        writer.write_frame(
            [[
                {
                    "frame": result.orig_img.copy(),
                    "items": [_video_panel_item(detection) for detection in detections],
                },
                {
                    "field_size_m": (
                        projector.geometry.field_length_m,
                        projector.geometry.field_width_m,
                    ),
                    "entities": [_field_entity(detection) for detection in field_detections],
                    "points": [_field_point(item, (255, 0, 255), 4) for item in keypoints_field],
                    "lines": [_field_line(item) for item in lines_field],
                    "text_lines": text_lines,
                },
            ]]
        )

    writer.close()

    payload = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frames": frames,
    }
    with open(output_dir / "homography.json", "w", encoding="utf-8") as file:
        json.dump(convert_to_serializable(payload), file, indent=2, ensure_ascii=False)

    print(f"Video: {output_dir / 'homography.mp4'}")
    print(f"JSON: {output_dir / 'homography.json'}")


if __name__ == "__main__":
    main()
