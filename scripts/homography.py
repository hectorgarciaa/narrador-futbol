import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import convert_to_serializable, default_render_color_bgr
from football_ai.tracking.phases.detection import DetectionPhase
from football_ai.tracking.phases.filtering import FilteringPhase
from football_ai.tracking.phases.reference_points import ProjectionPhase
from football_ai.tracking.phases.reference_points.geometry import project_image_points
from football_ai.visualization.simple_drawer import FieldPanel, VideoOutput, VideoPanel

from scripts.utils import (
    build_video_model_parser,
    iter_video_frames,
    load_config,
    prepare_context,
)

REJECTED_DETECTION_COLOR = [0, 0, 255]


def _trace_detections(packet):
    trace_detections = packet.get("trace", {}).get("detections")
    if trace_detections:
        return trace_detections

    clean = packet["clean"]
    return [
        {
            "det_id": int(det_id),
            "bbox_xyxy": list(bbox),
            "confidence": float(confidence),
            "class_name": str(class_name),
            "render_color_bgr": default_render_color_bgr(class_name),
        }
        for det_id, bbox, confidence, class_name in zip(
            clean["det_id"],
            clean["bbox_xyxy"],
            clean["confidence"],
            clean["class_name"],
        )
    ]


def _trace_filtering(packet, detector_packet):
    trace_detections = packet.get("trace", {}).get("accepted_detections")
    trace_rejections = packet.get("trace", {}).get("rejected_detections")
    if trace_detections is not None and trace_rejections is not None:
        merged = list(trace_detections) + list(trace_rejections)
        return sorted(merged, key=lambda item: int(item["det_id"]))

    clean = detector_packet["clean"]
    kept_ids = set(packet["clean"]["det_id"])
    field_positions = clean.get("field_positions_m", [])
    return [
        {
            "det_id": int(det_id),
            "bbox_xyxy": list(bbox),
            "confidence": float(confidence),
            "class_name": str(class_name),
            "field_position_m": list(field_position) if field_position is not None else None,
            "keep": int(det_id) in kept_ids,
            "is_finite_field_position": True,
            "inside_field": int(det_id) in kept_ids,
            "rescued_by_track_overlap": False,
            "max_iou_with_active_tracks": 0.0,
        }
        for det_id, bbox, confidence, class_name, field_position in zip(
            clean["det_id"],
            clean["bbox_xyxy"],
            clean["confidence"],
            clean["class_name"],
            field_positions,
        )
    ]


def _selected_attempt(reference_trace):
    diagnostics = reference_trace.get("diagnostics", {})
    selected_attempt_index = diagnostics.get("selected_attempt_index")
    if selected_attempt_index is None:
        return {}
    for attempt in reference_trace.get("attempts", []):
        if int(attempt.get("attempt_index", -1)) == int(selected_attempt_index):
            return attempt
    return {}


def _homography_text_lines(reference_packet, filtering_packet, frame_index):
    clean = reference_packet["clean"]
    reference_trace = reference_packet.get("trace", {})
    filtering_trace = filtering_packet.get("trace", {})
    selected_attempt = _selected_attempt(reference_trace)
    diagnostics = reference_trace.get("diagnostics", {})
    filtering_summary = filtering_trace.get("summary", {})
    quality_status = diagnostics.get("quality_status") or (
        "good" if clean.get("field_positions_usable_for_tracking") else "-"
    )
    quality_score = selected_attempt.get("quality_score", diagnostics.get("quality_score", 0.0))
    return [
        f"frame {frame_index}",
        f"homography: {'yes' if clean.get('homography_valid') else 'no'}",
        f"usable: {'yes' if clean.get('field_positions_usable_for_tracking') else 'no'}",
        f"kp: {len(reference_trace.get('keypoints', []))}",
        f"lineas: {len(reference_trace.get('lines', []))}",
        f"intentos: {len(reference_trace.get('attempts', []))}",
        f"estado: {quality_status}",
        f"score: {float(quality_score):.3f}",
        f"rechazo: {diagnostics.get('rejection_type') or '-'}",
        f"kept: {filtering_summary.get('total_kept', filtering_packet['clean']['num_detections'])}",
    ]


def _video_item(detector_trace_item, filtering_trace_item):
    keep = bool(filtering_trace_item.get("keep", False))
    extra_lines = []

    if not keep:
        if filtering_trace_item.get("rescued_by_track_overlap"):
            extra_lines.append("rescued_by_iou")
        elif filtering_trace_item.get("inside_field") is False:
            extra_lines.append("outside_field")
        elif filtering_trace_item.get("is_finite_field_position") is False:
            extra_lines.append("invalid_projection")

    return {
        "bbox": detector_trace_item["bbox_xyxy"],
        "color": detector_trace_item["render_color_bgr"] if keep else REJECTED_DETECTION_COLOR,
        "class_name": detector_trace_item["class_name"],
        "confidence": detector_trace_item["confidence"],
        "extra_lines": extra_lines,
    }


def _field_point(keypoint_trace_item):
    return {
        "position_m": keypoint_trace_item.get("field_position_m"),
        "color": (255, 0, 255),
        "canonical_id": None,
        "position_label": None,
        "radius": 4,
    }


def _field_line(line_trace_item):
    return {
        "point_1_m": line_trace_item.get("field_point_1_m"),
        "point_2_m": line_trace_item.get("field_point_2_m"),
        "color": (255, 255, 0),
        "thickness": 2,
    }


def _draw_reference_overlay(frame_bgr, reference_packet):
    overlay = frame_bgr.copy()
    trace = reference_packet.get("trace", {})
    for line in trace.get("lines", []):
        point_1 = line.get("image_point_1_px")
        point_2 = line.get("image_point_2_px")
        if point_1 is None or point_2 is None:
            continue
        p1 = (int(point_1[0]), int(point_1[1]))
        p2 = (int(point_2[0]), int(point_2[1]))
        cv2.line(overlay, p1, p2, (255, 255, 0), 2)

    for keypoint in trace.get("keypoints", []):
        point = keypoint.get("image_position_px")
        if point is None:
            continue
        center = (int(point[0]), int(point[1]))
        cv2.circle(overlay, center, 4, (255, 0, 255), -1)

    return overlay


def _visible_field_polygon(reference_packet):
    clean = reference_packet.get("clean", {})
    if not clean.get("homography_valid"):
        return None
    if not clean.get("field_positions_usable_for_tracking"):
        return None
    homography = np.asarray(clean.get("homography_image_to_field_3x3"), dtype=np.float64)
    if homography.shape != (3, 3):
        return None
    width = int(reference_packet.get("image_width") or 0)
    height = int(reference_packet.get("image_height") or 0)
    if width <= 0 or height <= 0:
        return None
    corners = np.asarray(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float32,
    )
    projected = project_image_points(corners, homography)
    if projected is None or len(projected) < 3:
        return None
    projected = np.asarray(projected, dtype=np.float32)
    finite = np.all(np.isfinite(projected), axis=1)
    if not np.any(finite):
        return None
    return projected[finite].tolist()


def _field_entity(det_trace_item):
    position = det_trace_item.get("field_position_m")
    if position is None:
        return None
    class_name = det_trace_item.get("class_name")
    keep = bool(det_trace_item.get("keep", False))
    color = (
        default_render_color_bgr(class_name)
        if keep
        else REJECTED_DETECTION_COLOR
    )
    radius = 4 if class_name == "ball" else 6
    return {
        "position_m": position,
        "color": color,
        "radius": radius,
        "canonical_id": None,
        "position_label": None,
    }


def main():
    parser = build_video_model_parser(
        "Detección + homografía PnLCalib",
        execution_mode_default="debug",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    fps, width, height, total_frames, output_dir, video_path, model_path = prepare_context(
        args,
        config,
        "homography",
    )

    writer = VideoOutput(
        output_dir / "homography.mp4",
        fps,
        [[VideoPanel((height, width)), FieldPanel((height, width))]],
    )

    detector_conf = dict(config.detection or {})
    detector_conf["model_path"] = str(model_path)
    detection_phase = DetectionPhase(detector_conf)

    projector_conf = dict(config.projector or {})
    projector_conf["project_root"] = config.project_root
    projection_phase = ProjectionPhase(projector_conf)
    filtering_phase = FilteringPhase()

    frames = {}
    for frame_index, frame_time_ms, frame_bgr in iter_video_frames(video_path, args.max_frames):
        detector_packet, detector_ms = detection_phase.process(
            frame_bgr,
            frame_index=frame_index,
            frame_time_ms=frame_time_ms,
            execution_mode=args.execution_mode,
        )
        reference_packet, projection_ms = projection_phase.process(
            frame_bgr,
            detector_packet,
            execution_mode=args.execution_mode,
        )
        filtering_packet, filtering_ms = filtering_phase.process(
            reference_packet,
            active_track_boxes_xyxy=[],
            geometry=projection_phase.geometry,
            execution_mode=args.execution_mode,
        )

        detector_trace = _trace_detections(detector_packet)
        filtering_trace = _trace_filtering(filtering_packet, reference_packet)
        filtering_by_det_id = {
            int(item["det_id"]): item
            for item in filtering_trace
        }

        shaded_polygon = _visible_field_polygon(reference_packet)
        writer.write_frame(
            [[
                {
                    "frame": _draw_reference_overlay(frame_bgr, reference_packet),
                    "items": [
                        _video_item(det_item, filtering_by_det_id.get(int(det_item["det_id"]), {}))
                        for det_item in detector_trace
                    ],
                },
                {
                    "field_size_m": (
                        projection_phase.geometry.field_length_m,
                        projection_phase.geometry.field_width_m,
                    ) if projection_phase.geometry is not None else (105.0, 68.0),
                    "shade_outside_polygon_m": shaded_polygon,
                    "entities": [
                        entity
                        for entity in (
                            _field_entity(item)
                            for item in filtering_trace
                        )
                        if entity is not None
                    ],
                    "points": [
                        _field_point(item)
                        for item in reference_packet.get("trace", {}).get("keypoints", [])
                    ],
                    "lines": [
                        _field_line(item)
                        for item in reference_packet.get("trace", {}).get("lines", [])
                    ],
                    "text_lines": _homography_text_lines(reference_packet, filtering_packet, frame_index),
                },
            ]]
        )

        frames[str(frame_index)] = {
            "detector": detector_packet,
            "reference_points": reference_packet,
            "filtering": filtering_packet,
            "profile_ms": {
                "detector_ms": detector_ms,
                "projection_ms": projection_ms,
                "filtering_ms": filtering_ms,
            },
        }

    writer.close()

    payload = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "execution_mode": args.execution_mode,
        "frames": frames,
    }
    with open(output_dir / "homography.json", "w", encoding="utf-8") as file:
        json.dump(convert_to_serializable(payload), file, indent=2, ensure_ascii=False)

    print(f"Video: {output_dir / 'homography.mp4'}")
    print(f"JSON: {output_dir / 'homography.json'}")


if __name__ == "__main__":
    main()
