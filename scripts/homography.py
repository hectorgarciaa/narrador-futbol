import json
import sys
from pathlib import Path

from football_ai.detection.detector import Detector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import config, convert_to_serializable
from football_ai.filtering import filter_reference_points
from football_ai.reference_points import PnLCalibFieldProjector
from football_ai.visualization.simple_drawer import FieldPanel, VideoOutput, VideoPanel

from scripts.utils import (
    build_video_model_parser,
    iter_video_frames,
    load_config,
    prepare_context,
)

REJECTED_DETECTION_COLOR = [0, 0, 255]

def _selected_attempt(reference_trace):
    diagnostics = reference_trace["diagnostics"]
    selected_attempt_index = int(diagnostics["selected_attempt_index"])
    attempts = reference_trace["attempts"]
    for attempt in attempts:
        if int(attempt["attempt_index"]) == selected_attempt_index:
            return attempt
    return {}


def _homography_text_lines(reference_trace, filtering_trace, frame_index):
    selected_attempt = _selected_attempt(reference_trace)
    diagnostics = reference_trace["diagnostics"]
    return [
        f"frame {frame_index}",
        f"kp: {len(reference_trace['keypoints'])}",
        f"lineas: {len(reference_trace['lines'])}",
        f"intentos: {len(reference_trace['attempts'])}",
        f"estado: {selected_attempt.get('quality_status') or '-'}",
        f"score: {selected_attempt.get('quality_score', 0.0):.3f}",
        f"rechazo: {diagnostics['rejection_type'] or '-'}",
        f"kept: {filtering_trace['summary']['total_kept']}",
    ]


def _video_item(detector_trace_item, filtering_trace_item):
    keep = bool(filtering_trace_item.get("keep", False))
    extra_lines = []
    reject_label = filtering_trace_item.get("reject_label")
    if reject_label and reject_label != "kept":
        extra_lines.append(reject_label)
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


def main():
    parser = build_video_model_parser("Detección + homografía PnLCalib")
    args = parser.parse_args()
    
    config = load_config(args.config)

    fps, width, height, total_frames, output_dir, video_path, model_path = prepare_context(args, config, "homography")

    writer = VideoOutput(
        output_dir / "homography.mp4",
        fps,
        [[VideoPanel((height, width)), FieldPanel((height, width))]],
    )

    detector = Detector(str(model_path), **config.detection)
    
    projector_conf = dict(config.projector.get("constructor", {}))
    projector = PnLCalibFieldProjector(project_root=config.project_root, **projector_conf)

    frames = {}
    for frame_index, frame_time_ms, frame_bgr in iter_video_frames(video_path, args.max_frames):
        detector_packet = detector.predict_frame(
            frame_bgr,
            frame_index=frame_index,
            frame_time_ms=frame_time_ms,
        )
        reference_packet = projector.project_frame(frame_bgr, detector_packet)
        filtering_packet = filter_reference_points(
            reference_packet,
            active_track_boxes_xyxy=[],
            geometry=projector.geometry,
            field_length_m=projector.geometry.field_length_m,
            field_width_m=projector.geometry.field_width_m,
        )

        detector_trace = detector_packet["trace"]
        reference_trace = reference_packet["trace"]
        filtering_trace = filtering_packet["trace"]
        trace_detections = detector_trace["detections"]
        trace_filtering = filtering_trace["detections"]

        writer.write_frame(
            [[
                {
                    "frame": frame_bgr.copy(),
                    "items": [
                        _video_item(det_item, filter_item)
                        for det_item, filter_item in zip(trace_detections, trace_filtering)
                    ],
                },
                {
                    "field_size_m": (
                        projector.geometry.field_length_m,
                        projector.geometry.field_width_m,
                    ),
                    "entities": [],
                    "points": [_field_point(item) for item in reference_trace["keypoints"]],
                    "lines": [_field_line(item) for item in reference_trace["lines"]],
                    "text_lines": _homography_text_lines(reference_trace, filtering_trace, frame_index),
                },
            ]]
        )

        frames[str(frame_index)] = {
            "detector": detector_packet,
            "reference_points": reference_packet,
            "filtering": filtering_packet,
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
        "frames": frames,
    }
    with open(output_dir / "homography.json", "w", encoding="utf-8") as file:
        json.dump(convert_to_serializable(payload), file, indent=2, ensure_ascii=False)

    print(f"Video: {output_dir / 'homography.mp4'}")
    print(f"JSON: {output_dir / 'homography.json'}")


if __name__ == "__main__":
    main()
