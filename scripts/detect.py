import json
import sys
from pathlib import Path

from football_ai.tracking.phases.detection import Detector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import convert_to_serializable
from football_ai.visualization.simple_drawer import VideoOutput, VideoPanel

from scripts.utils import (
    build_video_model_parser,
    iter_video_frames,
    load_config,
    prepare_context,
)


def _video_item(trace_detection):
    return {
        "bbox": trace_detection["bbox_xyxy"],
        "color": trace_detection["render_color_bgr"],
        "class_name": trace_detection["class_name"],
        "confidence": trace_detection["confidence"],
        "extra_lines": [],
    }


def main():
    parser = build_video_model_parser("Detección YOLO simple")
    args = parser.parse_args()

    config = load_config(args.config)
    fps, width, height, total_frames, output_dir, video_path, model_path = prepare_context(args, config, "detect")

    writer = VideoOutput(
        output_dir / "detections.mp4",
        fps,
        [[VideoPanel((height, width))]],
    )
    
    detector = Detector(str(model_path), **config.detection)

    frames = {}
    for frame_index, frame_time_ms, frame_bgr in iter_video_frames(video_path, args.max_frames):
        detector_packet = detector.predict_frame(
            frame_bgr,
            frame_index=frame_index,
            frame_time_ms=frame_time_ms,
        )
        trace_detections = detector_packet["trace"]["detections"]
        writer.write_frame(
            [[{"frame": frame_bgr.copy(), "items": [_video_item(item) for item in trace_detections]}]]
        )
        frames[str(frame_index)] = detector_packet

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
    with open(output_dir / "detections.json", "w", encoding="utf-8") as file:
        json.dump(convert_to_serializable(payload), file, indent=2, ensure_ascii=False)

    print(f"Video: {output_dir / 'detections.mp4'}")
    print(f"JSON: {output_dir / 'detections.json'}")


if __name__ == "__main__":
    main()
