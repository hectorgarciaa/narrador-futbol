#!/usr/bin/env python3

import itertools
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import convert_to_serializable
from football_ai.detection import Detector
from football_ai.visualization.simple_drawer import VideoOutput, VideoPanel, panel_color

from scripts._cli_common import (
    build_timestamp,
    build_video_model_parser,
    load_config,
    resolve_video_and_model_paths,
    slugify_model_name,
)


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


def main():
    parser = build_video_model_parser("Detección YOLO simple")
    args = parser.parse_args()

    config = load_config(args.config)
    video_path, model_path = resolve_video_and_model_paths(args, config)

    timestamp = build_timestamp()
    output_dir = PROJECT_ROOT / "output" / "detect" / slugify_model_name(model_path) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(str(model_path), **config.detection)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = VideoOutput(
        output_dir / "detections.mp4",
        fps,
        [[VideoPanel((height, width))]],
    )

    frames = {}
    results = detector.detect(str(video_path), stream=True)
    if args.max_frames is not None:
        results = itertools.islice(results, args.max_frames)
    for frame_index, result in enumerate(results):
        detections = _result_to_detections(result)
        panel_items = [
            {
                "bbox": detection["bbox"],
                "color": panel_color(detection["class"]),
                "class_name": detection["class"],
                "confidence": detection["confidence"],
            }
            for detection in detections
        ]
        writer.write_frame([[{"frame": result.orig_img.copy(), "items": panel_items}]])
        frames[str(frame_index)] = {"detections": detections}

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
