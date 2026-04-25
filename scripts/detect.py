#!/usr/bin/env python3

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.core import Config
from football_ai.visualization.simple_drawer import SimpleDrawer


CLASS_MAP = {
    "player": "player",
    "person": "player",
    "referee": "referee",
    "ball": "ball",
    "sports ball": "ball",
    "sports_ball": "ball",
    "goalkeeper": "goalkeeper",
    "goalie": "goalkeeper",
    "keeper": "goalkeeper",
    "goal-keeper": "goalkeeper",
}


def resolve_path(config, group, value):
    try:
        return config.get_path("paths", group, value)
    except KeyError:
        return (PROJECT_ROOT / value).resolve()


def normalize_class(name):
    return CLASS_MAP.get(str(name).strip().lower().replace("-", " "))


def slugify_model(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(value).stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("model")
    args = parser.parse_args()

    config = Config.from_yaml(PROJECT_ROOT / "config.yaml")
    video_path = resolve_path(config, "data", args.video)
    model_path = resolve_path(config, "models", args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "output" / "detect" / slugify_model(args.model) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    drawer = SimpleDrawer()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_output_path = output_dir / "detections.mp4"
    json_output_path = output_dir / "detections.json"
    writer = cv2.VideoWriter(
        str(video_output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frames = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, verbose=False)[0]
        detections = []

        for box in result.boxes:
            raw_class = model.names[int(box.cls)]
            normalized_class = normalize_class(raw_class)
            if normalized_class is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                {
                    "class": normalized_class,
                    "raw_class": raw_class,
                    "confidence": float(box.conf),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        writer.write(drawer.draw(frame.copy(), detections))
        frames.append({"frame": frame_index, "detections": detections})
        frame_index += 1

    cap.release()
    writer.release()

    with open(json_output_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "video": str(video_path),
                "model": str(model_path),
                "output_dir": str(output_dir),
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "frames": frames,
            },
            file,
            indent=2,
        )

    print(f"Video: {video_output_path}")
    print(f"JSON: {json_output_path}")


if __name__ == "__main__":
    main()
