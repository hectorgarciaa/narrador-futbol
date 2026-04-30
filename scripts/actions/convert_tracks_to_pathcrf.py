from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_ai.actions_incremental import ActionsDetector, ActionsDetectorConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convierte output/tracks.json al formato ancho compatible con PathCRF.",
    )
    parser.add_argument(
        "tracks_path",
        type=str,
        help="Ruta al tracks.json generado por scripts/track.py",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Ruta del parquet de salida. Si no se indica, se usa football_ai/actions/pathcrf/data/narrador/tracking_processed/<video>.parquet",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="FPS del vídeo usado para timestamps y derivadas.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    tracks_path = Path(args.tracks_path).expanduser().resolve()
    with tracks_path.open("r", encoding="utf-8") as f:
        tracks = json.load(f)

    detector = ActionsDetector(ActionsDetectorConfig(fps=float(args.fps)))
    frame_count = len(tracks.get("player", []))
    for frame_index in range(frame_count):
        detector.update(
            frame_index,
            {
                "tracks_frame": {
                    "player": tracks.get("player", [])[frame_index],
                    "goalkeeper": tracks.get("goalkeeper", [])[frame_index],
                    "referee": tracks.get("referee", [])[frame_index],
                    "ball": tracks.get("ball", [])[frame_index],
                },
                "possession": tracks.get("possession", [])[frame_index] if frame_index < len(tracks.get("possession", [])) else {},
            },
        )
    tracking_df, summary = detector.build_tracking_dataframe()
    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else tracks_path.with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_df.to_parquet(output_path, index=False)

    print(f"Parquet generado: {output_path}")
    print(f"Frames: {summary.frames}")
    print(f"Slots sintéticos home: {len(summary.synthetic_home_slots)}")
    print(f"Slots sintéticos away: {len(summary.synthetic_away_slots)}")
    print(f"Slots sintéticos referee: {len(summary.synthetic_referee_slots)}")


if __name__ == "__main__":
    main()
