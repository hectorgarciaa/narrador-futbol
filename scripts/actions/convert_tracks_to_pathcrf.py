from __future__ import annotations

import argparse
from pathlib import Path

from football_ai.actions import PathCRFAdapterConfig, convert_tracks_json_to_pathcrf


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

    config = PathCRFAdapterConfig(fps=float(args.fps))
    output_path, summary = convert_tracks_json_to_pathcrf(
        tracks_path=Path(args.tracks_path),
        output_path=Path(args.output_path) if args.output_path else None,
        config=config,
    )

    print(f"Parquet generado: {output_path}")
    print(f"Frames: {summary.frames}")
    print(f"Slots sintéticos home: {len(summary.synthetic_home_slots)}")
    print(f"Slots sintéticos away: {len(summary.synthetic_away_slots)}")
    print(f"Slots sintéticos referee: {len(summary.synthetic_referee_slots)}")


if __name__ == "__main__":
    main()
