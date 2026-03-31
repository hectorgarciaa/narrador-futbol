#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _coerce_reason(det: dict) -> str:
    reason = (det or {}).get("discard_reason")
    if reason is None:
        return "unknown"
    reason = str(reason).strip()
    return reason or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza *_debug_frames.json y resume por qué las detecciones devueltas por "
            "ByteTrack no se mapearon a IDs canónicos."
        )
    )
    parser.add_argument("debug_frames_json", type=str, help="Ruta a <video>_debug_frames.json")
    parser.add_argument("--top", type=int, default=25, help="Número de filas (frame, motivo) a mostrar")
    args = parser.parse_args()

    path = Path(args.debug_frames_json).expanduser()
    with path.open("r", encoding="utf-8") as f:
        frames = json.load(f)

    total_not_tracked = 0
    total_tracked_no_canonical = 0
    reasons = Counter()
    by_frame = defaultdict(Counter)

    for frame_id, frame in enumerate(frames or []):
        not_tracked = (frame or {}).get("discarded_yolo_not_tracked", []) or []
        tracked_no_canonical = (frame or {}).get("discarded_bytetrack_not_canonical", []) or []
        total_not_tracked += len(not_tracked)
        total_tracked_no_canonical += len(tracked_no_canonical)
        for det in tracked_no_canonical:
            r = _coerce_reason(det)
            reasons[r] += 1
            by_frame[frame_id][r] += 1

    print(f"Archivo: {path}")
    print(f"Frames: {len(frames or [])}")
    print(f"YOLO descartadas (no trackeadas por ByteTrack): {total_not_tracked}")
    print(f"ByteTrack pero sin canónico: {total_tracked_no_canonical}")
    print("")

    print("Top motivos (ByteTrack -> no canónico):")
    for reason, count in reasons.most_common(30):
        print(f"- {reason}: {count}")

    # Show the noisiest frames
    noisy = []
    for frame_id, c in by_frame.items():
        total = sum(c.values())
        if total <= 0:
            continue
        top_reason, top_count = c.most_common(1)[0]
        noisy.append((total, top_count, top_reason, frame_id))
    noisy.sort(reverse=True)

    print("")
    print(f"Frames con más descartes canónicos (top {args.top}):")
    for total, top_count, top_reason, frame_id in noisy[: max(0, int(args.top))]:
        print(f"- frame {frame_id}: {total} (top: {top_reason} x{top_count})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

