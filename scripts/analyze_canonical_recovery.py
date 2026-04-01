#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _frame_count(tracks: dict[str, list[dict[str, Any]]]) -> int:
    for value in tracks.values():
        if isinstance(value, list):
            return len(value)
    return 0


def _track_present(
    frame_map: dict[str, Any],
    track_id: int,
    *,
    exclude_synthetic_seed: bool,
) -> bool:
    payload = frame_map.get(str(track_id))
    if not isinstance(payload, dict):
        return False
    if exclude_synthetic_seed and payload.get("synthetic_seed"):
        return False
    return True


def _contiguous_segments(frame_indexes: list[int]) -> list[tuple[int, int]]:
    if not frame_indexes:
        return []
    segments: list[tuple[int, int]] = []
    start = frame_indexes[0]
    prev = frame_indexes[0]
    for frame_id in frame_indexes[1:]:
        if frame_id == prev + 1:
            prev = frame_id
            continue
        segments.append((start, prev))
        start = frame_id
        prev = frame_id
    segments.append((start, prev))
    return segments


def _summarize_lengths(lengths: list[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    ordered = sorted(lengths)
    p90_index = int(round(0.9 * (len(ordered) - 1)))
    return {
        "count": len(ordered),
        "mean": round(statistics.mean(ordered), 3),
        "median": float(statistics.median(ordered)),
        "p90": int(ordered[p90_index]),
        "max": int(ordered[-1]),
    }


def analyze_canonical_tracks(
    tracks: dict[str, list[dict[str, Any]]],
    *,
    classes: list[str],
    exclude_synthetic_seed: bool,
) -> dict[str, Any]:
    total_frames = _frame_count(tracks)
    result: dict[str, Any] = {
        "total_frames": total_frames,
        "classes": {},
    }

    for class_name in classes:
        class_frames = tracks.get(class_name)
        if not isinstance(class_frames, list):
            continue

        track_ids = sorted(
            {
                int(track_id)
                for frame_map in class_frames
                if isinstance(frame_map, dict)
                for track_id in frame_map.keys()
            }
        )
        gaps: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        per_track: list[dict[str, Any]] = []

        for track_id in track_ids:
            visible_frames = [
                frame_id
                for frame_id, frame_map in enumerate(class_frames)
                if isinstance(frame_map, dict)
                and _track_present(
                    frame_map,
                    track_id,
                    exclude_synthetic_seed=exclude_synthetic_seed,
                )
            ]
            if not visible_frames:
                continue

            segments = _contiguous_segments(visible_frames)
            track_gap_count = 0
            for previous_segment, next_segment in zip(segments, segments[1:]):
                gap_start = previous_segment[1] + 1
                gap_end = next_segment[0] - 1
                gap_length = gap_end - gap_start + 1
                gaps.append(
                    {
                        "track_id": track_id,
                        "lost_after_frame": previous_segment[1],
                        "recovered_at_frame": next_segment[0],
                        "gap_start": gap_start,
                        "gap_end": gap_end,
                        "gap_length_frames": gap_length,
                    }
                )
                track_gap_count += 1

            last_visible_frame = segments[-1][1]
            unresolved_tail = max(0, total_frames - 1 - last_visible_frame)
            if unresolved_tail > 0:
                unresolved.append(
                    {
                        "track_id": track_id,
                        "lost_after_frame": last_visible_frame,
                        "gap_start": last_visible_frame + 1,
                        "gap_end": total_frames - 1,
                        "gap_length_frames": unresolved_tail,
                    }
                )

            per_track.append(
                {
                    "track_id": track_id,
                    "first_visible_frame": segments[0][0],
                    "last_visible_frame": last_visible_frame,
                    "visible_frames": len(visible_frames),
                    "visible_segments": len(segments),
                    "recovered_gap_count": track_gap_count,
                    "has_unresolved_loss": unresolved_tail > 0,
                    "unresolved_gap_length_frames": unresolved_tail,
                }
            )

        gap_lengths = [gap["gap_length_frames"] for gap in gaps]
        unresolved_lengths = [gap["gap_length_frames"] for gap in unresolved]
        result["classes"][class_name] = {
            "canonical_ids_seen": len(per_track),
            "recovered_loss_events": len(gaps),
            "recovered_gap_stats": _summarize_lengths(gap_lengths),
            "unrecovered_tracks": len(unresolved),
            "unrecovered_gap_stats": _summarize_lengths(unresolved_lengths),
            "all_recovered_gaps": sorted(
                gaps,
                key=lambda item: (
                    item["gap_length_frames"],
                    item["track_id"],
                ),
                reverse=True,
            ),
            "top_recovered_gaps": sorted(
                gaps,
                key=lambda item: (
                    item["gap_length_frames"],
                    item["track_id"],
                ),
                reverse=True,
            )[:10],
            "unrecovered_tracks_detail": sorted(
                unresolved,
                key=lambda item: (
                    item["gap_length_frames"],
                    item["track_id"],
                ),
                reverse=True,
            ),
            "per_track": sorted(per_track, key=lambda item: item["track_id"]),
        }

    return result


def analyze_orphan_bytetrack_sequences(
    debug_frames: list[dict[str, Any]],
    *,
    min_sequence_length: int,
) -> dict[str, Any]:
    sequences: list[dict[str, Any]] = []
    active_sequences: dict[int, dict[str, Any]] = {}
    reason_counter: Counter[str] = Counter()

    for frame_id, frame in enumerate(debug_frames):
        detections = (frame or {}).get("discarded_bytetrack_not_canonical") or []
        current_by_track: dict[int, list[dict[str, Any]]] = {}
        for detection in detections:
            raw_track_id = detection.get("bytetrack_id")
            if raw_track_id is None:
                continue
            raw_track_id = int(raw_track_id)
            current_by_track.setdefault(raw_track_id, []).append(detection)
            discard_reason = str(detection.get("discard_reason") or "unknown")
            reason_counter[discard_reason] += 1

        for raw_track_id in list(active_sequences.keys()):
            if raw_track_id not in current_by_track:
                sequences.append(active_sequences.pop(raw_track_id))

        for raw_track_id, detections_for_track in current_by_track.items():
            if (
                raw_track_id in active_sequences
                and active_sequences[raw_track_id]["end_frame"] == frame_id - 1
            ):
                sequence = active_sequences[raw_track_id]
                sequence["end_frame"] = frame_id
                sequence["length_frames"] += 1
            else:
                sequence = {
                    "raw_track_id": raw_track_id,
                    "start_frame": frame_id,
                    "end_frame": frame_id,
                    "length_frames": 1,
                    "class_counter": Counter(),
                    "reason_counter": Counter(),
                }
                active_sequences[raw_track_id] = sequence

            for detection in detections_for_track:
                class_name = (
                    detection.get("class_tracker")
                    or detection.get("class_relabel")
                    or detection.get("class_yolo")
                    or "unknown"
                )
                discard_reason = str(detection.get("discard_reason") or "unknown")
                sequence["class_counter"][str(class_name)] += 1
                sequence["reason_counter"][discard_reason] += 1

    sequences.extend(active_sequences.values())

    normalized_sequences = []
    for sequence in sequences:
        normalized_sequences.append(
            {
                "raw_track_id": sequence["raw_track_id"],
                "start_frame": sequence["start_frame"],
                "end_frame": sequence["end_frame"],
                "length_frames": sequence["length_frames"],
                "dominant_class": sequence["class_counter"].most_common(1)[0][0],
                "dominant_reason": sequence["reason_counter"].most_common(1)[0][0],
                "class_counter": dict(sequence["class_counter"]),
                "reason_counter": dict(sequence["reason_counter"]),
            }
        )

    long_sequences = [
        sequence
        for sequence in normalized_sequences
        if sequence["length_frames"] >= min_sequence_length
    ]
    return {
        "total_discarded_bytetrack_not_canonical": int(sum(reason_counter.values())),
        "discard_reason_counter": dict(reason_counter.most_common()),
        "total_sequences": len(normalized_sequences),
        "sequences_at_least_min_length": len(long_sequences),
        "top_long_sequences": sorted(
            long_sequences,
            key=lambda item: (
                item["length_frames"],
                item["raw_track_id"],
            ),
            reverse=True,
        )[:20],
        "all_sequences": sorted(
            normalized_sequences,
            key=lambda item: (
                item["length_frames"],
                item["raw_track_id"],
            ),
            reverse=True,
        ),
    }


def enrich_with_orphan_overlap(
    report: dict[str, Any],
    *,
    min_orphan_sequence_length: int,
) -> None:
    orphan_report = report.get("orphan_bytetrack")
    if not orphan_report:
        return

    all_sequences = orphan_report.get("all_sequences", [])
    for class_name, class_report in report.get("classes", {}).items():
        class_sequences = [
            sequence
            for sequence in all_sequences
            if sequence.get("dominant_class") == class_name
            and int(sequence.get("length_frames", 0)) >= min_orphan_sequence_length
        ]

        def _annotate(gaps: list[dict[str, Any]]) -> dict[str, Any]:
            overlap_count = 0
            annotated = []
            for gap in gaps:
                gap_start = int(gap["gap_start"])
                gap_end = int(gap["gap_end"])
                overlaps = []
                for sequence in class_sequences:
                    overlap_frames = max(
                        0,
                        min(gap_end, int(sequence["end_frame"]))
                        - max(gap_start, int(sequence["start_frame"]))
                        + 1,
                    )
                    if overlap_frames <= 0:
                        continue
                    overlaps.append(
                        {
                            "raw_track_id": int(sequence["raw_track_id"]),
                            "overlap_frames": overlap_frames,
                            "sequence_length_frames": int(sequence["length_frames"]),
                            "sequence_start_frame": int(sequence["start_frame"]),
                            "sequence_end_frame": int(sequence["end_frame"]),
                            "dominant_reason": sequence["dominant_reason"],
                        }
                    )
                overlaps.sort(
                    key=lambda item: (
                        item["overlap_frames"],
                        item["sequence_length_frames"],
                        item["raw_track_id"],
                    ),
                    reverse=True,
                )
                if overlaps:
                    overlap_count += 1
                enriched_gap = dict(gap)
                enriched_gap["overlapping_orphan_sequences"] = overlaps[:5]
                annotated.append(enriched_gap)
            return {
                "gaps_with_orphan_overlap": overlap_count,
                "total_gaps": len(gaps),
                "examples": sorted(
                    annotated,
                    key=lambda item: (
                        item["overlapping_orphan_sequences"][0]["overlap_frames"]
                        if item["overlapping_orphan_sequences"]
                        else -1,
                        item["gap_length_frames"],
                    ),
                    reverse=True,
                )[:10],
            }

        class_report["orphan_overlap"] = {
            "minimum_orphan_sequence_length": min_orphan_sequence_length,
            "recovered": _annotate(class_report.get("all_recovered_gaps", [])),
            "unrecovered": _annotate(class_report.get("unrecovered_tracks_detail", [])),
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estudia pérdidas y recuperaciones de IDs canónicos y detecta secuencias "
            "de ByteTrack descartadas por la capa canónica."
        )
    )
    parser.add_argument("tracks_json", help="Ruta a <video>_tracks.json")
    parser.add_argument(
        "--debug-frames-json",
        help="Ruta a <video>_debug_frames.json para analizar ByteTrack no absorbido",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["player", "referee", "goalkeeper"],
        help="Clases canónicas a estudiar",
    )
    parser.add_argument(
        "--include-synthetic-seed",
        action="store_true",
        help="Incluye frames sintéticos de seeds reservadas en el conteo de presencia",
    )
    parser.add_argument(
        "--min-orphan-sequence-length",
        type=int,
        default=3,
        help="Longitud mínima para destacar secuencias de ByteTrack no absorbidas",
    )
    parser.add_argument(
        "--output-json",
        help="Si se indica, guarda el informe completo en JSON",
    )
    args = parser.parse_args()

    tracks_path = Path(args.tracks_json).expanduser()
    tracks = _load_json(tracks_path)
    report = analyze_canonical_tracks(
        tracks,
        classes=list(args.classes),
        exclude_synthetic_seed=not args.include_synthetic_seed,
    )
    report["tracks_json"] = str(tracks_path)
    report["exclude_synthetic_seed"] = not args.include_synthetic_seed

    if args.debug_frames_json:
        debug_frames_path = Path(args.debug_frames_json).expanduser()
        debug_frames = _load_json(debug_frames_path)
        report["debug_frames_json"] = str(debug_frames_path)
        report["orphan_bytetrack"] = analyze_orphan_bytetrack_sequences(
            debug_frames,
            min_sequence_length=max(1, int(args.min_orphan_sequence_length)),
        )
        enrich_with_orphan_overlap(
            report,
            min_orphan_sequence_length=max(1, int(args.min_orphan_sequence_length)),
        )

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Informe para: {tracks_path}")
    print(f"Frames totales: {report['total_frames']}")
    print(f"Excluir synthetic_seed: {report['exclude_synthetic_seed']}")
    print("")

    for class_name, class_report in report["classes"].items():
        recovered = class_report["recovered_loss_events"]
        unresolved = class_report["unrecovered_tracks"]
        recovered_stats = class_report["recovered_gap_stats"]
        unresolved_stats = class_report["unrecovered_gap_stats"]
        print(f"[{class_name}]")
        print(f"- IDs canónicos observados: {class_report['canonical_ids_seen']}")
        print(f"- Pérdidas recuperadas: {recovered}")
        print(
            "- Gap recuperado (frames): "
            f"media={recovered_stats['mean']} mediana={recovered_stats['median']} "
            f"p90={recovered_stats['p90']} max={recovered_stats['max']}"
        )
        print(f"- Tracks perdidos y no recuperados: {unresolved}")
        print(
            "- Gap no recuperado (frames): "
            f"media={unresolved_stats['mean']} mediana={unresolved_stats['median']} "
            f"p90={unresolved_stats['p90']} max={unresolved_stats['max']}"
        )
        print("- Top gaps recuperados:")
        for gap in class_report["top_recovered_gaps"][:5]:
            print(
                "  "
                f"id={gap['track_id']} lost_after={gap['lost_after_frame']} "
                f"recovered_at={gap['recovered_at_frame']} "
                f"gap={gap['gap_length_frames']}"
            )
        print("- No recuperados:")
        for gap in class_report["unrecovered_tracks_detail"][:10]:
            print(
                "  "
                f"id={gap['track_id']} lost_after={gap['lost_after_frame']} "
                f"gap={gap['gap_length_frames']}"
            )
        overlap_report = class_report.get("orphan_overlap")
        if overlap_report:
            recovered_overlap = overlap_report["recovered"]
            unrecovered_overlap = overlap_report["unrecovered"]
            print(
                "- Solape con secuencias ByteTrack no absorbidas "
                f"(len>={overlap_report['minimum_orphan_sequence_length']}): "
                f"recuperados={recovered_overlap['gaps_with_orphan_overlap']}/"
                f"{recovered_overlap['total_gaps']} "
                f"no_recuperados={unrecovered_overlap['gaps_with_orphan_overlap']}/"
                f"{unrecovered_overlap['total_gaps']}"
            )
        print("")

    orphan_report = report.get("orphan_bytetrack")
    if orphan_report:
        print("[ByteTrack no absorbido]")
        print(
            "- Detecciones ByteTrack sin canónico: "
            f"{orphan_report['total_discarded_bytetrack_not_canonical']}"
        )
        print(f"- Secuencias totales: {orphan_report['total_sequences']}")
        print(
            "- Secuencias largas: "
            f"{orphan_report['sequences_at_least_min_length']}"
        )
        print("- Motivos principales:")
        for reason, count in list(orphan_report["discard_reason_counter"].items())[:5]:
            print(f"  {reason}: {count}")
        print("- Secuencias largas principales:")
        for sequence in orphan_report["top_long_sequences"][:10]:
            print(
                "  "
                f"raw_id={sequence['raw_track_id']} "
                f"frames={sequence['start_frame']}-{sequence['end_frame']} "
                f"len={sequence['length_frames']} "
                f"class={sequence['dominant_class']} "
                f"reason={sequence['dominant_reason']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
