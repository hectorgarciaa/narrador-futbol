from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from football_ai.pathcrf_slot_mapping import person_slot_to_canonical_id, referee_slot_to_canonical_id


def _read_parquet_with_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "frame_id" in df.columns:
        return df.copy()
    out = df.reset_index()
    if "frame_id" in out.columns:
        return out
    if "index" in out.columns:
        return out.rename(columns={"index": "frame_id"})
    out["frame_id"] = range(len(out))
    return out


def _slot_to_canonical(slot: Any) -> str | None:
    if slot is None:
        return None
    slot_name = str(slot)
    person = person_slot_to_canonical_id(slot_name)
    if person is not None:
        return str(person)
    referee = referee_slot_to_canonical_id(slot_name)
    if referee is not None:
        return str(referee)
    return None


def _edge_row_to_canonical_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    src = _slot_to_canonical(row.get("edge_src"))
    dst = _slot_to_canonical(row.get("edge_dst"))
    if src is None or dst is None:
        src_track = row.get("edge_src_track_id")
        dst_track = row.get("edge_dst_track_id")
        if src is None and src_track is not None:
            src = str(src_track)
        if dst is None and dst_track is not None:
            dst = str(dst_track)
    if src is None or dst is None:
        return None
    return (str(src), str(dst))


@dataclass(frozen=True)
class EdgePoint:
    frame_id: int
    src: str
    dst: str


def _extract_edge_points(df: pd.DataFrame) -> list[EdgePoint]:
    points: list[EdgePoint] = []
    for row in df.to_dict(orient="records"):
        frame_id = row.get("frame_id")
        try:
            frame_id_int = int(frame_id)
        except (TypeError, ValueError):
            continue
        pair = _edge_row_to_canonical_pair(row)
        if pair is None:
            continue
        points.append(EdgePoint(frame_id=frame_id_int, src=pair[0], dst=pair[1]))
    return points


def _has_match(point: EdgePoint, reference: list[EdgePoint], tolerance_frames: int) -> bool:
    low = point.frame_id - tolerance_frames
    high = point.frame_id + tolerance_frames
    for candidate in reference:
        if candidate.src != point.src or candidate.dst != point.dst:
            continue
        if low <= candidate.frame_id <= high:
            return True
    return False


def _match_ratio(points: list[EdgePoint], reference: list[EdgePoint], tolerance_frames: int) -> float | None:
    if not points:
        return None
    matches = sum(1 for point in points if _has_match(point, reference, tolerance_frames))
    return float(matches) / float(len(points))


def _precision_recall_f1(pred: list[EdgePoint], gt: list[EdgePoint], tolerance_frames: int) -> dict[str, Any]:
    precision = _match_ratio(pred, gt, tolerance_frames)
    recall = _match_ratio(gt, pred, tolerance_frames)
    if precision is None or recall is None:
        return {"precision": precision, "recall": recall, "f1": None}
    if (precision + recall) <= 0.0:
        return {"precision": precision, "recall": recall, "f1": 0.0}
    f1 = 2.0 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _build_snapshot_internal_edges(summary_path: Path) -> pd.DataFrame:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = list(payload.get("snapshot_runs") or [])
    rows_by_frame: dict[int, dict[str, Any]] = {}
    for run in runs:
        edge_sequence_path = run.get("edge_sequence_path")
        if not edge_sequence_path:
            continue
        frame_id = run.get("frame_id")
        if frame_id is None:
            continue
        try:
            snapshot_frame = int(frame_id)
        except (TypeError, ValueError):
            continue
        edge_df = _read_parquet_with_frame(Path(edge_sequence_path))
        for item in edge_df.to_dict(orient="records"):
            local_frame = item.get("frame_id")
            try:
                absolute_frame = int(local_frame)
            except (TypeError, ValueError):
                # Fallback: al no haber frame_id válido nos quedamos con el frame del snapshot.
                absolute_frame = snapshot_frame
            rows_by_frame[absolute_frame] = {
                "frame_id": absolute_frame,
                "edge_src": item.get("edge_src"),
                "edge_dst": item.get("edge_dst"),
                "edge_team": item.get("edge_team"),
            }
    rows = [value for _, value in sorted(rows_by_frame.items(), key=lambda kv: kv[0])]
    return pd.DataFrame(rows, columns=["frame_id", "edge_src", "edge_dst", "edge_team"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa similitud de edges snapshot/incremental contra offline con tolerancia temporal."
        )
    )
    parser.add_argument("--offline-edge-path", type=Path, required=True, help="Parquet offline edge_sequence.")
    parser.add_argument(
        "--snapshot-summary-path",
        type=Path,
        required=True,
        help="live_snapshots_summary.json generado por run_live_snapshots_pathcrf.py.",
    )
    parser.add_argument(
        "--incremental-internal-edges-path",
        type=Path,
        required=True,
        help="internal_edges_per_frame.parquet generado por run_incremental_pathcrf.py.",
    )
    parser.add_argument("--tolerance-frames", type=int, default=5, help="Margen temporal ±N frames.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    offline_df = _read_parquet_with_frame(args.offline_edge_path.expanduser().resolve())
    snapshot_internal_df = _build_snapshot_internal_edges(args.snapshot_summary_path.expanduser().resolve())
    incremental_internal_df = _read_parquet_with_frame(args.incremental_internal_edges_path.expanduser().resolve())

    offline_points = _extract_edge_points(offline_df)
    snapshot_points = _extract_edge_points(snapshot_internal_df)
    incremental_points = _extract_edge_points(incremental_internal_df)

    tolerance = max(int(args.tolerance_frames), 0)
    snapshot_match = _match_ratio(snapshot_points, offline_points, tolerance)
    incremental_match = _match_ratio(incremental_points, offline_points, tolerance)
    snapshot_prf = _precision_recall_f1(snapshot_points, offline_points, tolerance)
    incremental_prf = _precision_recall_f1(incremental_points, offline_points, tolerance)

    summary = {
        "inputs": {
            "offline_edge_path": str(args.offline_edge_path),
            "snapshot_summary_path": str(args.snapshot_summary_path),
            "incremental_internal_edges_path": str(args.incremental_internal_edges_path),
            "tolerance_frames": tolerance,
        },
        "counts": {
            "offline_edges_canonical": len(offline_points),
            "snapshot_edges_canonical": len(snapshot_points),
            "incremental_edges_canonical": len(incremental_points),
        },
        "metrics": {
            "snapshot_vs_offline_edge_match_ratio": snapshot_match,
            "incremental_vs_offline_edge_match_ratio": incremental_match,
            "snapshot_vs_offline": snapshot_prf,
            "incremental_vs_offline": incremental_prf,
        },
    }

    summary_path = output_dir / "edge_similarity_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    snapshot_internal_df.to_parquet(output_dir / "snapshot_internal_edges_per_frame.parquet", index=False)
    print(f"Output dir: {output_dir}")
    print(f"Summary JSON: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

