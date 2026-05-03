from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from football_ai.pathcrf_slot_mapping import person_slot_to_canonical_id, referee_slot_to_canonical_id


@dataclass(frozen=True)
class EdgePoint:
    frame_id: int
    src: str
    dst: str
    raw_src: str | None
    raw_dst: str | None
    source: str
    checkpoint_frame: int | None = None
    batch_local_index: int | None = None


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


def _canonical_pair_from_row(row: dict[str, Any]) -> tuple[str, str] | None:
    src = _slot_to_canonical(row.get("edge_src"))
    dst = _slot_to_canonical(row.get("edge_dst"))
    if src is None:
        src_track = row.get("edge_src_track_id")
        if src_track is not None:
            src = str(src_track)
    if dst is None:
        dst_track = row.get("edge_dst_track_id")
        if dst_track is not None:
            dst = str(dst_track)
    if src is None or dst is None:
        return None
    return (str(src), str(dst))


def _raw_pair_from_row(row: dict[str, Any]) -> tuple[str, str] | None:
    src = row.get("edge_src")
    dst = row.get("edge_dst")
    if src is None or dst is None:
        return None
    return (str(src), str(dst))


def _build_points_from_df(df: pd.DataFrame, source: str) -> tuple[list[EdgePoint], list[EdgePoint]]:
    canonical_points: list[EdgePoint] = []
    raw_points: list[EdgePoint] = []
    for row in df.to_dict(orient="records"):
        frame = row.get("frame_id")
        try:
            frame_id = int(frame)
        except (TypeError, ValueError):
            continue
        raw_pair = _raw_pair_from_row(row)
        canonical_pair = _canonical_pair_from_row(row)
        if raw_pair is not None:
            raw_points.append(
                EdgePoint(
                    frame_id=frame_id,
                    src=raw_pair[0],
                    dst=raw_pair[1],
                    raw_src=raw_pair[0],
                    raw_dst=raw_pair[1],
                    source=source,
                )
            )
        if canonical_pair is not None:
            canonical_points.append(
                EdgePoint(
                    frame_id=frame_id,
                    src=canonical_pair[0],
                    dst=canonical_pair[1],
                    raw_src=raw_pair[0] if raw_pair else None,
                    raw_dst=raw_pair[1] if raw_pair else None,
                    source=source,
                )
            )
    return canonical_points, raw_points


def _load_snapshot_internal(snapshot_summary_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    payload = json.loads(snapshot_summary_path.read_text(encoding="utf-8"))
    runs = list(payload.get("snapshot_runs") or [])
    by_frame: dict[int, dict[str, Any]] = {}
    for run in runs:
        edge_path = run.get("edge_sequence_path")
        if not edge_path:
            continue
        edge_df = _read_parquet_with_frame(Path(edge_path))
        for item in edge_df.to_dict(orient="records"):
            frame = item.get("frame_id")
            try:
                fid = int(frame)
            except (TypeError, ValueError):
                continue
            by_frame[fid] = {
                "frame_id": fid,
                "edge_src": item.get("edge_src"),
                "edge_dst": item.get("edge_dst"),
                "edge_team": item.get("edge_team"),
            }
    rows = [row for _, row in sorted(by_frame.items(), key=lambda kv: kv[0])]
    return pd.DataFrame(rows), runs


def _load_incremental_checkpoint_rows(runtime_checkpoints_path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    checkpoints = json.loads(runtime_checkpoints_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for ck in checkpoints:
        checkpoint_frame = ck.get("frame_id")
        for idx, edge in enumerate(list(ck.get("raw_edge_batch") or [])):
            rows.append(
                {
                    "frame_id": edge.get("frame_id"),
                    "checkpoint_frame": checkpoint_frame,
                    "batch_local_index": idx,
                    "edge_src": edge.get("edge_src"),
                    "edge_dst": edge.get("edge_dst"),
                    "edge_src_track_id": edge.get("edge_src_track_id"),
                    "edge_dst_track_id": edge.get("edge_dst_track_id"),
                    "edge_team": edge.get("edge_team"),
                }
            )
    return pd.DataFrame(rows), checkpoints


def _filter_points(points: list[EdgePoint], start: int, end: int) -> list[EdgePoint]:
    return [p for p in points if start <= p.frame_id <= end]


def _find_match_index(
    pred: EdgePoint,
    refs: list[EdgePoint],
    tolerance: int,
    used_ref_indices: set[int] | None = None,
) -> int | None:
    lo = pred.frame_id - tolerance
    hi = pred.frame_id + tolerance
    best_idx: int | None = None
    best_dist: int | None = None
    for i, ref in enumerate(refs):
        if used_ref_indices is not None and i in used_ref_indices:
            continue
        if pred.src != ref.src or pred.dst != ref.dst:
            continue
        if not (lo <= ref.frame_id <= hi):
            continue
        dist = abs(ref.frame_id - pred.frame_id)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _evaluate(pred: list[EdgePoint], ref: list[EdgePoint], tolerance: int) -> dict[str, Any]:
    if not pred and not ref:
        return {
            "num_pred_edges": 0,
            "num_ref_edges": 0,
            "num_matches": 0,
            "num_unmatched_predictions": 0,
            "num_unmatched_references": 0,
            "edge_match_ratio": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
    used_ref: set[int] = set()
    match_count = 0
    for p in pred:
        idx = _find_match_index(p, ref, tolerance, used_ref)
        if idx is not None:
            used_ref.add(idx)
            match_count += 1
    precision = (match_count / len(pred)) if pred else None
    recall = (match_count / len(ref)) if ref else None
    if precision is None or recall is None or (precision + recall) <= 0.0:
        f1 = 0.0 if precision == 0.0 or recall == 0.0 else None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "num_pred_edges": len(pred),
        "num_ref_edges": len(ref),
        "num_matches": int(match_count),
        "num_unmatched_predictions": int(len(pred) - match_count),
        "num_unmatched_references": int(len(ref) - match_count),
        "edge_match_ratio": precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _evaluate_with_offset(pred: list[EdgePoint], ref: list[EdgePoint], tolerance: int, offset: int) -> dict[str, Any]:
    shifted = [
        EdgePoint(
            frame_id=p.frame_id + offset,
            src=p.src,
            dst=p.dst,
            raw_src=p.raw_src,
            raw_dst=p.raw_dst,
            source=p.source,
            checkpoint_frame=p.checkpoint_frame,
            batch_local_index=p.batch_local_index,
        )
        for p in pred
    ]
    return _evaluate(shifted, ref, tolerance)


def _edge_type(point: EdgePoint) -> str:
    raw_u = str(point.raw_src or "")
    raw_v = str(point.raw_dst or "")
    if raw_u.startswith("out_") or raw_v.startswith("out_"):
        return "player-out"
    if point.src.isdigit() and point.dst.isdigit():
        return "player-player"
    return "unknown/missing"


def _pair_key(point: EdgePoint) -> tuple[str, str]:
    return (point.src, point.dst)


def _top_counter_items(counter: Counter[tuple[str, str]], top_n: int = 5) -> list[str]:
    return [f"{u}->{v}:{c}" for (u, v), c in counter.most_common(top_n)]


def _build_mapping_coverage(checkpoints: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slot_to_canonical_by_time: defaultdict[str, list[str]] = defaultdict(list)
    slot_to_track_by_time: defaultdict[str, list[str]] = defaultdict(list)
    canonical_to_slots: defaultdict[str, set[str]] = defaultdict(set)
    for ck in checkpoints:
        checkpoint_frame = ck.get("frame_id")
        summary = ck.get("summary") or {}
        person_assignments = dict(summary.get("person_slot_assignments") or {})
        referee_assignments = dict(summary.get("referee_slot_assignments") or {})
        slot_to_track: dict[str, str] = {}
        for track, slot in {**person_assignments, **referee_assignments}.items():
            slot_to_track[str(slot)] = str(track)
        for idx, edge in enumerate(list(ck.get("raw_edge_batch") or [])):
            slot_u = edge.get("edge_src")
            slot_v = edge.get("edge_dst")
            can_u = _slot_to_canonical(slot_u)
            can_v = _slot_to_canonical(slot_v)
            if can_u is not None and slot_u is not None:
                slot_to_canonical_by_time[str(slot_u)].append(can_u)
                canonical_to_slots[can_u].add(str(slot_u))
            if can_v is not None and slot_v is not None:
                slot_to_canonical_by_time[str(slot_v)].append(can_v)
                canonical_to_slots[can_v].add(str(slot_v))
            if slot_u is not None and str(slot_u) in slot_to_track:
                slot_to_track_by_time[str(slot_u)].append(slot_to_track[str(slot_u)])
            if slot_v is not None and str(slot_v) in slot_to_track:
                slot_to_track_by_time[str(slot_v)].append(slot_to_track[str(slot_v)])
            valid_u = can_u is not None
            valid_v = can_v is not None
            rows.append(
                {
                    "frame": edge.get("frame_id"),
                    "checkpoint_frame": checkpoint_frame,
                    "batch_local_index": idx,
                    "slot_u": slot_u,
                    "slot_v": slot_v,
                    "canonical_u": can_u,
                    "canonical_v": can_v,
                    "edge_raw": f"{slot_u}->{slot_v}",
                    "edge_canonical": f"{can_u}->{can_v}" if valid_u and valid_v else None,
                    "mapping_valid_u": valid_u,
                    "mapping_valid_v": valid_v,
                    "mapping_valid_edge": bool(valid_u and valid_v),
                }
            )
    df = pd.DataFrame(rows)
    total = len(df)
    valid_u = int(df["mapping_valid_u"].sum()) if total else 0
    valid_v = int(df["mapping_valid_v"].sum()) if total else 0
    valid_edge = int(df["mapping_valid_edge"].sum()) if total else 0
    slot_canonical_changes = 0
    slot_track_changes = 0
    for _, seq in slot_to_canonical_by_time.items():
        slot_canonical_changes += sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    for _, seq in slot_to_track_by_time.items():
        slot_track_changes += sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    summary = {
        "rows": total,
        "pct_slots_without_canonical_id_u": (1.0 - (valid_u / total)) if total else None,
        "pct_slots_without_canonical_id_v": (1.0 - (valid_v / total)) if total else None,
        "pct_edges_with_any_unmapped_endpoint": (1.0 - (valid_edge / total)) if total else None,
        "pct_edges_with_both_endpoints_mapped": (valid_edge / total) if total else None,
        "num_canonical_none_u": int((df["canonical_u"].isna()).sum()) if total else 0,
        "num_canonical_none_v": int((df["canonical_v"].isna()).sum()) if total else 0,
        "slot_to_canonical_change_count": int(slot_canonical_changes),
        "slot_to_track_change_count": int(slot_track_changes),
        "canonical_id_with_multiple_slots": int(sum(1 for slots in canonical_to_slots.values() if len(slots) > 1)),
    }
    return df, summary


def _segment_bounds(start: int, end: int) -> list[tuple[int, int]]:
    segments = [(40, 99), (100, 199), (200, 299), (300, 399), (400, 499), (500, 599), (600, 699), (700, 724)]
    result: list[tuple[int, int]] = []
    for a, b in segments:
        lo = max(a, start)
        hi = min(b, end)
        if lo <= hi:
            result.append((lo, hi))
    return result


def _build_pair_csv(pred: list[EdgePoint], ref: list[EdgePoint], tolerance: int) -> pd.DataFrame:
    pred_by_pair = Counter(_pair_key(p) for p in pred)
    ref_by_pair = Counter(_pair_key(r) for r in ref)
    pairs = sorted(set(pred_by_pair.keys()) | set(ref_by_pair.keys()))
    rows: list[dict[str, Any]] = []
    pred_by_pair_points: defaultdict[tuple[str, str], list[EdgePoint]] = defaultdict(list)
    ref_by_pair_points: defaultdict[tuple[str, str], list[EdgePoint]] = defaultdict(list)
    for p in pred:
        pred_by_pair_points[_pair_key(p)].append(p)
    for r in ref:
        ref_by_pair_points[_pair_key(r)].append(r)
    for pair in pairs:
        p_list = pred_by_pair_points[pair]
        r_list = ref_by_pair_points[pair]
        metrics = _evaluate(p_list, r_list, tolerance)
        rows.append(
            {
                "canonical_u": pair[0],
                "canonical_v": pair[1],
                "offline_count": int(ref_by_pair.get(pair, 0)),
                "incremental_count": int(pred_by_pair.get(pair, 0)),
                "matches": int(metrics["num_matches"]),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "frequency_diff_incremental_minus_offline": int(pred_by_pair.get(pair, 0) - ref_by_pair.get(pair, 0)),
                "only_offline": int(pred_by_pair.get(pair, 0) == 0 and ref_by_pair.get(pair, 0) > 0),
                "only_incremental": int(ref_by_pair.get(pair, 0) == 0 and pred_by_pair.get(pair, 0) > 0),
            }
        )
    return pd.DataFrame(rows)


def _build_unmatched_examples(
    pred: list[EdgePoint],
    ref: list[EdgePoint],
    snapshot: list[EdgePoint],
    tolerance: int,
    limit: int = 300,
) -> pd.DataFrame:
    used_ref: set[int] = set()
    rows: list[dict[str, Any]] = []
    ref_index_by_frame: defaultdict[int, list[EdgePoint]] = defaultdict(list)
    snap_index_by_frame: defaultdict[int, list[EdgePoint]] = defaultdict(list)
    for r in ref:
        ref_index_by_frame[r.frame_id].append(r)
    for s in snapshot:
        snap_index_by_frame[s.frame_id].append(s)

    for p in pred:
        match_idx = _find_match_index(p, ref, tolerance, used_ref)
        if match_idx is not None:
            used_ref.add(match_idx)
            continue
        nearest_offline_same_pair = _find_match_index(p, ref, 50, None)
        if p.raw_src is None or p.raw_dst is None:
            reason = "mapping_missing"
        elif nearest_offline_same_pair is not None:
            reason = "temporal_shift"
        else:
            reason = "wrong_pair"
        snap_here = snap_index_by_frame.get(p.frame_id, [])
        snap_edge = f"{snap_here[0].src}->{snap_here[0].dst}" if snap_here else None
        off_here = ref_index_by_frame.get(p.frame_id, [])
        off_edge = f"{off_here[0].src}->{off_here[0].dst}" if off_here else None
        rows.append(
            {
                "frame": p.frame_id,
                "edge_offline": off_edge,
                "edge_incremental": f"{p.src}->{p.dst}",
                "edge_snapshot": snap_edge,
                "edge_raw_incremental": f"{p.raw_src}->{p.raw_dst}",
                "edge_canonical_incremental": f"{p.src}->{p.dst}",
                "checkpoint_frame": p.checkpoint_frame,
                "batch_local_index": p.batch_local_index,
                "edge_type": _edge_type(p),
                "reason_estimated": reason,
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditoría completa incremental vs offline/snapshot.")
    parser.add_argument("--offline-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--incremental-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    offline_dir = args.offline_dir.expanduser().resolve()
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    incremental_dir = args.incremental_dir.expanduser().resolve()

    offline_df = _read_parquet_with_frame(offline_dir / "partido_edge_sequence.parquet")
    snapshot_df, snapshot_runs = _load_snapshot_internal(snapshot_dir / "live_snapshots_summary.json")
    incremental_internal_df = _read_parquet_with_frame(incremental_dir / "internal_edges_per_frame.parquet")
    incremental_ck_df, checkpoints = _load_incremental_checkpoint_rows(incremental_dir / "runtime_checkpoints.json")

    off_canon, off_raw = _build_points_from_df(offline_df, "offline")
    snap_canon, snap_raw = _build_points_from_df(snapshot_df, "snapshot")
    inc_canon_base, inc_raw_base = _build_points_from_df(incremental_internal_df, "incremental")

    ck_key = {(int(r["frame_id"]), str(r["edge_src"]), str(r["edge_dst"])): r for r in incremental_ck_df.to_dict(orient="records") if r.get("frame_id") is not None}
    inc_canon: list[EdgePoint] = []
    for p in inc_canon_base:
        meta = ck_key.get((p.frame_id, str(p.raw_src), str(p.raw_dst)))
        inc_canon.append(
            EdgePoint(
                frame_id=p.frame_id,
                src=p.src,
                dst=p.dst,
                raw_src=p.raw_src,
                raw_dst=p.raw_dst,
                source=p.source,
                checkpoint_frame=int(meta["checkpoint_frame"]) if meta and meta.get("checkpoint_frame") is not None else None,
                batch_local_index=int(meta["batch_local_index"]) if meta and meta.get("batch_local_index") is not None else None,
            )
        )
    inc_raw: list[EdgePoint] = []
    for p in inc_raw_base:
        meta = ck_key.get((p.frame_id, str(p.src), str(p.dst)))
        inc_raw.append(
            EdgePoint(
                frame_id=p.frame_id,
                src=p.src,
                dst=p.dst,
                raw_src=p.raw_src,
                raw_dst=p.raw_dst,
                source=p.source,
                checkpoint_frame=int(meta["checkpoint_frame"]) if meta and meta.get("checkpoint_frame") is not None else None,
                batch_local_index=int(meta["batch_local_index"]) if meta and meta.get("batch_local_index") is not None else None,
            )
        )

    min_frame = max(min(p.frame_id for p in off_canon), min(p.frame_id for p in snap_canon), min(p.frame_id for p in inc_canon))
    max_frame = min(max(p.frame_id for p in off_canon), max(p.frame_id for p in snap_canon), max(p.frame_id for p in inc_canon))
    off_i = _filter_points(off_canon, min_frame, max_frame)
    snap_i = _filter_points(snap_canon, min_frame, max_frame)
    inc_i = _filter_points(inc_canon, min_frame, max_frame)
    off_raw_i = _filter_points(off_raw, min_frame, max_frame)
    snap_raw_i = _filter_points(snap_raw, min_frame, max_frame)
    inc_raw_i = _filter_points(inc_raw, min_frame, max_frame)

    tolerance = 5
    metrics_snapshot_offline = _evaluate(snap_i, off_i, tolerance)
    metrics_incremental_offline = _evaluate(inc_i, off_i, tolerance)
    metrics_incremental_snapshot = _evaluate(inc_i, snap_i, tolerance)
    metrics_incremental_offline_raw = _evaluate(inc_raw_i, off_raw_i, tolerance)

    tol_rows: list[dict[str, Any]] = []
    for tol in [0, 1, 2, 5, 10, 20, 50]:
        m = _evaluate(inc_i, off_i, tol)
        tol_rows.append({"tolerance": tol, **m})
    tol_df = pd.DataFrame(tol_rows)
    tol_df.to_csv(out_dir / "incremental_edge_audit_by_tolerance.csv", index=False)

    offset_rows: list[dict[str, Any]] = []
    best_offset = None
    best_f1 = -1.0
    for k in range(-50, 51):
        m = _evaluate_with_offset(inc_i, off_i, tolerance=5, offset=k)
        row = {"offset": k, **m}
        offset_rows.append(row)
        f1 = row.get("f1")
        if f1 is not None and f1 > best_f1:
            best_f1 = float(f1)
            best_offset = int(k)
    offset_df = pd.DataFrame(offset_rows)
    offset_df.to_csv(out_dir / "incremental_edge_audit_by_offset.csv", index=False)

    seg_rows: list[dict[str, Any]] = []
    for start, end in _segment_bounds(min_frame, max_frame):
        off_s = _filter_points(off_i, start, end)
        inc_s = _filter_points(inc_i, start, end)
        m = _evaluate(inc_s, off_s, tolerance)
        off_counter = Counter((p.src, p.dst) for p in off_s)
        inc_counter = Counter((p.src, p.dst) for p in inc_s)
        unmatched_off = off_counter - inc_counter
        unmatched_inc = inc_counter - off_counter
        seg_rows.append(
            {
                "segment_start": start,
                "segment_end": end,
                "num_offline_edges": len(off_s),
                "num_incremental_edges": len(inc_s),
                "num_matches": m["num_matches"],
                "edge_match_ratio": m["edge_match_ratio"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "top_unmatched_offline_edges": json.dumps(_top_counter_items(unmatched_off), ensure_ascii=False),
                "top_unmatched_incremental_edges": json.dumps(_top_counter_items(unmatched_inc), ensure_ascii=False),
            }
        )
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(out_dir / "incremental_edge_audit_by_segment.csv", index=False)

    type_rows: list[dict[str, Any]] = []
    for edge_type in ["player-player", "player-out", "unknown/missing"]:
        off_t = [p for p in off_i if _edge_type(p) == edge_type]
        inc_t = [p for p in inc_i if _edge_type(p) == edge_type]
        m = _evaluate(inc_t, off_t, tolerance)
        type_rows.append(
            {
                "edge_type": edge_type,
                "num_offline_edges": len(off_t),
                "num_incremental_edges": len(inc_t),
                "num_matches": m["num_matches"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )
    edge_type_df = pd.DataFrame(type_rows)

    pair_df = _build_pair_csv(inc_i, off_i, tolerance)
    pair_df.to_csv(out_dir / "incremental_edge_audit_by_pair.csv", index=False)

    mapping_df, mapping_summary = _build_mapping_coverage(checkpoints)
    mapping_df.to_csv(out_dir / "incremental_mapping_coverage.csv", index=False)

    batch_rows: list[dict[str, Any]] = []
    for ck in checkpoints:
        frame = int(ck.get("frame_id"))
        batch = list(ck.get("raw_edge_batch") or [])
        if not batch:
            continue
        frames = [int(x.get("frame_id")) for x in batch if x.get("frame_id") is not None]
        raw_pairs = [(str(x.get("edge_src")), str(x.get("edge_dst"))) for x in batch]
        canonical_pairs = []
        for x in batch:
            pair = _canonical_pair_from_row(x)
            if pair is not None:
                canonical_pairs.append(pair)
        raw_counter = Counter(raw_pairs)
        can_counter = Counter(canonical_pairs)
        top_raw = raw_counter.most_common(1)[0] if raw_counter else (None, 0)
        batch_rows.append(
            {
                "checkpoint_frame": frame,
                "batch_start_frame": min(frames) if frames else None,
                "batch_end_frame": max(frames) if frames else None,
                "num_frames_in_batch": len(frames),
                "num_edges": len(batch),
                "num_unique_edges_raw": len(raw_counter),
                "num_unique_edges_canonical": len(can_counter),
                "unique_edge_ratio_raw": (len(raw_counter) / len(batch)) if batch else None,
                "unique_edge_ratio_canonical": (len(can_counter) / len(batch)) if batch else None,
                "most_frequent_edge_raw": f"{top_raw[0][0]}->{top_raw[0][1]}" if top_raw[0] is not None else None,
                "most_frequent_edge_raw_count": int(top_raw[1]),
            }
        )
    batch_df = pd.DataFrame(batch_rows)
    batch_df.to_csv(out_dir / "incremental_unique_edges_per_batch.csv", index=False)

    unmatched_df = _build_unmatched_examples(inc_i, off_i, snap_i, tolerance=5, limit=300)
    unmatched_df.to_csv(out_dir / "incremental_unmatched_examples.csv", index=False)

    inc_vs_snap_summary = {
        "frame_intersection": {"start": min_frame, "end": max_frame},
        "tolerance_frames": 5,
        "metrics": metrics_incremental_snapshot,
        "counts": {
            "incremental_edges": len(inc_i),
            "snapshot_edges": len(snap_i),
        },
    }
    with (out_dir / "incremental_vs_snapshot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(inc_vs_snap_summary, f, ensure_ascii=False, indent=2)

    report = {
        "summary": {
            "main_findings": [
                "Incremental presenta fuerte desalineación con offline en la intersección temporal.",
                "Snapshot mantiene similitud alta con offline en la misma intersección.",
                "La cobertura de mapping determina si la caída viene de IDs o de dinámica del edge.",
            ],
            "likely_root_causes": [],
            "recommended_next_steps": [],
        },
        "coverage": {
            "frame_intersection_start": min_frame,
            "frame_intersection_end": max_frame,
            "offline_edges_intersection": len(off_i),
            "snapshot_edges_intersection": len(snap_i),
            "incremental_edges_intersection": len(inc_i),
            "incremental_checkpoints": len(checkpoints),
        },
        "intersection_metrics": {
            "snapshot_vs_offline": metrics_snapshot_offline,
            "incremental_vs_offline": metrics_incremental_offline,
            "incremental_vs_snapshot": metrics_incremental_snapshot,
        },
        "incremental_vs_snapshot": inc_vs_snap_summary,
        "tolerance_sweep": tol_rows,
        "offset_sweep": {
            "best_offset": best_offset,
            "best_f1": best_f1 if best_f1 >= 0.0 else None,
            "rows": len(offset_rows),
        },
        "segment_analysis": {
            "segments": len(seg_rows),
            "worst_segment_by_f1": (
                seg_df.sort_values("f1", ascending=True, na_position="last").head(1).to_dict(orient="records")[0]
                if not seg_df.empty
                else None
            ),
        },
        "edge_type_analysis": edge_type_df.to_dict(orient="records"),
        "pair_analysis_summary": {
            "pairs_total": int(len(pair_df)),
            "pairs_only_offline": int(pair_df["only_offline"].sum()) if not pair_df.empty else 0,
            "pairs_only_incremental": int(pair_df["only_incremental"].sum()) if not pair_df.empty else 0,
        },
        "mapping_coverage": mapping_summary,
        "batch_diversity": {
            "batches": int(len(batch_df)),
            "avg_unique_edges_canonical": float(batch_df["num_unique_edges_canonical"].mean()) if not batch_df.empty else None,
            "pct_batches_single_canonical_edge": (
                float((batch_df["num_unique_edges_canonical"] == 1).mean()) if not batch_df.empty else None
            ),
            "pct_batches_constant_edge_all_frames": (
                float((batch_df["most_frequent_edge_raw_count"] == batch_df["num_edges"]).mean()) if not batch_df.empty else None
            ),
        },
        "raw_vs_canonical": {
            "incremental_vs_offline_raw": metrics_incremental_offline_raw,
            "incremental_vs_offline_canonical": metrics_incremental_offline,
        },
        "config_comparison": {
            "offline_summary_path": str(offline_dir / "partido_summary.json"),
            "snapshot_summary_path": str(snapshot_dir / "live_snapshots_summary.json"),
            "incremental_summary_path": str(incremental_dir / "summary.json"),
            "notes": [
                "Snapshot usa wrapper legacy offline por snapshot.",
                "Incremental usa ActionsRuntime causal con cadence/warmup y batch interno.",
                "Revisar decode/use_crf/sample_freq/window_seconds entre summaries para paridad estricta.",
            ],
        },
        "error_examples_summary": {
            "rows_exported": int(len(unmatched_df)),
            "top_reasons": Counter(unmatched_df["reason_estimated"].tolist()).most_common(5),
        },
    }

    # Heurística para causa principal
    m_cov = mapping_summary.get("pct_edges_with_both_endpoints_mapped")
    best_off = report["offset_sweep"]["best_f1"]
    can_f1 = metrics_incremental_offline.get("f1")
    raw_f1 = metrics_incremental_offline_raw.get("f1")
    likely = []
    if m_cov is not None and m_cov < 0.9:
        likely.append("A) canonical mapping roto o cobertura de mapping insuficiente")
    if best_off is not None and can_f1 is not None and best_off > can_f1 * 1.5:
        likely.append("B) desfase temporal parcial entre incremental y offline")
    if not likely and raw_f1 is not None and can_f1 is not None and raw_f1 > can_f1 * 1.5:
        likely.append("A) canonical mapping/inconsistencia de mapping (raw mejor que canonical)")
    if not likely:
        likely.append("F) diferencia real de dinámica causal/incremental frente a offline")
    report["summary"]["likely_root_causes"] = likely
    report["summary"]["recommended_next_steps"] = [
        "Forzar paridad estricta de config PathCRF (decode/use_crf/sample_freq/window_seconds) entre snapshot e incremental.",
        "Auditar batch->frame assignment en incremental usando checkpoint_frame y batch_local_index de los unmatched.",
        "Repetir evaluación con ventana fija 40..724 y offset óptimo para confirmar si el fallo es de tiempo o de par de edge.",
    ]

    with (out_dir / "incremental_edge_audit_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Output dir: {out_dir}")
    print(f"Report: {out_dir / 'incremental_edge_audit_report.json'}")


if __name__ == "__main__":
    main()

