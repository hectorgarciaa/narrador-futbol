from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_ai.pathcrf_slot_mapping import (
    person_slot_to_canonical_id,
    referee_slot_to_canonical_id,
)


@dataclass(frozen=True)
class EdgePoint:
    frame_id: int
    src: str
    dst: str
    source: str


def _read_parquet_frame(path: Path) -> pd.DataFrame:
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


def _deduplicate_rolling_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "checkpoint_frame" not in df.columns:
        return df.drop_duplicates(subset=["frame_id"], keep="last")
    df_sorted = df.sort_values("checkpoint_frame", ascending=True)
    return df_sorted.drop_duplicates(subset=["frame_id"], keep="last").sort_values("frame_id")


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


def _pair_from_row(row: dict[str, Any]) -> tuple[int, tuple[str, str]] | None:
    frame = row.get("frame_id")
    try:
        frame_id = int(frame)
    except (TypeError, ValueError):
        return None
    if "canonical_src" in row and "canonical_dst" in row:
        can_src = str(row["canonical_src"]) if row.get("canonical_src") not in (None,) else None
        can_dst = str(row["canonical_dst"]) if row.get("canonical_dst") not in (None,) else None
    else:
        slot_src = row.get("edge_src")
        slot_dst = row.get("edge_dst")
        can_src = _slot_to_canonical(slot_src)
        can_dst = _slot_to_canonical(slot_dst)
    if can_src is None or can_dst is None:
        return None
    return (frame_id, (str(can_src), str(can_dst)))


def _build_points(df: pd.DataFrame, source: str) -> list[EdgePoint]:
    points: list[EdgePoint] = []
    for row in df.to_dict(orient="records"):
        result = _pair_from_row(row)
        if result is None:
            continue
        fid, (src, dst) = result
        points.append(EdgePoint(frame_id=fid, src=src, dst=dst, source=source))
    return points


def _filter_points(points: list[EdgePoint], start: int, end: int) -> list[EdgePoint]:
    return [p for p in points if start <= p.frame_id <= end]


def _match(pred: list[EdgePoint], ref: list[EdgePoint], tolerance: int) -> dict[str, Any]:
    used_ref: set[int] = set()
    match_count = 0
    for p in pred:
        lo = p.frame_id - tolerance
        hi = p.frame_id + tolerance
        best_idx = None
        best_dist = None
        for i, r in enumerate(ref):
            if i in used_ref:
                continue
            if p.src != r.src or p.dst != r.dst:
                continue
            if not (lo <= r.frame_id <= hi):
                continue
            dist = abs(r.frame_id - p.frame_id)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            used_ref.add(best_idx)
            match_count += 1

    precision = (match_count / len(pred)) if pred else None
    recall = (match_count / len(ref)) if ref else None
    if precision is not None and recall is not None and (precision + recall) > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    elif precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = None
    return {
        "num_pred_edges": len(pred),
        "num_ref_edges": len(ref),
        "num_matches": match_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "edge_match_ratio": precision,
    }


def _segment_bounds(start: int, end: int) -> list[tuple[int, int]]:
    segments = [(40, 99), (100, 199), (200, 299), (300, 399), (400, 499), (500, 599), (600, 699), (700, 724)]
    result = []
    for a, b in segments:
        lo, hi = max(a, start), min(b, end)
        if lo <= hi:
            result.append((lo, hi))
    return result


def _eval_by_segment(
    pred: list[EdgePoint], ref: list[EdgePoint], tolerance: int
) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in _segment_bounds(40, ref[-1].frame_id if ref else 724):
        p = _filter_points(pred, max(lo, 40), hi)
        r = _filter_points(ref, max(lo, 40), hi)
        m = _match(p, r, tolerance)
        rows.append({"segment_start": max(lo, 40), "segment_end": hi, **m})
    return rows


def _eval_by_pair(
    pred: list[EdgePoint], ref: list[EdgePoint], tolerance: int
) -> pd.DataFrame:
    pred_by_pair = defaultdict(list)
    ref_by_pair = defaultdict(list)
    for p in pred:
        pred_by_pair[(p.src, p.dst)].append(p)
    for r in ref:
        ref_by_pair[(r.src, r.dst)].append(r)
    all_pairs = sorted(set(pred_by_pair.keys()) | set(ref_by_pair.keys()))
    rows = []
    for pair in all_pairs:
        p_list = pred_by_pair[pair]
        r_list = ref_by_pair[pair]
        m = _match(p_list, r_list, tolerance)
        rows.append(
            {
                "canonical_u": pair[0],
                "canonical_v": pair[1],
                "ref_count": len(r_list),
                "pred_count": len(p_list),
                "matches": m["num_matches"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )
    return pd.DataFrame(rows)


def _load_snapshot_internal(snapshot_summary_path: Path) -> pd.DataFrame:
    payload = json.loads(snapshot_summary_path.read_text(encoding="utf-8"))
    runs = list(payload.get("snapshot_runs") or [])
    by_frame: dict[int, dict[str, Any]] = {}
    for run in runs:
        edge_path = run.get("edge_sequence_path")
        if not edge_path:
            continue
        edge_df = _read_parquet_frame(Path(edge_path))
        for item in edge_df.to_dict(orient="records"):
            try:
                fid = int(item["frame_id"])
            except (TypeError, ValueError):
                continue
            by_frame[fid] = {
                "frame_id": fid,
                "edge_src": item.get("edge_src"),
                "edge_dst": item.get("edge_dst"),
            }
    rows = [row for _, row in sorted(by_frame.items())]
    return pd.DataFrame(rows)


def _load_legacy_edges(incremental_dir: Path) -> pd.DataFrame:
    internal_path = incremental_dir / "internal_edges_per_frame.parquet"
    if internal_path.exists():
        return _read_parquet_frame(internal_path)
    emitted_path = incremental_dir / "emitted_edges.parquet"
    if emitted_path.exists():
        return _read_parquet_frame(emitted_path)
    return pd.DataFrame()


def _resolve_offline_edge_path(offline_dir: Path) -> Path | None:
    for name in ("partido_edge_sequence.parquet", "partido_corto_edge_sequence.parquet"):
        candidate = offline_dir / name
        if candidate.exists():
            return candidate
    for p in offline_dir.glob("*_edge_sequence.parquet"):
        return p
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluacion de rolling PathCRF vs offline/snapshot.")
    p.add_argument(
        "--rolling-dir", type=Path, required=True, help="Directorio con rolling outputs."
    )
    p.add_argument(
        "--offline-dir",
        type=Path,
        required=True,
        help="Directorio con edge_sequence offline.",
    )
    p.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Directorio con live_snapshots_summary.json.",
    )
    p.add_argument(
        "--incremental-dir",
        type=Path,
        default=None,
        help="Directorio con legacy incremental outputs.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio para reportes de evaluacion.",
    )
    p.add_argument("--frame-start", type=int, default=40)
    p.add_argument("--frame-end", type=int, default=724)
    p.add_argument("--tolerance", type=int, default=5)
    p.add_argument(
        "--use-emitted-only",
        action="store_true",
        help="Evaluar solo emitted_edges, no el rolling completo.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    rolling_dir = args.rolling_dir.expanduser().resolve()
    offline_dir = args.offline_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tolerance = args.tolerance
    fs, fe = args.frame_start, args.frame_end

    edge_path = _resolve_offline_edge_path(offline_dir)
    if edge_path is None:
        raise FileNotFoundError(f"No se encontro edge_sequence.parquet en {offline_dir}")
    offline_df = _read_parquet_frame(edge_path)
    offline_points = _build_points(offline_df, "offline")
    offline_i = _filter_points(offline_points, fs, fe)

    rolling_summary = rolling_dir / "summary.json"
    rolling_config = None
    if rolling_summary.exists():
        rolling_config = json.loads(rolling_summary.read_text(encoding="utf-8")).get("config")

    if args.use_emitted_only:
        sources = [(rolling_dir / "emitted_edges.parquet", "emitted", False)]
    else:
        sources = [
            (rolling_dir / "emitted_edges.parquet", "emitted", False),
            (rolling_dir / "rolling_edges_per_frame.parquet", "full", True),
        ]

    eval_results = []
    for edge_path_src, label, needs_dedup in sources:
        edge_path_src = Path(edge_path_src)
        if not edge_path_src.exists():
            continue
        df = _read_parquet_frame(edge_path_src)
        if df.empty:
            continue
        if needs_dedup:
            df = _deduplicate_rolling_edges(df)
            label = f"{label}_dedup"
        pred_points = _build_points(df, label)
        pred_i = _filter_points(pred_points, fs, fe)
        metrics = _match(pred_i, offline_i, tolerance)
        metrics["comparison"] = f"{label}_vs_offline"
        if rolling_config:
            metrics["cadence_frames"] = rolling_config.get("cadence_frames")
            metrics["context_frames"] = rolling_config.get("context_frames")
            metrics["emit_delay_frames"] = rolling_config.get("emit_delay_frames")
            metrics["emit_frames"] = rolling_config.get("emit_frames")
            metrics["use_crf"] = rolling_config.get("use_crf")
            metrics["decode"] = rolling_config.get("decode")
        eval_results.append(metrics)

        print(f"\n{label} vs offline [{fs}..{fe}]:")
        for k in ["num_pred_edges", "num_ref_edges", "num_matches", "precision", "recall", "f1"]:
            v = metrics.get(k)
            if v is not None:
                print(f"  {k}: {v}")

        seg_rows = _eval_by_segment(pred_points, offline_points, tolerance)
        seg_df = pd.DataFrame(seg_rows)
        seg_path = out_dir / f"rolling_eval_by_segment_{label}.csv"
        seg_df.to_csv(seg_path, index=False)

        pair_df = _eval_by_pair(pred_points, offline_points, tolerance)
        pair_path = out_dir / f"rolling_eval_by_pair_{label}.csv"
        pair_df.to_csv(pair_path, index=False)

    eval_offline_df = pd.DataFrame(eval_results)
    eval_offline_path = out_dir / "rolling_eval_against_offline.csv"
    eval_offline_df.to_csv(eval_offline_path, index=False)

    eval_dfs_snapshot: list[dict[str, Any]] = []
    if args.snapshot_dir is not None:
        snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
        summary_path = snapshot_dir / "live_snapshots_summary.json"
        if summary_path.exists():
            snap_df = _load_snapshot_internal(summary_path)
            if not snap_df.empty:
                snap_points = _build_points(snap_df, "snapshot_internal")
                snap_i = _filter_points(snap_points, fs, fe)

                for edge_path_src, label, needs_dedup in sources:
                    edge_path_src = Path(edge_path_src)
                    if not edge_path_src.exists():
                        continue
                    df = _read_parquet_frame(edge_path_src)
                    if df.empty:
                        continue
                    if needs_dedup:
                        df = _deduplicate_rolling_edges(df)
                        label = f"{label}_dedup"
                    pred = _build_points(df, label)
                    pred_i = _filter_points(pred, fs, fe)
                    m = _match(pred_i, snap_i, tolerance)
                    m["comparison"] = f"{label}_vs_snapshot_internal"
                    eval_dfs_snapshot.append(m)
                    print(f"\n{label} vs snapshot_internal [{fs}..{fe}]:")
                    for k in ["num_pred_edges", "num_ref_edges", "num_matches", "precision", "recall", "f1"]:
                        v = m.get(k)
                        if v is not None:
                            print(f"  {k}: {v}")

    if eval_dfs_snapshot:
        snap_eval_df = pd.DataFrame(eval_dfs_snapshot)
        snap_eval_path = out_dir / "rolling_eval_against_snapshot_internal.csv"
        snap_eval_df.to_csv(snap_eval_path, index=False)

    eval_dfs_legacy: list[dict[str, Any]] = []
    if args.incremental_dir is not None:
        incremental_dir = Path(args.incremental_dir).expanduser().resolve()
        if incremental_dir.exists():
            legacy_df = _load_legacy_edges(incremental_dir)
            if not legacy_df.empty:
                legacy_points = _build_points(legacy_df, "legacy_incremental")
                legacy_i = _filter_points(legacy_points, fs, fe)
                m = _match(legacy_i, offline_i, tolerance)
                m["comparison"] = "legacy_incremental_vs_offline"
                eval_dfs_legacy.append(m)
                print(f"\nlegacy incremental vs offline [{fs}..{fe}]:")
                for k in ["num_pred_edges", "num_ref_edges", "num_matches", "precision", "recall", "f1"]:
                    v = m.get(k)
                    if v is not None:
                        print(f"  {k}: {v}")

    if eval_dfs_legacy:
        leg_df = pd.DataFrame(eval_dfs_legacy)
        leg_path = out_dir / "rolling_eval_legacy_inc_vs_offline.csv"
        leg_df.to_csv(leg_path, index=False)

    all_eval = {}
    if not eval_offline_df.empty:
        for _, r in eval_offline_df.iterrows():
            comp = r.get("comparison", "unknown")
            all_eval[comp] = {
                "num_pred_edges": r.get("num_pred_edges"),
                "num_ref_edges": r.get("num_ref_edges"),
                "num_matches": int(r.get("num_matches", 0)),
                "precision": r.get("precision"),
                "recall": r.get("recall"),
                "f1": r.get("f1"),
            }
    for d in eval_dfs_snapshot:
        all_eval[d.get("comparison", "unknown")] = {
            "num_pred_edges": d.get("num_pred_edges"),
            "num_ref_edges": d.get("num_ref_edges"),
            "num_matches": int(d.get("num_matches", 0)),
            "precision": d.get("precision"),
            "recall": d.get("recall"),
            "f1": d.get("f1"),
        }
    for d in eval_dfs_legacy:
        all_eval[d.get("comparison", "unknown")] = {
            "num_pred_edges": d.get("num_pred_edges"),
            "num_ref_edges": d.get("num_ref_edges"),
            "num_matches": int(d.get("num_matches", 0)),
            "precision": d.get("precision"),
            "recall": d.get("recall"),
            "f1": d.get("f1"),
        }

    summary_path = out_dir / "rolling_eval_summary.json"
    summary_payload = {
        "frame_range": {"start": fs, "end": fe},
        "tolerance_frames": tolerance,
        "evaluations": all_eval,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nReportes en: {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()