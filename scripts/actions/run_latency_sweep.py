from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from football_ai.pathcrf_slot_mapping import (
    person_slot_to_canonical_id,
    referee_slot_to_canonical_id,
)

DELAYS = [0, 25, 50, 75, 100, 125, 150, 200]
EMIT_FRAMES = 10
CADENCE_FRAMES = 10
FPS = 25.0
MIN_FRAMES_WARMUP = 50
FRAME_START = 40
FRAME_END = 724
TOLERANCE_FRAMES = 5

ROLLING_DIR = PROJECT_ROOT / "output/actions/pathcrf_rolling_adapter"
OFFLINE_DIR = PROJECT_ROOT / "output/actions/pathcrf_compare/video_prueba/offline_run_current"
OUTPUT_DIR = PROJECT_ROOT / "output/actions/pathcrf_rolling_latency_sweep"


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


def _canonical_pair(row: dict[str, Any]) -> tuple | None:
    if "canonical_src" in row and "canonical_dst" in row:
        cs = row["canonical_src"]
        cd = row["canonical_dst"]
    else:
        cs = _slot_to_canonical(row.get("edge_src"))
        cd = _slot_to_canonical(row.get("edge_dst"))
    if cs is None or cd is None:
        return None
    try:
        fid = int(row["frame_id"])
    except (TypeError, ValueError):
        return None
    return (fid, str(cs), str(cd))


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


def load_offline_edges() -> list[tuple[int, str, str]]:
    edge_path = OFFLINE_DIR / "tracks_edge_sequence.parquet"
    if not edge_path.exists():
        raise FileNotFoundError(f"Offline edge sequence not found: {edge_path}")
    df = _read_parquet_frame(edge_path)
    edges = []
    for row in df.to_dict(orient="records"):
        pair = _canonical_pair(row)
        if pair:
            edges.append(pair)
    return edges


def simulate_emitted_edges(
    rolling_df: pd.DataFrame,
    emit_delay: int,
    emit_frames: int = EMIT_FRAMES,
    min_warmup: int = MIN_FRAMES_WARMUP,
    cadence: int = CADENCE_FRAMES,
    max_frames: int = FRAME_END + 200,
) -> list[tuple[int, str, str]]:
    emitted: dict[int, tuple[int, str, str]] = {}
    checkpoints = sorted(rolling_df["checkpoint_frame"].unique())

    for cp in checkpoints:
        cp = int(cp)
        if (cp + 1) < min_warmup:
            continue
        emit_end = cp - int(emit_delay)
        emit_start = emit_end - int(emit_frames) + 1
        if emit_start > emit_end:
            emit_start = emit_end

        cp_rows = rolling_df[rolling_df["checkpoint_frame"] == cp]
        for _, row in cp_rows.iterrows():
            fid = int(row["frame_id"])
            if not (emit_start <= fid <= emit_end):
                continue
            pair = _canonical_pair(row.to_dict())
            if pair is None:
                continue
            if fid in emitted:
                continue
            emitted[fid] = pair

    return [emitted[fid] for fid in sorted(emitted)]


def simulate_dedup_edges(rolling_df: pd.DataFrame) -> list[tuple[int, str, str]]:
    df_sorted = rolling_df.sort_values("checkpoint_frame", ascending=True)
    df_dedup = df_sorted.drop_duplicates(subset=["frame_id"], keep="last").sort_values("frame_id")
    edges = []
    for _, row in df_dedup.iterrows():
        pair = _canonical_pair(row.to_dict())
        if pair:
            edges.append(pair)
    return edges


def filter_range(edges: list[tuple[int, str, str]], lo: int, hi: int) -> list[tuple[int, str, str]]:
    return [(fid, s, d) for fid, s, d in edges if lo <= fid <= hi]


def match_edges(
    pred: list[tuple[int, str, str]],
    ref: list[tuple[int, str, str]],
    tolerance: int,
) -> dict[str, Any]:
    used_ref: set[int] = set()
    match_count = 0
    for fid_p, src_p, dst_p in pred:
        lo = fid_p - tolerance
        hi = fid_p + tolerance
        best_idx = None
        best_dist = None
        for i, (fid_r, src_r, dst_r) in enumerate(ref):
            if i in used_ref:
                continue
            if src_p != src_r or dst_p != dst_r:
                continue
            if not (lo <= fid_r <= hi):
                continue
            dist = abs(fid_r - fid_p)
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
        "num_pred": len(pred),
        "num_ref": len(ref),
        "num_matches": match_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def eval_by_segment(
    pred: list[tuple[int, str, str]],
    ref: list[tuple[int, str, str]],
    tolerance: int,
    seg_start: int,
    seg_end: int,
) -> list[dict[str, Any]]:
    segs = [(40, 99), (100, 199), (200, 299), (300, 399), (400, 499),
            (500, 599), (600, 699), (700, seg_end)]
    rows = []
    for a, b in segs:
        lo, hi = max(a, seg_start), min(b, seg_end)
        if lo > hi:
            continue
        p = filter_range(pred, lo, hi)
        r = filter_range(ref, lo, hi)
        m = match_edges(p, r, tolerance)
        rows.append({"segment_start": lo, "segment_end": hi, **m})
    return rows


def eval_by_pair(
    pred: list[tuple[int, str, str]],
    ref: list[tuple[int, str, str]],
    tolerance: int,
) -> pd.DataFrame:
    pred_by: dict[tuple[str, str], list] = defaultdict(list)
    ref_by: dict[tuple[str, str], list] = defaultdict(list)
    for fid, src, dst in pred:
        pred_by[(src, dst)].append(fid)
    for fid, src, dst in ref:
        ref_by[(src, dst)].append(fid)
    all_pairs = sorted(set(pred_by) | set(ref_by))
    rows = []
    for pair in all_pairs:
        p_list = pred_by[pair]
        r_list = ref_by[pair]
        p_edges = [(f, pair[0], pair[1]) for f in p_list]
        r_edges = [(f, pair[0], pair[1]) for f in r_list]
        m = match_edges(p_edges, r_edges, tolerance)
        rows.append({
            "canonical_u": pair[0],
            "canonical_v": pair[1],
            "ref_count": len(r_edges),
            "pred_count": len(p_edges),
            "matches": m["num_matches"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
        })
    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rolling_path = ROLLING_DIR / "rolling_edges_per_frame.parquet"
    if not rolling_path.exists():
        raise FileNotFoundError(f"Rolling edges not found: {rolling_path}")
    rolling_df = pd.read_parquet(rolling_path)
    print(f"Loaded rolling edges: {len(rolling_df)} rows")

    offline_edges = load_offline_edges()
    offline_filt = filter_range(offline_edges, FRAME_START, FRAME_END)
    print(f"Offline edges [{FRAME_START}..{FRAME_END}]: {len(offline_filt)}")

    dedup_edges = simulate_dedup_edges(rolling_df)
    dedup_filt = filter_range(dedup_edges, FRAME_START, FRAME_END)
    print(f"Dedup edges [{FRAME_START}..{FRAME_END}]: {len(dedup_filt)}")

    dedup_metrics = match_edges(dedup_filt, offline_filt, TOLERANCE_FRAMES)
    print(f"\nDedup vs offline: F1={dedup_metrics['f1']:.4f} ({dedup_metrics['num_matches']} matches)")

    all_segment_rows = []
    all_pair_dfs = []
    summary_rows = []

    for delay in DELAYS:
        emit_edges = simulate_emitted_edges(rolling_df, delay)
        emit_filt = filter_range(emit_edges, FRAME_START, FRAME_END)
        emit_m = match_edges(emit_filt, offline_filt, TOLERANCE_FRAMES)
        latency_sec = delay / FPS
        coverage = len(emit_filt) / len(offline_filt) if offline_filt else 0.0

        print(f"\ndelay={delay} ({latency_sec:.1f}s): F1={emit_m['f1']:.4f}, coverage={coverage:.3f}, "
              f"pred={emit_m['num_pred']}, match={emit_m['num_matches']}")

        summary_rows.append({
            "emit_delay_frames": delay,
            "latency_seconds": round(latency_sec, 2),
            "mode": "emitted",
            "f1": round(emit_m["f1"], 5) if emit_m["f1"] is not None else None,
            "precision": round(emit_m["precision"], 5) if emit_m["precision"] is not None else None,
            "recall": round(emit_m["recall"], 5) if emit_m["recall"] is not None else None,
            "num_pred_edges": emit_m["num_pred"],
            "num_ref_edges": emit_m["num_ref"],
            "num_matches": emit_m["num_matches"],
            "coverage": round(coverage, 5),
            "cadence_frames": CADENCE_FRAMES,
            "emit_frames": EMIT_FRAMES,
            "fps": FPS,
            "frame_start": FRAME_START,
            "frame_end": FRAME_END,
            "tolerance_frames": TOLERANCE_FRAMES,
        })

        seg_rows = eval_by_segment(emit_edges, offline_edges, TOLERANCE_FRAMES, FRAME_START, FRAME_END)
        for sr in seg_rows:
            sr["emit_delay_frames"] = delay
            sr["mode"] = "emitted"
        all_segment_rows.extend(seg_rows)

        pair_df = eval_by_pair(emit_edges, offline_edges, TOLERANCE_FRAMES)
        pair_df["emit_delay_frames"] = delay
        pair_df["mode"] = "emitted"
        all_pair_dfs.append(pair_df)

    dedup_cov = len(dedup_filt) / len(offline_filt) if offline_filt else 0.0
    summary_rows.append({
        "emit_delay_frames": 999,
        "latency_seconds": "dedup",
        "mode": "dedup",
        "f1": round(dedup_metrics["f1"], 5) if dedup_metrics["f1"] is not None else None,
        "precision": round(dedup_metrics["precision"], 5) if dedup_metrics["precision"] is not None else None,
        "recall": round(dedup_metrics["recall"], 5) if dedup_metrics["recall"] is not None else None,
        "num_pred_edges": dedup_metrics["num_pred"],
        "num_ref_edges": dedup_metrics["num_ref"],
        "num_matches": dedup_metrics["num_matches"],
        "coverage": round(dedup_cov, 5),
        "cadence_frames": CADENCE_FRAMES,
        "emit_frames": EMIT_FRAMES,
        "fps": FPS,
        "frame_start": FRAME_START,
        "frame_end": FRAME_END,
        "tolerance_frames": TOLERANCE_FRAMES,
    })

    dedup_seg = eval_by_segment(dedup_edges, offline_edges, TOLERANCE_FRAMES, FRAME_START, FRAME_END)
    for sr in dedup_seg:
        sr["emit_delay_frames"] = 999
        sr["mode"] = "dedup"
    all_segment_rows.extend(dedup_seg)

    dedup_pair = eval_by_pair(dedup_edges, offline_edges, TOLERANCE_FRAMES)
    dedup_pair["emit_delay_frames"] = 999
    dedup_pair["mode"] = "dedup"
    all_pair_dfs.append(dedup_pair)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUTPUT_DIR / "rolling_latency_sweep_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary CSV: {summary_csv}")

    summary_json_path = OUTPUT_DIR / "rolling_latency_sweep_summary.json"
    with summary_json_path.open("w") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False, default=str)
    print(f"Summary JSON: {summary_json_path}")

    seg_df = pd.DataFrame(all_segment_rows)
    seg_csv = OUTPUT_DIR / "rolling_latency_by_segment.csv"
    seg_df.to_csv(seg_csv, index=False)
    print(f"Segment CSV: {seg_csv}")

    pair_concat = pd.concat(all_pair_dfs, ignore_index=True)
    pair_csv = OUTPUT_DIR / "rolling_latency_by_pair.csv"
    pair_concat.to_csv(pair_csv, index=False)
    print(f"Pair CSV: {pair_csv}")

    print("\n===== SWEEP SUMMARY =====")
    print(f"{'Delay':>6s} | {'Lat(s)':>6s} | {'Mode':>8s} | {'F1':>8s} | {'Cov':>6s} | {'Pred':>5s} | {'Match':>5s}")
    print("-" * 70)
    for r in summary_rows:
        d = f"{r['emit_delay_frames']}" if r['emit_delay_frames'] != 999 else "dedup"
        lat = f"{r['latency_seconds']}" if r['latency_seconds'] != 'dedup' else "dedup"
        f1 = f"{r['f1']:.4f}" if r['f1'] is not None else "None"
        print(f"{d:>6s} | {lat:>6s} | {r['mode']:>8s} | {f1:>8s} | {r['coverage']:.3f} | {r['num_pred_edges']:>5d} | {r['num_matches']:>5d}")

    delays_under = {}
    for threshold in [1.0, 2.0, 3.0, 4.0, 5.0]:
        best = None
        for r in summary_rows:
            if r["mode"] != "emitted":
                continue
            if r["latency_seconds"] is not None and r["latency_seconds"] <= threshold:
                if r["f1"] is not None and (best is None or r["f1"] > best["f1"]):
                    best = r
        if best:
            delays_under[f"{threshold:.0f}s"] = {
                "delay_frames": best["emit_delay_frames"],
                "latency_seconds": best["latency_seconds"],
                "f1": best["f1"],
            }
    print(f"\nBest F1 under latency budgets: {json.dumps(delays_under, indent=2)}")

    f1_75 = next((r for r in summary_rows if r["mode"] == "emitted" and r["f1"] and r["f1"] >= 0.75), None)
    f1_85 = next((r for r in summary_rows if r["mode"] == "emitted" and r["f1"] and r["f1"] >= 0.85), None)
    f1_95 = next((r for r in summary_rows if r["mode"] == "emitted" and r["f1"] and r["f1"] >= 0.95), None)
    print(f"\nF1 >= 0.75: {'delay=' + str(f1_75['emit_delay_frames']) + ' (' + str(f1_75['latency_seconds']) + 's) F1=' + str(round(f1_75['f1'], 4)) if f1_75 else 'NOT REACHED'}")
    print(f"F1 >= 0.85: {'delay=' + str(f1_85['emit_delay_frames']) + ' (' + str(f1_85['latency_seconds']) + 's) F1=' + str(round(f1_85['f1'], 4)) if f1_85 else 'NOT REACHED'}")
    print(f"F1 >= 0.95: {'delay=' + str(f1_95['emit_delay_frames']) + ' (' + str(f1_95['latency_seconds']) + 's) F1=' + str(round(f1_95['f1'], 4)) if f1_95 else 'NOT REACHED'}")
    print(f"Dedup parity: F1={dedup_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
