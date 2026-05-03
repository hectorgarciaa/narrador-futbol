from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from football_ai.actions import (
    PathCRFAdapterConfig,
    PathCRFInferenceConfig,
    PathCRFRenderConfig,
    run_pathcrf_pipeline,
)
from football_ai.actions_incremental import ActionsDetectorConfig, ActionsRuntime, ActionsRuntimeConfig
from football_ai.pathcrf_slot_mapping import person_slot_to_canonical_id, referee_slot_to_canonical_id


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


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


def _canonical_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    src = _slot_to_canonical(row.get("edge_src"))
    dst = _slot_to_canonical(row.get("edge_dst"))
    if src is None and row.get("edge_src_track_id") is not None:
        src = str(row["edge_src_track_id"])
    if dst is None and row.get("edge_dst_track_id") is not None:
        dst = str(row["edge_dst_track_id"])
    if src is None or dst is None:
        return None
    return (str(src), str(dst))


def _raw_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    if row.get("edge_src") is None or row.get("edge_dst") is None:
        return None
    return (str(row["edge_src"]), str(row["edge_dst"]))


def _restrict_df(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df[(df["frame_id"] >= start) & (df["frame_id"] <= end)].copy()


def _to_points(df: pd.DataFrame, repr_type: str) -> list[tuple[int, str, str]]:
    pts: list[tuple[int, str, str]] = []
    for row in df.to_dict(orient="records"):
        frame = row.get("frame_id")
        try:
            fid = int(frame)
        except (TypeError, ValueError):
            continue
        if repr_type == "canonical":
            pair = _canonical_pair(row)
        elif repr_type == "raw":
            pair = _raw_pair(row)
        else:
            pair = (str(row.get("edge_src_track_id")), str(row.get("edge_dst_track_id")))
            if pair[0] == "None" or pair[1] == "None":
                pair = None
        if pair is None:
            continue
        pts.append((fid, pair[0], pair[1]))
    return pts


def _match_metrics(pred: list[tuple[int, str, str]], ref: list[tuple[int, str, str]], tolerance: int = 5) -> dict[str, Any]:
    used = set()
    matches = 0
    for i, (f, u, v) in enumerate(pred):
        lo, hi = f - tolerance, f + tolerance
        best_j = None
        best_dist = None
        for j, (fr, ur, vr) in enumerate(ref):
            if j in used:
                continue
            if u != ur or v != vr:
                continue
            if not (lo <= fr <= hi):
                continue
            d = abs(fr - f)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_j = j
        if best_j is not None:
            used.add(best_j)
            matches += 1
    precision = matches / len(pred) if pred else None
    recall = matches / len(ref) if ref else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "num_pred_edges": len(pred),
        "num_ref_edges": len(ref),
        "num_matches": matches,
        "num_unmatched_predictions": len(pred) - matches,
        "num_unmatched_references": len(ref) - matches,
        "edge_match_ratio": precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _slice_tracks_window(tracks: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in tracks.items():
        if isinstance(v, list):
            out[k] = v[start : end + 1]
        else:
            out[k] = v
    return out


def _run_snapshot_on_incremental_windows(
    *,
    tracks: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    out_dir: Path,
    inference_config: PathCRFInferenceConfig,
) -> pd.DataFrame:
    runs_dir = out_dir / "snapshot_on_incremental_windows"
    runs_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: dict[int, dict[str, Any]] = {}
    for ck in checkpoints:
        batch = list(ck.get("raw_edge_batch") or [])
        if not batch:
            continue
        frames = sorted(int(x["frame_id"]) for x in batch if x.get("frame_id") is not None)
        if not frames:
            continue
        start, end = frames[0], frames[-1]
        checkpoint_frame = int(ck["frame_id"])
        tracks_w = _slice_tracks_window(tracks, start, end)
        tracks_path = runs_dir / f"tracks_window_{start:06d}_{end:06d}.json"
        tracks_path.write_text(json.dumps(tracks_w, ensure_ascii=False), encoding="utf-8")
        run_out = runs_dir / f"ck_{checkpoint_frame:06d}"
        result = run_pathcrf_pipeline(
            output_dir=run_out,
            tracks_path=tracks_path,
            adapter_config=PathCRFAdapterConfig(fps=25.0),
            inference_config=inference_config,
            render_config=PathCRFRenderConfig(enabled=False),
        )
        edge_df = _read_parquet_with_frame(result.edge_sequence_path)
        edge_df = edge_df.sort_values("frame_id", kind="stable").reset_index(drop=True)
        for idx, row in edge_df.iterrows():
            global_frame = start + int(idx)
            frame_rows[global_frame] = {
                "frame_id": global_frame,
                "checkpoint_frame": checkpoint_frame,
                "batch_start_frame": start,
                "batch_end_frame": end,
                "batch_local_index": int(idx),
                "edge_src": row.get("edge_src"),
                "edge_dst": row.get("edge_dst"),
                "edge_src_track_id": None,
                "edge_dst_track_id": None,
                "source": "snapshot_on_incremental_windows",
            }
    rows = [v for _, v in sorted(frame_rows.items(), key=lambda kv: kv[0])]
    return pd.DataFrame(rows)


def _run_incremental_stateless_checkpoint(
    *,
    tracks: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    runtime_config: ActionsRuntimeConfig,
) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for ck in checkpoints:
        batch = list(ck.get("raw_edge_batch") or [])
        if not batch:
            continue
        frames = sorted(int(x["frame_id"]) for x in batch if x.get("frame_id") is not None)
        if not frames:
            continue
        start, end = frames[0], frames[-1]
        checkpoint_frame = int(ck["frame_id"])
        runtime = ActionsRuntime(runtime_config)
        detector = runtime.detector
        for frame in range(start, end + 1):
            clean_packet = {
                "tracks_frame": {
                    cls: (tracks.get(cls, [])[frame] if frame < len(tracks.get(cls, [])) else {})
                    for cls in ("player", "goalkeeper", "referee", "ball")
                },
                "possession": (tracks.get("possession", [])[frame] if frame < len(tracks.get("possession", [])) else {}),
            }
            detector.update(frame, clean_packet)
        tracking_df, summary = detector.build_tracking_dataframe()
        # En stateless el detector se alimenta solo con [start..end], así que
        # este dataframe ya representa la ventana del checkpoint.
        # Filtrar por frame global puede vaciar la ventana si frame_id es local.
        window_df = tracking_df.copy()
        if window_df.empty:
            continue
        # PathCRF espera columnas de estado temporal explícitas.
        if "period_id" not in window_df.columns:
            window_df["period_id"] = 1
        if "timestamp" not in window_df.columns:
            fps = float(runtime.config.detector.fps) if runtime.config.detector.fps else 25.0
            window_df["timestamp"] = window_df["frame_id"].astype(float) / max(fps, 1e-6)
        if "phase_id" not in window_df.columns:
            window_df["phase_id"] = 1
        if "episode_id" not in window_df.columns:
            window_df["episode_id"] = 1
        if "ball_state" not in window_df.columns:
            window_df["ball_state"] = "alive"
        edge_df, _ = runtime._run_pathcrf(window_df)
        edge_df = edge_df.sort_values("frame_id", kind="stable").reset_index(drop=True)
        for idx, row in edge_df.iterrows():
            local_f = row.get("frame_id")
            try:
                global_frame = start + int(local_f)
            except (TypeError, ValueError):
                global_frame = start + int(idx)
            su = row.get("edge_src")
            sv = row.get("edge_dst")
            tu = None
            tv = None
            if su is not None:
                for tid, slot_name in summary.person_slot_assignments.items():
                    if str(slot_name) == str(su):
                        tu = str(tid)
                        break
            if sv is not None:
                for tid, slot_name in summary.person_slot_assignments.items():
                    if str(slot_name) == str(sv):
                        tv = str(tid)
                        break
            out_rows.append(
                {
                    "frame_id": global_frame,
                    "checkpoint_frame": checkpoint_frame,
                    "batch_start_frame": start,
                    "batch_end_frame": end,
                    "batch_local_index": int(idx),
                    "edge_src": su,
                    "edge_dst": sv,
                    "edge_src_track_id": tu,
                    "edge_dst_track_id": tv,
                    "source": "incremental_stateless_checkpoint",
                }
            )
    if not out_rows:
        return pd.DataFrame(columns=["frame_id", "edge_src", "edge_dst"])
    df = pd.DataFrame(out_rows).sort_values("frame_id", kind="stable").drop_duplicates(["frame_id"], keep="last")
    return df.reset_index(drop=True)


def _build_stateful_debug_dump(incremental_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoints = json.loads((incremental_dir / "runtime_checkpoints.json").read_text(encoding="utf-8"))
    frame_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for ck in checkpoints:
        checkpoint_frame = int(ck["frame_id"])
        batch = list(ck.get("raw_edge_batch") or [])
        if not batch:
            continue
        frames = [int(x["frame_id"]) for x in batch if x.get("frame_id") is not None]
        patterns = [f"{x.get('edge_src')}->{x.get('edge_dst')}" for x in batch]
        compressed = []
        last = None
        count = 0
        for p in patterns:
            if p == last:
                count += 1
            else:
                if last is not None:
                    compressed.append((last, count))
                last = p
                count = 1
        if last is not None:
            compressed.append((last, count))
        change_points = []
        for idx in range(1, len(patterns)):
            if patterns[idx] != patterns[idx - 1]:
                change_points.append(idx)
        cnt = Counter(patterns)
        top, topc = cnt.most_common(1)[0]
        for idx, edge in enumerate(batch):
            su = edge.get("edge_src")
            sv = edge.get("edge_dst")
            frame_rows.append(
                {
                    "frame": edge.get("frame_id"),
                    "checkpoint_frame": checkpoint_frame,
                    "batch_start_frame": min(frames) if frames else None,
                    "batch_end_frame": max(frames) if frames else None,
                    "batch_local_index": idx,
                    "slot_u_after_postprocess": su,
                    "slot_v_after_postprocess": sv,
                    "track_u_after_postprocess": edge.get("edge_src_track_id"),
                    "track_v_after_postprocess": edge.get("edge_dst_track_id"),
                    "canonical_u_after_postprocess": _slot_to_canonical(su),
                    "canonical_v_after_postprocess": _slot_to_canonical(sv),
                    "top_k_edges_logits": None,
                    "top_k_edges_canonical": None,
                    "entropy": None,
                    "margin_top1_top2": None,
                    "mask_valid": None,
                    "padding_valid": None,
                    "is_warmup": checkpoint_frame < 50,
                    "note_available_stage": "after_postprocess_only",
                }
            )
        batch_rows.append(
            {
                "checkpoint_frame": checkpoint_frame,
                "batch_start_frame": min(frames) if frames else None,
                "batch_end_frame": max(frames) if frames else None,
                "num_frames_in_batch": len(batch),
                "num_unique_edges_before_crf": None,
                "num_unique_edges_after_crf": None,
                "num_unique_edges_after_postprocess": len(cnt),
                "unique_edge_ratio_before_crf": None,
                "unique_edge_ratio_after_crf": None,
                "unique_edge_ratio_after_postprocess": len(cnt) / len(batch) if batch else None,
                "change_points_before_crf": None,
                "change_points_after_crf": None,
                "change_points_after_postprocess": json.dumps(change_points, ensure_ascii=False),
                "compressed_pattern_before_crf": None,
                "compressed_pattern_after_crf": None,
                "compressed_pattern_after_postprocess": json.dumps(compressed, ensure_ascii=False),
                "most_frequent_edge_before_crf": None,
                "most_frequent_edge_after_crf": None,
                "most_frequent_edge_after_postprocess": top,
                "most_frequent_edge_ratio_before_crf": None,
                "most_frequent_edge_ratio_after_crf": None,
                "most_frequent_edge_ratio_after_postprocess": topc / len(batch) if batch else None,
            }
        )
    return pd.DataFrame(frame_rows), pd.DataFrame(batch_rows)


def _by_pair(pred_df: pd.DataFrame, ref_df: pd.DataFrame, name_pred: str, name_ref: str, tol: int = 5) -> pd.DataFrame:
    pred_pts = _to_points(pred_df, "canonical")
    ref_pts = _to_points(ref_df, "canonical")
    pred_by = defaultdict(list)
    ref_by = defaultdict(list)
    for p in pred_pts:
        pred_by[(p[1], p[2])].append(p)
    for r in ref_pts:
        ref_by[(r[1], r[2])].append(r)
    rows = []
    for pair in sorted(set(pred_by.keys()) | set(ref_by.keys())):
        m = _match_metrics(pred_by.get(pair, []), ref_by.get(pair, []), tolerance=tol)
        rows.append(
            {
                "comparison": f"{name_pred}_vs_{name_ref}",
                "canonical_u": pair[0],
                "canonical_v": pair[1],
                "pred_count": len(pred_by.get(pair, [])),
                "ref_count": len(ref_by.get(pair, [])),
                **m,
            }
        )
    return pd.DataFrame(rows)


def _by_segment(pred_df: pd.DataFrame, ref_df: pd.DataFrame, name_pred: str, name_ref: str, tol: int = 5) -> pd.DataFrame:
    segs = [(40, 99), (100, 199), (200, 299), (300, 399), (400, 499), (500, 599), (600, 699), (700, 724)]
    rows = []
    for s, e in segs:
        m = _match_metrics(_to_points(_restrict_df(pred_df, s, e), "canonical"), _to_points(_restrict_df(ref_df, s, e), "canonical"), tolerance=tol)
        rows.append({"comparison": f"{name_pred}_vs_{name_ref}", "segment_start": s, "segment_end": e, **m})
    return pd.DataFrame(rows)


def _by_batch_local_index(pred_df: pd.DataFrame, ref_df: pd.DataFrame, name_pred: str, name_ref: str, tol: int = 5) -> pd.DataFrame:
    rows = []
    if "batch_local_index" not in pred_df.columns:
        return pd.DataFrame()
    for idx in sorted(set(pred_df["batch_local_index"].dropna().astype(int).tolist())):
        p = pred_df[pred_df["batch_local_index"] == idx].copy()
        m = _match_metrics(_to_points(p, "canonical"), _to_points(ref_df, "canonical"), tolerance=tol)
        rows.append({"comparison": f"{name_pred}_vs_{name_ref}", "batch_local_index": idx, **m})
    return pd.DataFrame(rows)


def _unmatched_examples(pred_df: pd.DataFrame, ref_df: pd.DataFrame, snapshot_df: pd.DataFrame, limit: int = 300) -> pd.DataFrame:
    pred = _to_points(pred_df, "canonical")
    ref = _to_points(ref_df, "canonical")
    snap = {int(r["frame_id"]): r for r in snapshot_df.to_dict(orient="records")}
    ref_map = defaultdict(list)
    for i, r in enumerate(ref):
        ref_map[r[0]].append(r)
    used = set()
    rows = []
    for p in pred:
        lo, hi = p[0] - 5, p[0] + 5
        best = None
        bestj = None
        for j, r in enumerate(ref):
            if j in used:
                continue
            if p[1] != r[1] or p[2] != r[2]:
                continue
            if not (lo <= r[0] <= hi):
                continue
            d = abs(r[0] - p[0])
            if best is None or d < best:
                best = d
                bestj = j
        if bestj is not None:
            used.add(bestj)
            continue
        near_pair = False
        for r in ref:
            if p[1] == r[1] and p[2] == r[2] and abs(r[0] - p[0]) <= 50:
                near_pair = True
                break
        reason = "temporal_shift" if near_pair else "wrong_pair"
        srow = snap.get(p[0], {})
        rows.append(
            {
                "frame": p[0],
                "edge_offline": ";".join([f"{r[1]}->{r[2]}" for r in ref_map.get(p[0], [])[:1]]) if ref_map.get(p[0]) else None,
                "edge_incremental": f"{p[1]}->{p[2]}",
                "edge_snapshot": f"{srow.get('edge_src')}->{srow.get('edge_dst')}" if srow else None,
                "reason_estimated": reason,
            }
        )
        if len(rows) >= limit:
            break
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parity ablation for incremental vs snapshot/offline.")
    parser.add_argument("--offline-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--incremental-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracks-path", type=Path, default=Path("output/tracks_json/tracker/partido_tracks.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    offline_dir = args.offline_dir.expanduser().resolve()
    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    incremental_dir = args.incremental_dir.expanduser().resolve()
    tracks_path = args.tracks_path.expanduser().resolve()
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))

    inc_summary = _read_json(incremental_dir / "summary.json")
    inc_ck = json.loads((incremental_dir / "runtime_checkpoints.json").read_text(encoding="utf-8"))
    snap_summary = _read_json(snapshot_dir / "live_snapshots_summary.json")
    off_summary = _read_json(offline_dir / "partido_summary.json")

    inference_cfg = PathCRFInferenceConfig(
        repo_path=Path(off_summary["repo_path"]),
        trial=int(off_summary["trial"]),
        model_file=Path(off_summary["model_path"]).name,
        use_crf=bool(off_summary["use_crf"]),
        decode=str(off_summary["decode"]),
        correct_episode_lasts=bool(off_summary["correct_episode_lasts"]),
        evaluate=bool(off_summary["evaluate"]),
        window_seconds=off_summary.get("window_seconds"),
        fps=float(off_summary["fps"]) if off_summary.get("fps") is not None else None,
        sample_freq=int(off_summary["sample_freq"]) if off_summary.get("sample_freq") is not None else None,
        min_event_duration=int(off_summary.get("min_event_duration", 10)),
        device=str(off_summary.get("device", "auto")),
    )
    runtime_cfg = ActionsRuntimeConfig(
        detector=ActionsDetectorConfig(
            fps=float(inc_summary["runtime_config"]["detector"]["fps"]),
            window_size_frames=inc_summary["runtime_config"]["detector"].get("window_size_frames"),
        ),
        repo_path=Path(inc_summary["runtime_config"]["repo_path"]),
        trial=int(inc_summary["runtime_config"]["trial"]),
        model_file=str(inc_summary["runtime_config"]["model_file"]),
        use_crf=bool(inc_summary["runtime_config"]["use_crf"]),
        decode=str(inc_summary["runtime_config"]["decode"]),
        correct_episode_lasts=bool(inc_summary["runtime_config"]["correct_episode_lasts"]),
        evaluate=bool(inc_summary["runtime_config"]["evaluate"]),
        cadence_frames=int(inc_summary["runtime_config"]["cadence_frames"]),
        min_frames_warmup=int(inc_summary["runtime_config"]["min_frames_warmup"]),
        min_event_duration=int(inc_summary["runtime_config"]["min_event_duration"]),
        window_seconds=inc_summary["runtime_config"].get("window_seconds"),
        fps=float(inc_summary["runtime_config"]["fps"]) if inc_summary["runtime_config"].get("fps") is not None else None,
        sample_freq=inc_summary["runtime_config"].get("sample_freq"),
        device=str(inc_summary["runtime_config"]["device"]),
        confirmation_cooldown_frames=int(inc_summary["runtime_config"]["confirmation_cooldown_frames"]),
        inference_batch_frames=inc_summary["runtime_config"].get("inference_batch_frames"),
    )

    # A) snapshot_on_incremental_windows
    snap_on_inc_df = _run_snapshot_on_incremental_windows(
        tracks=tracks,
        checkpoints=inc_ck,
        out_dir=out_dir,
        inference_config=inference_cfg,
    )
    snap_on_inc_path = out_dir / "snapshot_on_incremental_windows_edges.parquet"
    snap_on_inc_df.to_parquet(snap_on_inc_path, index=False)

    # B) incremental_stateless_checkpoint
    inc_stateless_df = _run_incremental_stateless_checkpoint(
        tracks=tracks,
        checkpoints=inc_ck,
        runtime_config=runtime_cfg,
    )
    inc_stateless_path = out_dir / "incremental_stateless_edges.parquet"
    inc_stateless_df.to_parquet(inc_stateless_path, index=False)

    # C) stateful debug dump from current run
    stateful_debug_df, batch_debug_df = _build_stateful_debug_dump(incremental_dir)
    stateful_debug_path = out_dir / "incremental_stateful_debug_dump.parquet"
    batch_debug_path = out_dir / "incremental_batch_decode_debug.csv"
    stateful_debug_df.to_parquet(stateful_debug_path, index=False)
    batch_debug_df.to_csv(batch_debug_path, index=False)

    offline_edges = _read_parquet_with_frame(offline_dir / "partido_edge_sequence.parquet")
    snapshot_full_edges = _read_parquet_with_frame(snapshot_dir / "live_sparse_edge_sequence.parquet")
    incremental_stateful_edges = _read_parquet_with_frame(incremental_dir / "internal_edges_per_frame.parquet")

    # Harmonize intersection fixed 40..724
    start, end = 40, 724
    off_i = _restrict_df(offline_edges, start, end)
    snap_full_i = _restrict_df(snapshot_full_edges, start, end)
    inc_state_i = _restrict_df(incremental_stateful_edges, start, end)
    snap_on_inc_i = _restrict_df(snap_on_inc_df, start, end)
    inc_stateless_i = _restrict_df(inc_stateless_df, start, end)

    def metric(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
        return _match_metrics(_to_points(a, "canonical"), _to_points(b, "canonical"), tolerance=5)

    comparisons = {
        "snapshot_on_incremental_windows_vs_offline": metric(snap_on_inc_i, off_i),
        "snapshot_on_incremental_windows_vs_snapshot_full": metric(snap_on_inc_i, snap_full_i),
        "snapshot_on_incremental_windows_vs_incremental_stateful_actual": metric(snap_on_inc_i, inc_state_i),
        "incremental_stateless_vs_offline": metric(inc_stateless_i, off_i),
        "incremental_stateless_vs_snapshot_full": metric(inc_stateless_i, snap_full_i),
        "incremental_stateless_vs_snapshot_on_incremental_windows": metric(inc_stateless_i, snap_on_inc_i),
        "incremental_stateless_vs_incremental_stateful_actual": metric(inc_stateless_i, inc_state_i),
        "incremental_stateful_actual_vs_offline": metric(inc_state_i, off_i),
    }

    by_pair = pd.concat(
        [
            _by_pair(snap_on_inc_i, off_i, "snapshot_on_incremental_windows", "offline"),
            _by_pair(inc_stateless_i, off_i, "incremental_stateless", "offline"),
            _by_pair(inc_state_i, off_i, "incremental_stateful_actual", "offline"),
        ],
        ignore_index=True,
    )
    by_pair.to_csv(out_dir / "parity_ablation_by_pair.csv", index=False)

    by_segment = pd.concat(
        [
            _by_segment(snap_on_inc_i, off_i, "snapshot_on_incremental_windows", "offline"),
            _by_segment(inc_stateless_i, off_i, "incremental_stateless", "offline"),
            _by_segment(inc_state_i, off_i, "incremental_stateful_actual", "offline"),
        ],
        ignore_index=True,
    )
    by_segment.to_csv(out_dir / "parity_ablation_by_segment.csv", index=False)

    by_batch_local_idx = pd.concat(
        [
            _by_batch_local_index(snap_on_inc_i, off_i, "snapshot_on_incremental_windows", "offline"),
            _by_batch_local_index(inc_stateless_i, off_i, "incremental_stateless", "offline"),
            _by_batch_local_index(inc_state_i, off_i, "incremental_stateful_actual", "offline"),
        ],
        ignore_index=True,
    )
    by_batch_local_idx.to_csv(out_dir / "parity_ablation_by_batch_local_index.csv", index=False)

    unmatched = _unmatched_examples(inc_state_i, off_i, snap_on_inc_i, limit=400)
    unmatched.to_csv(out_dir / "parity_ablation_unmatched_examples.csv", index=False)

    # Pattern check (5,4,1)/(5,5)/(9,1)/(10)
    pattern_rows = []
    if not batch_debug_df.empty:
        for _, row in batch_debug_df.iterrows():
            comp = str(row.get("compressed_pattern_after_postprocess"))
            pattern_rows.append(comp)
    pattern_counter = Counter(pattern_rows)
    batch_local_counter = Counter(stateful_debug_df["batch_local_index"].dropna().astype(int).tolist()) if not stateful_debug_df.empty else Counter()
    change_at_idx_counter = Counter()
    if not stateful_debug_df.empty:
        for ck, g in stateful_debug_df.groupby("checkpoint_frame"):
            g2 = g.sort_values("batch_local_index")
            vals = (g2["slot_u_after_postprocess"].astype(str) + "->" + g2["slot_v_after_postprocess"].astype(str)).tolist()
            idxs = g2["batch_local_index"].astype(int).tolist()
            for i in range(1, len(vals)):
                if vals[i] != vals[i - 1]:
                    change_at_idx_counter[idxs[i]] += 1

    effective_cfg = {
        "offline_summary": off_summary,
        "snapshot_summary": snap_summary,
        "incremental_summary": inc_summary,
        "runtime_config_used_for_stateless": asdict(runtime_cfg),
        "inference_config_used_for_snapshot_on_incremental_windows": asdict(inference_cfg),
        "intersection_frames": {"start": start, "end": end},
        "tolerance_frames": 5,
    }
    _write_json(out_dir / "parity_ablation_effective_configs.json", effective_cfg)

    summary = {
        "summary": {
            "main_findings": [],
            "likely_root_causes": [],
            "recommended_next_steps": [],
        },
        "comparisons": comparisons,
        "pattern_analysis": {
            "compressed_pattern_after_postprocess_top": pattern_counter.most_common(10),
            "change_points_after_postprocess_by_batch_local_index": dict(change_at_idx_counter),
            "batch_local_index_counts": dict(batch_local_counter),
            "available_stages": [
                "after_postprocess",
            ],
            "missing_stages": [
                "before_crf",
                "after_crf_pre_postprocess",
                "logits_top_k",
                "entropy",
                "mask/padding internals",
            ],
        },
        "questions": {
            "1_snapshot_on_incremental_windows_similar_to_offline": comparisons["snapshot_on_incremental_windows_vs_offline"],
            "2_snapshot_on_incremental_windows_similar_to_snapshot_full": comparisons["snapshot_on_incremental_windows_vs_snapshot_full"],
            "3_incremental_stateless_improves_vs_stateful": {
                "stateless_vs_offline_f1": comparisons["incremental_stateless_vs_offline"]["f1"],
                "stateful_vs_offline_f1": comparisons["incremental_stateful_actual_vs_offline"]["f1"],
            },
            "4_incremental_stateless_similar_to_snapshot_on_incremental_windows": comparisons["incremental_stateless_vs_snapshot_on_incremental_windows"],
            "5_pattern_stage_visibility": "only_after_postprocess_visible_in_current_runtime_artifacts",
            "6_pattern_associated_with_batch_local_index_5_9": {
                "counts": dict(change_at_idx_counter),
                "is_5_or_9_prominent": (change_at_idx_counter.get(5, 0) + change_at_idx_counter.get(9, 0)) > 0,
            },
            "7_padding_masking_warmup_signals": {
                "warmup_in_stateful_debug_rows": int((stateful_debug_df["is_warmup"] == True).sum()) if not stateful_debug_df.empty else 0,  # noqa: E712
                "padding_masking_available": False,
            },
            "8_most_likely_primary_cause": None,
            "9_next_code_change_recommended": None,
        },
    }

    f1_a = comparisons["snapshot_on_incremental_windows_vs_offline"]["f1"] or 0.0
    f1_b = comparisons["incremental_stateless_vs_offline"]["f1"] or 0.0
    f1_c = comparisons["incremental_stateful_actual_vs_offline"]["f1"] or 0.0
    if f1_a > 0.7 and f1_b < 0.2:
        cause = "B/C/E/F/G: runtime incremental path diverges from snapshot decoder path even on same windows."
    elif f1_b > f1_c * 2:
        cause = "B: estado causal contamina fuertemente la predicción."
    elif f1_a < 0.3:
        cause = "A: composición/longitud de ventana insuficiente incluso con decoder snapshot."
    else:
        cause = "F/G: asignación temporal/frame-level o postproceso incremental."
    summary["questions"]["8_most_likely_primary_cause"] = cause
    summary["questions"]["9_next_code_change_recommended"] = (
        "Instrumentar runtime para exponer salida pre/post CRF por frame y comparar contra snapshot_on_incremental_windows en mismos checkpoints."
    )
    summary["summary"]["main_findings"] = [
        "Se ejecutaron variantes A/B/C en intersección 40..724 con métrica canonical ±5.",
        "Se evaluó paridad con mismas ventanas de checkpoints incremental.",
        "Se exportó debug por batch_local_index y patrones comprimidos after_postprocess.",
    ]
    summary["summary"]["likely_root_causes"] = [cause]
    summary["summary"]["recommended_next_steps"] = [
        "Agregar hooks before_crf/after_crf en ActionsRuntime para confirmar si el patrón nace antes o después de CRF.",
        "Alinear estrictamente decoder/postprocess incremental con snapshot en el mismo batch y comparar frame a frame.",
    ]

    _write_json(out_dir / "parity_ablation_summary.json", summary)

    print(f"Output dir: {out_dir}")
    print(f"Summary: {out_dir / 'parity_ablation_summary.json'}")


if __name__ == "__main__":
    main()
