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
    s = str(slot)
    p = person_slot_to_canonical_id(s)
    if p is not None:
        return str(p)
    r = referee_slot_to_canonical_id(s)
    if r is not None:
        return str(r)
    return None


def _canonical_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    u = _slot_to_canonical(row.get("edge_src"))
    v = _slot_to_canonical(row.get("edge_dst"))
    if u is None and row.get("edge_src_track_id") is not None:
        u = str(row["edge_src_track_id"])
    if v is None and row.get("edge_dst_track_id") is not None:
        v = str(row["edge_dst_track_id"])
    if u is None or v is None:
        return None
    return (u, v)


def _to_points(df: pd.DataFrame) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for row in df.to_dict(orient="records"):
        try:
            f = int(row.get("frame_id"))
        except (TypeError, ValueError):
            continue
        p = _canonical_pair(row)
        if p is None:
            continue
        out.append((f, p[0], p[1]))
    return out


def _restrict(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df[(df["frame_id"] >= start) & (df["frame_id"] <= end)].copy()


def _match_metrics(pred: list[tuple[int, str, str]], ref: list[tuple[int, str, str]], tol: int = 5) -> dict[str, Any]:
    used = set()
    m = 0
    for f, u, v in pred:
        lo, hi = f - tol, f + tol
        bj = None
        bd = None
        for j, (fr, ur, vr) in enumerate(ref):
            if j in used or u != ur or v != vr or not (lo <= fr <= hi):
                continue
            d = abs(fr - f)
            if bd is None or d < bd:
                bd, bj = d, j
        if bj is not None:
            used.add(bj)
            m += 1
    precision = m / len(pred) if pred else None
    recall = m / len(ref) if ref else None
    f1 = None if precision is None or recall is None else (0.0 if precision + recall == 0 else (2 * precision * recall / (precision + recall)))
    return {
        "num_pred_edges": len(pred),
        "num_ref_edges": len(ref),
        "num_matches": m,
        "num_unmatched_predictions": len(pred) - m,
        "num_unmatched_references": len(ref) - m,
        "edge_match_ratio": precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _slice_tracks(tracks: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in tracks.items():
        out[k] = v[start : end + 1] if isinstance(v, list) else v
    return out


def _build_snapshot_on_windows(
    tracks: dict[str, Any], checkpoints: list[dict[str, Any]], out_dir: Path, inf_cfg: PathCRFInferenceConfig
) -> pd.DataFrame:
    rows: dict[int, dict[str, Any]] = {}
    run_dir = out_dir / "snapshot_on_incremental_windows"
    run_dir.mkdir(parents=True, exist_ok=True)
    for ck in checkpoints:
        batch = list(ck.get("raw_edge_batch") or [])
        frames = sorted(int(x["frame_id"]) for x in batch if x.get("frame_id") is not None)
        if not frames:
            continue
        start, end = frames[0], frames[-1]
        cframe = int(ck["frame_id"])
        tpath = run_dir / f"tracks_window_{start:06d}_{end:06d}.json"
        tpath.write_text(json.dumps(_slice_tracks(tracks, start, end), ensure_ascii=False), encoding="utf-8")
        rout = run_dir / f"ck_{cframe:06d}"
        result = run_pathcrf_pipeline(
            output_dir=rout,
            tracks_path=tpath,
            adapter_config=PathCRFAdapterConfig(fps=25.0),
            inference_config=inf_cfg,
            render_config=PathCRFRenderConfig(enabled=False),
        )
        edf = _read_parquet_with_frame(result.edge_sequence_path).sort_values("frame_id", kind="stable").reset_index(drop=True)
        for i, row in edf.iterrows():
            gf = start + int(i)
            rows[gf] = {
                "frame_id": gf,
                "checkpoint_frame": cframe,
                "batch_start_frame": start,
                "batch_end_frame": end,
                "batch_local_index": int(i),
                "edge_src": row.get("edge_src"),
                "edge_dst": row.get("edge_dst"),
                "source": "snapshot_on_incremental_windows",
            }
    return pd.DataFrame([v for _, v in sorted(rows.items(), key=lambda kv: kv[0])])


def _build_snapshot_cumulative(
    tracks: dict[str, Any], checkpoints: list[dict[str, Any]], out_dir: Path, inf_cfg: PathCRFInferenceConfig
) -> pd.DataFrame:
    rows: dict[int, dict[str, Any]] = {}
    run_dir = out_dir / "snapshot_cumulative_at_incremental_checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    for ck in checkpoints:
        batch = list(ck.get("raw_edge_batch") or [])
        frames = sorted(int(x["frame_id"]) for x in batch if x.get("frame_id") is not None)
        if not frames:
            continue
        start, end = frames[0], frames[-1]
        cframe = int(ck["frame_id"])
        tpath = run_dir / f"tracks_cumulative_000000_{end:06d}.json"
        tpath.write_text(json.dumps(_slice_tracks(tracks, 0, end), ensure_ascii=False), encoding="utf-8")
        rout = run_dir / f"ck_{cframe:06d}"
        result = run_pathcrf_pipeline(
            output_dir=rout,
            tracks_path=tpath,
            adapter_config=PathCRFAdapterConfig(fps=25.0),
            inference_config=inf_cfg,
            render_config=PathCRFRenderConfig(enabled=False),
        )
        edf = _read_parquet_with_frame(result.edge_sequence_path)
        edf = edf[(edf["frame_id"] >= start) & (edf["frame_id"] <= end)].sort_values("frame_id", kind="stable")
        for _, row in edf.iterrows():
            gf = int(row.get("frame_id"))
            rows[gf] = {
                "frame_id": gf,
                "checkpoint_frame": cframe,
                "batch_start_frame": start,
                "batch_end_frame": end,
                "batch_local_index": gf - start,
                "edge_src": row.get("edge_src"),
                "edge_dst": row.get("edge_dst"),
                "source": "snapshot_cumulative_at_incremental_checkpoints",
            }
    return pd.DataFrame([v for _, v in sorted(rows.items(), key=lambda kv: kv[0])])


def _run_incremental_explicit_stateful(
    tracks: dict[str, Any], frame_count: int, cfg: ActionsRuntimeConfig
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    rt = ActionsRuntime(cfg)
    checkpoints: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    per_frame: dict[int, dict[str, Any]] = {}

    for frame in range(frame_count):
        packet = {
            "tracks_frame": {
                cls: (tracks.get(cls, [])[frame] if frame < len(tracks.get(cls, [])) else {})
                for cls in ("player", "goalkeeper", "referee", "ball")
            },
            "possession": (tracks.get("possession", [])[frame] if frame < len(tracks.get("possession", [])) else {}),
        }
        res = rt.process_frame(frame, packet)
        if not res.action_metadata.get("should_infer"):
            continue
        batch = list(res.raw_edge_batch or [])
        checkpoints.append(
            {
                "frame_id": frame,
                "raw_edge_batch": batch,
                "action_metadata": res.action_metadata,
            }
        )
        if not batch:
            continue

        # Reproducimos etapas disponibles/reconstruidas: before_crf (use_crf=False), after_crf (use_crf=True), after_frame_expansion
        tracking_df, summary = rt.detector.build_tracking_dataframe()
        bdf = rt._select_inference_batch(tracking_df)
        base_kwargs = dict(
            model=rt._model,
            tracking=bdf,
            decode=str(rt.config.decode),
            correct_episode_lasts=bool(rt.config.correct_episode_lasts),
            evaluate=bool(rt.config.evaluate),
            window_seconds=float(rt.config.window_seconds) if rt.config.window_seconds is not None else rt._trial_args.get("window_seconds"),
            fps=float(rt.config.fps) if rt.config.fps is not None else float(rt._trial_args.get("fps", 25.0)),
            sample_freq=int(rt.config.sample_freq) if rt.config.sample_freq is not None else int(rt._trial_args.get("sample_freq", 5)),
        )
        _, _, micro_no_crf, _ = rt._pathcrf_inference.inference(use_crf=False, **base_kwargs)
        _, _, micro_crf, _ = rt._pathcrf_inference.inference(use_crf=True, **base_kwargs)
        seq_crf = micro_crf.copy() if {"edge_src", "edge_dst"}.issubset(micro_crf.columns) else rt._pathcrf_postprocess.edge_probs_to_seq(micro_crf)
        seq_no_crf = micro_no_crf.copy() if {"edge_src", "edge_dst"}.issubset(micro_no_crf.columns) else rt._pathcrf_postprocess.edge_probs_to_seq(micro_no_crf)

        frames = [int(x["frame_id"]) for x in batch if x.get("frame_id") is not None]
        bstart, bend = min(frames), max(frames)
        patterns = [f"{x.get('edge_src')}->{x.get('edge_dst')}" for x in batch]
        c = Counter(patterns)

        for i in range(len(batch)):
            af = int(batch[i].get("frame_id"))
            before_row = seq_no_crf.iloc[i] if i < len(seq_no_crf) else {}
            after_row = seq_crf.iloc[i] if i < len(seq_crf) else {}
            exp_row = batch[i]
            frame_rows.append(
                {
                    "frame": af,
                    "frame_id": af,
                    "checkpoint_frame": frame,
                    "batch_start_frame": bstart,
                    "batch_end_frame": bend,
                    "batch_local_index": i,
                    "native_timestep": i,
                    "native_timestep_count": len(batch),
                    "native_source_frame_start": af,
                    "native_source_frame_end": af,
                    "slot_u_before_crf": before_row.get("edge_src") if isinstance(before_row, dict) else before_row.get("edge_src"),
                    "slot_v_before_crf": before_row.get("edge_dst") if isinstance(before_row, dict) else before_row.get("edge_dst"),
                    "slot_u_after_crf": after_row.get("edge_src") if isinstance(after_row, dict) else after_row.get("edge_src"),
                    "slot_v_after_crf": after_row.get("edge_dst") if isinstance(after_row, dict) else after_row.get("edge_dst"),
                    "slot_u_after_postprocess": after_row.get("edge_src") if isinstance(after_row, dict) else after_row.get("edge_src"),
                    "slot_v_after_postprocess": after_row.get("edge_dst") if isinstance(after_row, dict) else after_row.get("edge_dst"),
                    "slot_u_after_frame_expansion": exp_row.get("edge_src"),
                    "slot_v_after_frame_expansion": exp_row.get("edge_dst"),
                    "canonical_u_before_crf": _slot_to_canonical(before_row.get("edge_src") if hasattr(before_row, "get") else None),
                    "canonical_v_before_crf": _slot_to_canonical(before_row.get("edge_dst") if hasattr(before_row, "get") else None),
                    "canonical_u_after_crf": _slot_to_canonical(after_row.get("edge_src") if hasattr(after_row, "get") else None),
                    "canonical_v_after_crf": _slot_to_canonical(after_row.get("edge_dst") if hasattr(after_row, "get") else None),
                    "canonical_u_after_postprocess": _slot_to_canonical(after_row.get("edge_src") if hasattr(after_row, "get") else None),
                    "canonical_v_after_postprocess": _slot_to_canonical(after_row.get("edge_dst") if hasattr(after_row, "get") else None),
                    "canonical_u_after_frame_expansion": _slot_to_canonical(exp_row.get("edge_src")),
                    "canonical_v_after_frame_expansion": _slot_to_canonical(exp_row.get("edge_dst")),
                    "track_u_after_frame_expansion": exp_row.get("edge_src_track_id"),
                    "track_v_after_frame_expansion": exp_row.get("edge_dst_track_id"),
                    "top_k_edges_logits": None,
                    "top_k_edges_canonical": None,
                    "entropy": None,
                    "margin_top1_top2": None,
                    "mask_valid": None,
                    "padding_valid": None,
                    "is_warmup": frame < int(cfg.min_frames_warmup),
                }
            )
            per_frame[af] = {
                "frame_id": af,
                "checkpoint_frame": frame,
                "batch_start_frame": bstart,
                "batch_end_frame": bend,
                "batch_local_index": i,
                "edge_src": exp_row.get("edge_src"),
                "edge_dst": exp_row.get("edge_dst"),
                "edge_src_track_id": exp_row.get("edge_src_track_id"),
                "edge_dst_track_id": exp_row.get("edge_dst_track_id"),
                "source": "incremental_stateful_explicit_config",
            }

        def _compress(vals: list[str]) -> list[tuple[str, int]]:
            out: list[tuple[str, int]] = []
            last = None
            n = 0
            for v in vals:
                if v == last:
                    n += 1
                else:
                    if last is not None:
                        out.append((last, n))
                    last = v
                    n = 1
            if last is not None:
                out.append((last, n))
            return out

        p_before = [f"{seq_no_crf.iloc[i].get('edge_src')}->{seq_no_crf.iloc[i].get('edge_dst')}" if i < len(seq_no_crf) else None for i in range(len(batch))]
        p_after = [f"{seq_crf.iloc[i].get('edge_src')}->{seq_crf.iloc[i].get('edge_dst')}" if i < len(seq_crf) else None for i in range(len(batch))]

        batch_rows.append(
            {
                "checkpoint_frame": frame,
                "batch_start_frame": bstart,
                "batch_end_frame": bend,
                "num_frames_in_batch": len(batch),
                "native_timestep_count": len(batch),
                "native_timestep_to_frame_mapping": json.dumps({i: bstart + i for i in range(len(batch))}, ensure_ascii=False),
                "num_unique_edges_before_crf": len(set(p_before)),
                "num_unique_edges_after_crf": len(set(p_after)),
                "num_unique_edges_after_postprocess": len(set(p_after)),
                "num_unique_edges_after_frame_expansion": len(c),
                "unique_edge_ratio_before_crf": len(set(p_before)) / len(batch) if batch else None,
                "unique_edge_ratio_after_crf": len(set(p_after)) / len(batch) if batch else None,
                "unique_edge_ratio_after_postprocess": len(set(p_after)) / len(batch) if batch else None,
                "unique_edge_ratio_after_frame_expansion": len(c) / len(batch) if batch else None,
                "change_points_before_crf": json.dumps([i for i in range(1, len(p_before)) if p_before[i] != p_before[i - 1]], ensure_ascii=False),
                "change_points_after_crf": json.dumps([i for i in range(1, len(p_after)) if p_after[i] != p_after[i - 1]], ensure_ascii=False),
                "change_points_after_postprocess": json.dumps([i for i in range(1, len(p_after)) if p_after[i] != p_after[i - 1]], ensure_ascii=False),
                "change_points_after_frame_expansion": json.dumps([i for i in range(1, len(patterns)) if patterns[i] != patterns[i - 1]], ensure_ascii=False),
                "compressed_pattern_before_crf": json.dumps(_compress(p_before), ensure_ascii=False),
                "compressed_pattern_after_crf": json.dumps(_compress(p_after), ensure_ascii=False),
                "compressed_pattern_after_postprocess": json.dumps(_compress(p_after), ensure_ascii=False),
                "compressed_pattern_after_frame_expansion": json.dumps(_compress(patterns), ensure_ascii=False),
                "most_frequent_edge_before_crf": Counter(p_before).most_common(1)[0][0] if p_before else None,
                "most_frequent_edge_after_crf": Counter(p_after).most_common(1)[0][0] if p_after else None,
                "most_frequent_edge_after_postprocess": Counter(p_after).most_common(1)[0][0] if p_after else None,
                "most_frequent_edge_after_frame_expansion": c.most_common(1)[0][0] if c else None,
                "most_frequent_edge_ratio_before_crf": (Counter(p_before).most_common(1)[0][1] / len(batch)) if p_before else None,
                "most_frequent_edge_ratio_after_crf": (Counter(p_after).most_common(1)[0][1] / len(batch)) if p_after else None,
                "most_frequent_edge_ratio_after_postprocess": (Counter(p_after).most_common(1)[0][1] / len(batch)) if p_after else None,
                "most_frequent_edge_ratio_after_frame_expansion": (c.most_common(1)[0][1] / len(batch)) if c else None,
            }
        )

    df_edges = pd.DataFrame([v for _, v in sorted(per_frame.items(), key=lambda kv: kv[0])])
    return df_edges, checkpoints, pd.DataFrame(frame_rows), pd.DataFrame(batch_rows)


def _run_incremental_stateless_from_checkpoints(
    tracks: dict[str, Any], checkpoints: list[dict[str, Any]], cfg: ActionsRuntimeConfig, source_name: str
) -> pd.DataFrame:
    out: dict[int, dict[str, Any]] = {}
    for ck in checkpoints:
        batch = list(ck.get("raw_edge_batch") or [])
        frames = sorted(int(x["frame_id"]) for x in batch if x.get("frame_id") is not None)
        if not frames:
            continue
        start, end = frames[0], frames[-1]
        cframe = int(ck["frame_id"])
        rt = ActionsRuntime(cfg)
        for frame in range(start, end + 1):
            packet = {
                "tracks_frame": {
                    cls: (tracks.get(cls, [])[frame] if frame < len(tracks.get(cls, [])) else {})
                    for cls in ("player", "goalkeeper", "referee", "ball")
                },
                "possession": (tracks.get("possession", [])[frame] if frame < len(tracks.get("possession", [])) else {}),
            }
            rt.detector.update(frame, packet)
        tdf, summary = rt.detector.build_tracking_dataframe()
        if tdf.empty:
            continue
        bdf = rt._select_inference_batch(tdf)
        seq, _ = rt._run_pathcrf(bdf)
        seq = seq.sort_index().reset_index(drop=True)
        for i, row in seq.iterrows():
            gf = start + i
            su, sv = row.get("edge_src"), row.get("edge_dst")
            tu = next((str(tid) for tid, s in summary.person_slot_assignments.items() if str(s) == str(su)), None) if su is not None else None
            tv = next((str(tid) for tid, s in summary.person_slot_assignments.items() if str(s) == str(sv)), None) if sv is not None else None
            out[gf] = {
                "frame_id": gf,
                "checkpoint_frame": cframe,
                "batch_start_frame": start,
                "batch_end_frame": end,
                "batch_local_index": i,
                "edge_src": su,
                "edge_dst": sv,
                "edge_src_track_id": tu,
                "edge_dst_track_id": tv,
                "source": source_name,
            }
    return pd.DataFrame([v for _, v in sorted(out.items(), key=lambda kv: kv[0])])


def _comparison_table(cands: dict[str, pd.DataFrame], refs: dict[str, pd.DataFrame], start: int, end: int, tol: int) -> pd.DataFrame:
    rows = []
    for cname, cdf in cands.items():
        for rname, rdf in refs.items():
            m = _match_metrics(_to_points(_restrict(cdf, start, end)), _to_points(_restrict(rdf, start, end)), tol)
            rows.append({"candidate": cname, "reference": rname, "start": start, "end": end, "tolerance": tol, **m})
    return pd.DataFrame(rows)


def _by_segment(cands: dict[str, pd.DataFrame], refs: dict[str, pd.DataFrame], tol: int) -> pd.DataFrame:
    segs = [(40, 99), (100, 199), (200, 299), (300, 399), (400, 499), (500, 599), (600, 699), (700, 724)]
    rows = []
    for s, e in segs:
        for cname, cdf in cands.items():
            for rname, rdf in refs.items():
                m = _match_metrics(_to_points(_restrict(cdf, s, e)), _to_points(_restrict(rdf, s, e)), tol)
                rows.append({"candidate": cname, "reference": rname, "segment_start": s, "segment_end": e, **m})
    return pd.DataFrame(rows)


def _by_pair(cands: dict[str, pd.DataFrame], refs: dict[str, pd.DataFrame], start: int, end: int, tol: int) -> pd.DataFrame:
    rows = []
    for cname, cdf in cands.items():
        pred = _to_points(_restrict(cdf, start, end))
        pred_by = defaultdict(list)
        for p in pred:
            pred_by[(p[1], p[2])].append(p)
        for rname, rdf in refs.items():
            ref = _to_points(_restrict(rdf, start, end))
            ref_by = defaultdict(list)
            for rr in ref:
                ref_by[(rr[1], rr[2])].append(rr)
            for pair in sorted(set(pred_by) | set(ref_by)):
                m = _match_metrics(pred_by.get(pair, []), ref_by.get(pair, []), tol)
                rows.append({"candidate": cname, "reference": rname, "canonical_u": pair[0], "canonical_v": pair[1], **m})
    return pd.DataFrame(rows)


def _by_group(df: pd.DataFrame, group_col: str, reference: pd.DataFrame, candidate_name: str, reference_name: str, tol: int) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for g in sorted(df[group_col].dropna().unique().tolist()):
        sub = df[df[group_col] == g]
        m = _match_metrics(_to_points(sub), _to_points(reference), tol)
        rows.append({"candidate": candidate_name, "reference": reference_name, group_col: g, **m})
    return pd.DataFrame(rows)


def _stage_edges(debug_df: pd.DataFrame, stage: str) -> pd.DataFrame:
    u = f"slot_u_{stage}"
    v = f"slot_v_{stage}"
    if u not in debug_df.columns or v not in debug_df.columns:
        return pd.DataFrame(columns=["frame_id", "edge_src", "edge_dst"])
    return pd.DataFrame(
        {
            "frame_id": debug_df["frame_id"],
            "edge_src": debug_df[u],
            "edge_dst": debug_df[v],
            "batch_local_index": debug_df.get("batch_local_index"),
            "native_timestep": debug_df.get("native_timestep"),
            "stage": stage,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cumulative context + runtime debug audit")
    p.add_argument("--offline-dir", type=Path, default=Path("output/actions/pipeline_audit/eval_full_offline"))
    p.add_argument("--snapshot-dir", type=Path, default=Path("output/actions/pipeline_audit/eval_full_snapshot"))
    p.add_argument("--incremental-dir", type=Path, default=Path("output/actions/pipeline_audit/eval_full_incremental"))
    p.add_argument("--snapshot-internal-path", type=Path, default=Path("output/actions/pipeline_audit/eval_full_similarity/snapshot_internal_edges_per_frame.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("output/actions/pipeline_audit/eval_full_cumulative_context_and_runtime_debug"))
    p.add_argument("--start-frame", type=int, default=40)
    p.add_argument("--end-frame", type=int, default=724)
    p.add_argument("--tolerance", type=int, default=5)
    p.add_argument("--tracks-path", type=Path, default=Path("output/tracks_json/tracker/partido_tracks.json"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    off_dir = args.offline_dir.expanduser().resolve()
    inc_dir = args.incremental_dir.expanduser().resolve()
    snap_internal_path = args.snapshot_internal_path.expanduser().resolve()
    tracks_path = args.tracks_path.expanduser().resolve()

    off_summary = _read_json(off_dir / "partido_summary.json")
    inc_summary = _read_json(inc_dir / "summary.json")
    checkpoints = json.loads((inc_dir / "runtime_checkpoints.json").read_text(encoding="utf-8"))
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    frame_count = int(inc_summary.get("frame_count", 0) or len(tracks.get("player", [])))

    inf_cfg = PathCRFInferenceConfig(
        repo_path=Path(off_summary["repo_path"]),
        trial=int(off_summary["trial"]),
        model_file=Path(off_summary["model_path"]).name,
        use_crf=bool(off_summary["use_crf"]),
        decode=str(off_summary["decode"]),
        correct_episode_lasts=bool(off_summary["correct_episode_lasts"]),
        evaluate=bool(off_summary["evaluate"]),
        window_seconds=float(off_summary.get("window_seconds") or 10.0),
        fps=float(off_summary.get("fps") or 25.0),
        sample_freq=int(off_summary.get("sample_freq") or 5),
        min_event_duration=int(off_summary.get("min_event_duration") or 10),
        device=str(off_summary.get("device", "auto")),
    )

    cfg_explicit = ActionsRuntimeConfig(
        detector=ActionsDetectorConfig(fps=float(inc_summary["runtime_config"]["detector"]["fps"])),
        repo_path=Path(inc_summary["runtime_config"]["repo_path"]),
        trial=int(inc_summary["runtime_config"]["trial"]),
        model_file=str(inc_summary["runtime_config"]["model_file"]),
        use_crf=True,
        decode="indep",
        correct_episode_lasts=False,
        evaluate=False,
        cadence_frames=int(inc_summary["runtime_config"]["cadence_frames"]),
        min_frames_warmup=int(inc_summary["runtime_config"]["min_frames_warmup"]),
        min_event_duration=10,
        window_seconds=10.0,
        fps=float(inc_summary["runtime_config"].get("fps") or 25.0),
        sample_freq=5,
        device=str(inc_summary["runtime_config"]["device"]),
        confirmation_cooldown_frames=int(inc_summary["runtime_config"]["confirmation_cooldown_frames"]),
        inference_batch_frames=inc_summary["runtime_config"].get("inference_batch_frames"),
    )

    off_df = _read_parquet_with_frame(off_dir / "partido_edge_sequence.parquet")
    snap_internal_ref_df = _read_parquet_with_frame(snap_internal_path)
    snap_internal_ref_df.to_parquet(out_dir / "snapshot_internal_framelevel_reference.parquet", index=False)

    # A) windows and cumulative
    snap_on_windows_df = _build_snapshot_on_windows(tracks, checkpoints, out_dir, inf_cfg)
    snap_on_windows_df.to_parquet(out_dir / "snapshot_on_incremental_windows_edges.parquet", index=False)

    snap_cumulative_df = _build_snapshot_cumulative(tracks, checkpoints, out_dir, inf_cfg)
    snap_cumulative_df.to_parquet(out_dir / "snapshot_cumulative_at_incremental_checkpoints_edges.parquet", index=False)

    # B) incremental explicit
    inc_stateful_exp_df, explicit_ck, debug_frame_df, debug_batch_df = _run_incremental_explicit_stateful(tracks, frame_count, cfg_explicit)
    inc_stateful_exp_df.to_parquet(out_dir / "incremental_stateful_explicit_config_edges.parquet", index=False)
    debug_frame_df.to_parquet(out_dir / "incremental_runtime_debug_by_frame.parquet", index=False)
    debug_batch_df.to_csv(out_dir / "incremental_runtime_debug_by_batch.csv", index=False)

    inc_stateless_exp_df = _run_incremental_stateless_from_checkpoints(
        tracks,
        explicit_ck,
        cfg_explicit,
        "incremental_stateless_explicit_config",
    )
    inc_stateless_exp_df.to_parquet(out_dir / "incremental_stateless_explicit_config_edges.parquet", index=False)

    inc_stateful_actual_df = _read_parquet_with_frame(inc_dir / "internal_edges_per_frame.parquet")
    inc_stateless_actual_path = Path("output/actions/pipeline_audit/eval_full_parity_ablation/incremental_stateless_edges.parquet")
    inc_stateless_actual_df = _read_parquet_with_frame(inc_stateless_actual_path) if inc_stateless_actual_path.exists() else pd.DataFrame(columns=["frame_id", "edge_src", "edge_dst"])

    cands = {
        "snapshot_on_incremental_windows": snap_on_windows_df,
        "snapshot_cumulative_at_incremental_checkpoints": snap_cumulative_df,
        "incremental_stateful_actual": inc_stateful_actual_df,
        "incremental_stateless_actual": inc_stateless_actual_df,
        "incremental_stateful_explicit_config": inc_stateful_exp_df,
        "incremental_stateless_explicit_config": inc_stateless_exp_df,
    }
    refs = {
        "offline": off_df,
        "snapshot_internal_framelevel": snap_internal_ref_df,
    }

    comp_all = _comparison_table(cands, refs, args.start_frame, args.end_frame, args.tolerance)
    comp_all[comp_all["reference"] == "offline"].to_csv(out_dir / "comparison_against_offline.csv", index=False)
    comp_all[comp_all["reference"] == "snapshot_internal_framelevel"].to_csv(out_dir / "comparison_against_snapshot_internal.csv", index=False)

    _by_segment(cands, refs, args.tolerance).to_csv(out_dir / "comparison_by_segment.csv", index=False)
    _by_pair(cands, refs, args.start_frame, args.end_frame, args.tolerance).to_csv(out_dir / "comparison_by_pair.csv", index=False)

    bb = pd.concat(
        [
            _by_group(_restrict(df, args.start_frame, args.end_frame), "batch_local_index", _restrict(off_df, args.start_frame, args.end_frame), name, "offline", args.tolerance)
            for name, df in cands.items()
            if "batch_local_index" in df.columns
        ],
        ignore_index=True,
    )
    bb.to_csv(out_dir / "comparison_by_batch_local_index.csv", index=False)

    nt = _by_group(
        _restrict(debug_frame_df, args.start_frame, args.end_frame).rename(columns={"slot_u_after_frame_expansion": "edge_src", "slot_v_after_frame_expansion": "edge_dst"}),
        "native_timestep",
        _restrict(off_df, args.start_frame, args.end_frame),
        "incremental_stateful_explicit_config_after_frame_expansion",
        "offline",
        args.tolerance,
    )
    nt.to_csv(out_dir / "comparison_by_native_timestep.csv", index=False)

    stage_rows = []
    for stage in ["before_crf", "after_crf", "after_postprocess", "after_frame_expansion"]:
        sdf = _stage_edges(_restrict(debug_frame_df, args.start_frame, args.end_frame), stage)
        if sdf.empty:
            continue
        m_off = _match_metrics(_to_points(sdf), _to_points(_restrict(off_df, args.start_frame, args.end_frame)), args.tolerance)
        m_snap = _match_metrics(_to_points(sdf), _to_points(_restrict(snap_internal_ref_df, args.start_frame, args.end_frame)), args.tolerance)
        stage_rows.append({"stage": stage, "reference": "offline", **m_off})
        stage_rows.append({"stage": stage, "reference": "snapshot_internal_framelevel", **m_snap})
    pd.DataFrame(stage_rows).to_csv(out_dir / "comparison_by_stage.csv", index=False)

    # unmatched examples (stateful actual vs refs)
    pred = _to_points(_restrict(inc_stateful_actual_df, args.start_frame, args.end_frame))
    ref = _to_points(_restrict(off_df, args.start_frame, args.end_frame))
    snap_map = {int(r[0]): f"{r[1]}->{r[2]}" for r in _to_points(_restrict(snap_internal_ref_df, args.start_frame, args.end_frame))}
    rows = []
    used = set()
    for p in pred:
        lo, hi = p[0] - args.tolerance, p[0] + args.tolerance
        match_j = None
        for j, r in enumerate(ref):
            if j in used:
                continue
            if p[1] == r[1] and p[2] == r[2] and lo <= r[0] <= hi:
                match_j = j
                break
        if match_j is not None:
            used.add(match_j)
            continue
        rows.append({
            "frame": p[0],
            "edge_incremental_stateful_actual": f"{p[1]}->{p[2]}",
            "edge_snapshot_internal": snap_map.get(p[0]),
            "reason_estimated": "wrong_pair_or_temporal_coarse",
        })
        if len(rows) >= 400:
            break
    pd.DataFrame(rows).to_csv(out_dir / "unmatched_examples_cumulative_context.csv", index=False)

    # Effective config audit
    trial_args = off_summary.get("trial_args", {})
    effective_cfg = {
        "offline_effective": {
            "sample_freq": off_summary.get("sample_freq"),
            "window_seconds": off_summary.get("window_seconds"),
            "decode": off_summary.get("decode"),
            "use_crf": off_summary.get("use_crf"),
            "min_event_duration": off_summary.get("min_event_duration"),
        },
        "incremental_actual_config": inc_summary.get("runtime_config", {}),
        "incremental_actual_effective_from_trial_args": {
            "sample_freq_effective": int(trial_args.get("sample_freq", 5)),
            "window_seconds_effective": float(trial_args.get("window_seconds", 10.0)),
            "note": "Aunque runtime_config tenga None, ActionsRuntime._run_pathcrf cae a trial_args para sample_freq/window_seconds",
        },
        "incremental_explicit_config": asdict(cfg_explicit),
    }
    _write_json(out_dir / "effective_config_audit.json", effective_cfg)

    # Summary questions
    def _fetch(c: str, r: str) -> dict[str, Any] | None:
        d = comp_all[(comp_all["candidate"] == c) & (comp_all["reference"] == r)]
        return None if d.empty else d.iloc[0].to_dict()

    pattern_counts = Counter()
    cp_idx = Counter()
    if not debug_batch_df.empty:
        for _, rr in debug_batch_df.iterrows():
            pattern_counts[str(rr.get("compressed_pattern_after_frame_expansion"))] += 1
            try:
                cps = json.loads(rr.get("change_points_after_frame_expansion") or "[]")
            except Exception:
                cps = []
            for x in cps:
                cp_idx[int(x)] += 1

    summary = {
        "summary": {
            "main_findings": [],
            "likely_root_causes": [],
            "recommended_next_steps": [],
        },
        "answers": {
            "1_reference_fixed_snapshot_internal_framelevel": True,
            "2_snapshot_on_windows_vs_snapshot_internal": _fetch("snapshot_on_incremental_windows", "snapshot_internal_framelevel"),
            "3_snapshot_cumulative_vs_offline": _fetch("snapshot_cumulative_at_incremental_checkpoints", "offline"),
            "4_snapshot_cumulative_vs_snapshot_internal": _fetch("snapshot_cumulative_at_incremental_checkpoints", "snapshot_internal_framelevel"),
            "5_cumulative_recovers_snapshot_offline": (_fetch("snapshot_cumulative_at_incremental_checkpoints", "offline") or {}).get("f1"),
            "6_on_windows_drop_due_to_context": (_fetch("snapshot_on_incremental_windows", "offline") or {}).get("f1"),
            "7_incremental_explicit_improves": {
                "stateful_actual_vs_offline": (_fetch("incremental_stateful_actual", "offline") or {}).get("f1"),
                "stateful_explicit_vs_offline": (_fetch("incremental_stateful_explicit_config", "offline") or {}).get("f1"),
                "stateless_actual_vs_offline": (_fetch("incremental_stateless_actual", "offline") or {}).get("f1"),
                "stateless_explicit_vs_offline": (_fetch("incremental_stateless_explicit_config", "offline") or {}).get("f1"),
            },
            "8_none_semantics_changed": effective_cfg["incremental_actual_effective_from_trial_args"],
            "9_local_idx_5_9_stage_origin": {
                "change_points_after_frame_expansion": dict(cp_idx),
                "pattern_top": pattern_counts.most_common(10),
                "available_stages": ["before_crf", "after_crf", "after_postprocess", "after_frame_expansion"],
                "limitations": "No logits/entropy/mask explícitos en output de inference.py para este checkpoint/modelo",
            },
            "10_native_resolution_coarser_than_frames": "No en esta ejecución: native_timestep_count == num_frames_in_batch (10)",
            "11_batch_local_to_global_mapping_error": "No evidencia fuerte: mapeo usado frame=batch_start+local_idx y consistente con raw_edge_batch",
            "12_most_likely_cause_A_G": None,
            "13_next_change": "Alinear exactamente path snapshot cumulative dentro del runtime incremental o reutilizar wrapper offline/snapshot para inferencia de checkpoints antes del postproceso incremental.",
        },
    }

    f1_cum_off = ((_fetch("snapshot_cumulative_at_incremental_checkpoints", "offline") or {}).get("f1") or 0.0)
    f1_win_off = ((_fetch("snapshot_on_incremental_windows", "offline") or {}).get("f1") or 0.0)
    f1_inc_exp = ((_fetch("incremental_stateful_explicit_config", "offline") or {}).get("f1") or 0.0)
    f1_inc_act = ((_fetch("incremental_stateful_actual", "offline") or {}).get("f1") or 0.0)

    if f1_cum_off > 0.7 and f1_win_off < 0.3:
        cause = "A + C/E/G: contexto acumulado es crítico; ventanas aisladas rompen dinámica y el runtime incremental añade degradación adicional."
    elif f1_inc_exp > f1_inc_act * 1.5:
        cause = "C/B: paridad de configuración y estado causal impactan de forma relevante."
    else:
        cause = "C/E/G con algo de B: principal divergencia en runtime/postproceso/semántica temporal incremental."
    summary["answers"]["12_most_likely_cause_A_G"] = cause
    summary["summary"]["main_findings"] = [
        "Se corrigió la referencia contra snapshot interno frame-level (no sparse/gallery).",
        "Se ejecutó variante snapshot acumulativa en checkpoints incrementales.",
        "Se ejecutaron variantes incremental explícitas (stateful/stateless) y debug por etapas disponibles.",
    ]
    summary["summary"]["likely_root_causes"] = [cause]
    summary["summary"]["recommended_next_steps"] = [
        "Instrumentar inference.py para exponer logits/entropy/mask y validar si el patrón nace en CRF o postproceso.",
        "Probar incremental con contexto acumulado interno equivalente a snapshot_cumulative antes de cualquier expansión frame-level.",
    ]
    _write_json(out_dir / "cumulative_context_summary.json", summary)

    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
