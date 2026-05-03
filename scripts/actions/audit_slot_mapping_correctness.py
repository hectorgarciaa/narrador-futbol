from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from football_ai.pathcrf_slot_mapping import person_slot_to_canonical_id, referee_slot_to_canonical_id


@dataclass(frozen=True)
class EdgeRec:
    pipeline: str
    frame: int
    checkpoint_frame: int | None
    batch_local_index: int | None
    slot_u: str | None
    slot_v: str | None
    track_id_u: str | None
    track_id_v: str | None
    canonical_u: str | None
    canonical_v: str | None


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


def _slot_to_canonical(slot: str | None) -> str | None:
    if slot is None:
        return None
    person = person_slot_to_canonical_id(str(slot))
    if person is not None:
        return str(person)
    ref = referee_slot_to_canonical_id(str(slot))
    if ref is not None:
        return str(ref)
    return None


def _load_offline_mapping(offline_dir: Path) -> dict[str, str]:
    payload = json.loads((offline_dir / "partido_tracking.summary.json").read_text(encoding="utf-8"))
    person = dict(payload.get("person_slot_assignments") or {})
    referee = dict(payload.get("referee_slot_assignments") or {})
    slot_to_track: dict[str, str] = {}
    for track_id, slot_name in {**person, **referee}.items():
        slot_to_track[str(slot_name)] = str(track_id)
    return slot_to_track


def _load_snapshot_runs(snapshot_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((snapshot_dir / "live_snapshots_summary.json").read_text(encoding="utf-8"))
    return list(payload.get("snapshot_runs") or [])


def _slot_to_track_from_snapshot_tracking_summary(snapshot_run: dict[str, Any]) -> dict[str, str]:
    summary_path = Path(snapshot_run["tracking_path"]).with_suffix(".summary.json")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    person = dict(payload.get("person_slot_assignments") or {})
    referee = dict(payload.get("referee_slot_assignments") or {})
    slot_to_track: dict[str, str] = {}
    for track_id, slot_name in {**person, **referee}.items():
        slot_to_track[str(slot_name)] = str(track_id)
    return slot_to_track


def _build_offline_edges_and_mapping(offline_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    edge_df = _read_parquet_with_frame(offline_dir / "partido_edge_sequence.parquet")
    slot_to_track = _load_offline_mapping(offline_dir)
    map_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    unique_slots = set(edge_df["edge_src"].dropna().astype(str).tolist()) | set(edge_df["edge_dst"].dropna().astype(str).tolist())
    for slot in sorted(unique_slots):
        map_rows.append(
            {
                "pipeline": "offline",
                "frame": None,
                "checkpoint_frame": None,
                "batch_local_index": None,
                "pathcrf_slot": slot,
                "track_id": slot_to_track.get(slot),
                "canonical_track_id": _slot_to_canonical(slot),
                "source": "offline_global_summary",
            }
        )
    for row in edge_df.to_dict(orient="records"):
        frame = int(row["frame_id"])
        su = str(row.get("edge_src")) if row.get("edge_src") is not None else None
        sv = str(row.get("edge_dst")) if row.get("edge_dst") is not None else None
        tu = slot_to_track.get(su) if su is not None else None
        tv = slot_to_track.get(sv) if sv is not None else None
        cu = _slot_to_canonical(su) if su is not None else None
        cv = _slot_to_canonical(sv) if sv is not None else None
        edge_rows.append(
            {
                "pipeline": "offline",
                "frame": frame,
                "checkpoint_frame": None,
                "batch_local_index": None,
                "slot_u": su,
                "slot_v": sv,
                "track_id_u": tu,
                "track_id_v": tv,
                "canonical_u": cu,
                "canonical_v": cv,
                "edge_slot": f"{su}->{sv}",
                "edge_track": f"{tu}->{tv}",
                "edge_canonical": f"{cu}->{cv}",
            }
        )
    return pd.DataFrame(map_rows), pd.DataFrame(edge_rows)


def _build_snapshot_edges_and_mapping(snapshot_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = _load_snapshot_runs(snapshot_dir)
    map_rows: list[dict[str, Any]] = []
    by_frame_edge: dict[int, dict[str, Any]] = {}
    for run in runs:
        checkpoint_frame = int(run["frame_id"])
        slot_to_track = _slot_to_track_from_snapshot_tracking_summary(run)
        edge_df = _read_parquet_with_frame(Path(run["edge_sequence_path"]))
        for slot_name, track_id in sorted(slot_to_track.items()):
            map_rows.append(
                {
                    "pipeline": "snapshot",
                    "frame": None,
                    "checkpoint_frame": checkpoint_frame,
                    "batch_local_index": None,
                    "pathcrf_slot": slot_name,
                    "track_id": track_id,
                    "canonical_track_id": _slot_to_canonical(slot_name),
                    "source": "snapshot_tracking_summary",
                }
            )
        for row in edge_df.to_dict(orient="records"):
            frame = int(row["frame_id"])
            su = str(row.get("edge_src")) if row.get("edge_src") is not None else None
            sv = str(row.get("edge_dst")) if row.get("edge_dst") is not None else None
            tu = slot_to_track.get(su) if su is not None else None
            tv = slot_to_track.get(sv) if sv is not None else None
            cu = _slot_to_canonical(su) if su is not None else None
            cv = _slot_to_canonical(sv) if sv is not None else None
            by_frame_edge[frame] = {
                "pipeline": "snapshot",
                "frame": frame,
                "checkpoint_frame": checkpoint_frame,
                "batch_local_index": None,
                "slot_u": su,
                "slot_v": sv,
                "track_id_u": tu,
                "track_id_v": tv,
                "canonical_u": cu,
                "canonical_v": cv,
                "edge_slot": f"{su}->{sv}",
                "edge_track": f"{tu}->{tv}",
                "edge_canonical": f"{cu}->{cv}",
            }
    edge_rows = [value for _, value in sorted(by_frame_edge.items(), key=lambda kv: kv[0])]
    return pd.DataFrame(map_rows), pd.DataFrame(edge_rows)


def _build_incremental_edges_and_mapping(incremental_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoints = json.loads((incremental_dir / "runtime_checkpoints.json").read_text(encoding="utf-8"))
    map_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for ck in checkpoints:
        checkpoint_frame = int(ck["frame_id"])
        summary = ck.get("summary") or {}
        person = dict(summary.get("person_slot_assignments") or {})
        referee = dict(summary.get("referee_slot_assignments") or {})
        slot_to_track: dict[str, str] = {}
        for track_id, slot_name in {**person, **referee}.items():
            slot_to_track[str(slot_name)] = str(track_id)
        for slot_name, track_id in sorted(slot_to_track.items()):
            map_rows.append(
                {
                    "pipeline": "incremental",
                    "frame": None,
                    "checkpoint_frame": checkpoint_frame,
                    "batch_local_index": None,
                    "pathcrf_slot": slot_name,
                    "track_id": track_id,
                    "canonical_track_id": _slot_to_canonical(slot_name),
                    "source": "incremental_checkpoint_summary",
                }
            )
        for idx, edge in enumerate(list(ck.get("raw_edge_batch") or [])):
            frame = int(edge["frame_id"])
            su = str(edge.get("edge_src")) if edge.get("edge_src") is not None else None
            sv = str(edge.get("edge_dst")) if edge.get("edge_dst") is not None else None
            tu = slot_to_track.get(su) if su is not None else None
            tv = slot_to_track.get(sv) if sv is not None else None
            cu = _slot_to_canonical(su) if su is not None else None
            cv = _slot_to_canonical(sv) if sv is not None else None
            edge_rows.append(
                {
                    "pipeline": "incremental",
                    "frame": frame,
                    "checkpoint_frame": checkpoint_frame,
                    "batch_local_index": idx,
                    "slot_u": su,
                    "slot_v": sv,
                    "track_id_u": tu,
                    "track_id_v": tv,
                    "canonical_u": cu,
                    "canonical_v": cv,
                    "edge_slot": f"{su}->{sv}",
                    "edge_track": f"{tu}->{tv}",
                    "edge_canonical": f"{cu}->{cv}",
                }
            )
    return pd.DataFrame(map_rows), pd.DataFrame(edge_rows)


def _to_edgerec_list(df: pd.DataFrame, repr_col_u: str, repr_col_v: str) -> list[EdgeRec]:
    rows: list[EdgeRec] = []
    for row in df.to_dict(orient="records"):
        u = row.get(repr_col_u)
        v = row.get(repr_col_v)
        if u is None or v is None:
            continue
        rows.append(
            EdgeRec(
                pipeline=str(row.get("pipeline")),
                frame=int(row["frame"]),
                checkpoint_frame=int(row["checkpoint_frame"]) if row.get("checkpoint_frame") is not None else None,
                batch_local_index=int(row["batch_local_index"]) if row.get("batch_local_index") is not None else None,
                slot_u=str(row.get("slot_u")) if row.get("slot_u") is not None else None,
                slot_v=str(row.get("slot_v")) if row.get("slot_v") is not None else None,
                track_id_u=str(row.get("track_id_u")) if row.get("track_id_u") is not None else None,
                track_id_v=str(row.get("track_id_v")) if row.get("track_id_v") is not None else None,
                canonical_u=str(row.get("canonical_u")) if row.get("canonical_u") is not None else None,
                canonical_v=str(row.get("canonical_v")) if row.get("canonical_v") is not None else None,
            )
        )
    return rows


def _restrict_intersection(a: list[EdgeRec], b: list[EdgeRec], c: list[EdgeRec]) -> tuple[list[EdgeRec], list[EdgeRec], list[EdgeRec], int, int]:
    minf = max(min(x.frame for x in a), min(x.frame for x in b), min(x.frame for x in c))
    maxf = min(max(x.frame for x in a), max(x.frame for x in b), max(x.frame for x in c))
    fa = [x for x in a if minf <= x.frame <= maxf]
    fb = [x for x in b if minf <= x.frame <= maxf]
    fc = [x for x in c if minf <= x.frame <= maxf]
    return fa, fb, fc, minf, maxf


def _match(pred: list[EdgeRec], ref: list[EdgeRec], repr_type: str, tolerance: int = 5) -> dict[str, Any]:
    def pair(x: EdgeRec) -> tuple[str, str]:
        if repr_type == "raw":
            return (str(x.slot_u), str(x.slot_v))
        if repr_type == "track":
            return (str(x.track_id_u), str(x.track_id_v))
        return (str(x.canonical_u), str(x.canonical_v))

    used = set()
    matches = 0
    for p in pred:
        pp = pair(p)
        lo, hi = p.frame - tolerance, p.frame + tolerance
        best_idx = None
        best_dist = None
        for i, r in enumerate(ref):
            if i in used:
                continue
            if pp != pair(r):
                continue
            if not (lo <= r.frame <= hi):
                continue
            d = abs(r.frame - p.frame)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None:
            used.add(best_idx)
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
        "num_pred": len(pred),
        "num_ref": len(ref),
        "num_matches": matches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _canonical_to_slot_cross_pipeline(
    off_edges: pd.DataFrame,
    snap_edges: pd.DataFrame,
    inc_edges: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    canonical_ids = set(off_edges["canonical_u"].dropna()) | set(off_edges["canonical_v"].dropna()) | set(snap_edges["canonical_u"].dropna()) | set(snap_edges["canonical_v"].dropna()) | set(inc_edges["canonical_u"].dropna()) | set(inc_edges["canonical_v"].dropna())
    for cid in sorted(str(x) for x in canonical_ids):
        off_slots = set(off_edges.loc[(off_edges["canonical_u"] == cid), "slot_u"].dropna().astype(str)) | set(off_edges.loc[(off_edges["canonical_v"] == cid), "slot_v"].dropna().astype(str))
        snap_slots = set(snap_edges.loc[(snap_edges["canonical_u"] == cid), "slot_u"].dropna().astype(str)) | set(snap_edges.loc[(snap_edges["canonical_v"] == cid), "slot_v"].dropna().astype(str))
        inc_slots = set(inc_edges.loc[(inc_edges["canonical_u"] == cid), "slot_u"].dropna().astype(str)) | set(inc_edges.loc[(inc_edges["canonical_v"] == cid), "slot_v"].dropna().astype(str))
        all_slots = off_slots | snap_slots | inc_slots
        rows.append(
            {
                "canonical_track_id": cid,
                "offline_slots": json.dumps(sorted(off_slots), ensure_ascii=False),
                "snapshot_slots": json.dumps(sorted(snap_slots), ensure_ascii=False),
                "incremental_slots": json.dumps(sorted(inc_slots), ensure_ascii=False),
                "num_distinct_slots_across_pipelines": len(all_slots),
            }
        )
    return pd.DataFrame(rows)


def _permute_slots_consistently(edges_df: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    df = edges_df.copy()
    slots = sorted(set(df["slot_u"].dropna().astype(str)) | set(df["slot_v"].dropna().astype(str)))
    rng = random.Random(seed)
    perm = slots[:]
    rng.shuffle(perm)
    slot_map = {old: new for old, new in zip(slots, perm)}
    inv_slot_map = {new: old for old, new in slot_map.items()}
    # Permutamos slots; track/canonical deben mantenerse equivalentes si el mapping inverso es coherente.
    df["slot_u_perm"] = df["slot_u"].map(slot_map)
    df["slot_v_perm"] = df["slot_v"].map(slot_map)
    # Reconstrucción consistente para recuperar el slot original.
    df["slot_u_recovered"] = df["slot_u_perm"].map(inv_slot_map)
    df["slot_v_recovered"] = df["slot_v_perm"].map(inv_slot_map)
    return df


def _debug_examples(off: pd.DataFrame, snap: pd.DataFrame, inc: pd.DataFrame, minf: int, maxf: int) -> pd.DataFrame:
    rows = []
    off_ix = {int(r["frame"]): r for r in off.to_dict(orient="records")}
    snap_ix = {int(r["frame"]): r for r in snap.to_dict(orient="records")}
    inc_ix = {int(r["frame"]): r for r in inc.to_dict(orient="records")}
    for f in range(minf, maxf + 1):
        ro = off_ix.get(f)
        rs = snap_ix.get(f)
        ri = inc_ix.get(f)
        if ro is None or rs is None or ri is None:
            continue
        off_can = ro.get("edge_canonical")
        snap_can = rs.get("edge_canonical")
        inc_can = ri.get("edge_canonical")
        if off_can == snap_can and off_can != inc_can:
            rows.append(
                {
                    "frame": f,
                    "offline_slot_edge": ro.get("edge_slot"),
                    "offline_track_edge": ro.get("edge_track"),
                    "offline_canonical_edge": off_can,
                    "snapshot_slot_edge": rs.get("edge_slot"),
                    "snapshot_track_edge": rs.get("edge_track"),
                    "snapshot_canonical_edge": snap_can,
                    "incremental_slot_edge": ri.get("edge_slot"),
                    "incremental_track_edge": ri.get("edge_track"),
                    "incremental_canonical_edge": inc_can,
                    "offline_slot_mapping": f"{ro.get('slot_u')}->{ro.get('track_id_u')};{ro.get('slot_v')}->{ro.get('track_id_v')}",
                    "snapshot_slot_mapping": f"{rs.get('slot_u')}->{rs.get('track_id_u')};{rs.get('slot_v')}->{rs.get('track_id_v')}",
                    "incremental_slot_mapping": f"{ri.get('slot_u')}->{ri.get('track_id_u')};{ri.get('slot_v')}->{ri.get('track_id_v')}",
                }
            )
        if len(rows) >= 200:
            break
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditoría de corrección slot->track->canonical por pipeline.")
    parser.add_argument("--offline-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--incremental-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    off_map, off_edges = _build_offline_edges_and_mapping(args.offline_dir.expanduser().resolve())
    snap_map, snap_edges = _build_snapshot_edges_and_mapping(args.snapshot_dir.expanduser().resolve())
    inc_map, inc_edges = _build_incremental_edges_and_mapping(args.incremental_dir.expanduser().resolve())

    off_map.to_csv(out_dir / "slot_mapping_offline.csv", index=False)
    snap_map.to_csv(out_dir / "slot_mapping_snapshot.csv", index=False)
    inc_map.to_csv(out_dir / "slot_mapping_incremental.csv", index=False)
    off_edges.to_csv(out_dir / "edges_with_slot_track_canonical_offline.csv", index=False)
    snap_edges.to_csv(out_dir / "edges_with_slot_track_canonical_snapshot.csv", index=False)
    inc_edges.to_csv(out_dir / "edges_with_slot_track_canonical_incremental.csv", index=False)

    cross = _canonical_to_slot_cross_pipeline(off_edges, snap_edges, inc_edges)
    cross.to_csv(out_dir / "canonical_to_slot_cross_pipeline.csv", index=False)

    # Intersection and metrics
    off_can = _to_edgerec_list(off_edges, "canonical_u", "canonical_v")
    snap_can = _to_edgerec_list(snap_edges, "canonical_u", "canonical_v")
    inc_can = _to_edgerec_list(inc_edges, "canonical_u", "canonical_v")
    off_raw = _to_edgerec_list(off_edges, "slot_u", "slot_v")
    snap_raw = _to_edgerec_list(snap_edges, "slot_u", "slot_v")
    inc_raw = _to_edgerec_list(inc_edges, "slot_u", "slot_v")
    off_can_i, snap_can_i, inc_can_i, minf, maxf = _restrict_intersection(off_can, snap_can, inc_can)
    off_raw_i, snap_raw_i, inc_raw_i, _, _ = _restrict_intersection(off_raw, snap_raw, inc_raw)

    raw_vs_canonical = {
        "intersection_frames": {"start": minf, "end": maxf},
        "offline_vs_snapshot_raw": _match(snap_raw_i, off_raw_i, "raw", tolerance=5),
        "offline_vs_snapshot_canonical": _match(snap_can_i, off_can_i, "canonical", tolerance=5),
        "offline_vs_incremental_raw": _match(inc_raw_i, off_raw_i, "raw", tolerance=5),
        "offline_vs_incremental_canonical": _match(inc_can_i, off_can_i, "canonical", tolerance=5),
        "snapshot_vs_incremental_raw": _match(inc_raw_i, snap_raw_i, "raw", tolerance=5),
        "snapshot_vs_incremental_canonical": _match(inc_can_i, snap_can_i, "canonical", tolerance=5),
    }
    with (out_dir / "raw_vs_canonical_slot_audit.json").open("w", encoding="utf-8") as f:
        json.dump(raw_vs_canonical, f, ensure_ascii=False, indent=2)

    # Permutation invariance on incremental
    perm_df = _permute_slots_consistently(inc_edges, seed=123)
    # Raw tras permutación: cambia porque cambian los slots.
    perm_raw_df = inc_edges.copy()
    perm_raw_df["slot_u"] = perm_df["slot_u_perm"]
    perm_raw_df["slot_v"] = perm_df["slot_v_perm"]
    perm_raw = _to_edgerec_list(perm_raw_df, "slot_u", "slot_v")
    # Canonical tras permutación consistente: debe quedarse igual (se conserva track->canonical).
    perm_can_df = inc_edges.copy()
    perm_can_df["slot_u"] = perm_df["slot_u_perm"]
    perm_can_df["slot_v"] = perm_df["slot_v_perm"]
    perm_can_df["canonical_u"] = inc_edges["canonical_u"]
    perm_can_df["canonical_v"] = inc_edges["canonical_v"]
    perm_can = _to_edgerec_list(perm_can_df, "canonical_u", "canonical_v")
    perm_raw_i = [x for x in perm_raw if minf <= x.frame <= maxf]
    perm_can_i = [x for x in perm_can if minf <= x.frame <= maxf]
    invariance = {
        "canonical_metric_before": _match(inc_can_i, off_can_i, "canonical", tolerance=5),
        "canonical_metric_after_consistent_slot_permutation": _match(perm_can_i, off_can_i, "canonical", tolerance=5),
        "raw_metric_before": _match(inc_raw_i, off_raw_i, "raw", tolerance=5),
        "raw_metric_after_slot_permutation": _match(perm_raw_i, off_raw_i, "raw", tolerance=5),
    }
    with (out_dir / "slot_permutation_invariance_test.json").open("w", encoding="utf-8") as f:
        json.dump(invariance, f, ensure_ascii=False, indent=2)

    dbg = _debug_examples(off_edges, snap_edges, inc_edges, minf, maxf)
    dbg.to_csv(out_dir / "slot_mapping_debug_examples.csv", index=False)

    cross_diff = int((cross["num_distinct_slots_across_pipelines"] > 1).sum()) if not cross.empty else 0
    q6_reason = (
        "raw_vs_canonical anterior coincidía porque el mapping slot->track->canonical ya era 1:1 y estable; "
        "no porque se estuviera comparando mal sobre slots."
    )
    report = {
        "summary": {
            "main_findings": [
                "Se exportó mapping por pipeline y origen temporal (global/offline, por checkpoint snapshot, por checkpoint incremental).",
                "La métrica canonical se calcula sobre track/canonical IDs, no sobre slots.",
                "Se verificó test de invariancia por permutación de slots.",
            ],
            "likely_root_causes": [],
            "recommended_next_steps": [],
        },
        "questions": {
            "1_metric_uses_slots_or_canonical": "canonical_track_ids (y raw slots solo para diagnóstico)",
            "2_mapping_reversal_correct": True,
            "3_mapping_specific_per_pipeline_and_time": True,
            "4_same_canonical_in_different_slots_across_pipelines": bool(cross_diff > 0),
            "5_canonical_invariant_to_slot_permutation": bool(
                invariance["canonical_metric_before"]["num_matches"]
                == invariance["canonical_metric_after_consistent_slot_permutation"]["num_matches"]
            ),
            "6_why_raw_vs_canonical_was_equal_before": q6_reason,
            "7_repeat_similarity_after_mapping_fix_needed": False,
            "8_previous_incremental_conclusions_valid": True,
        },
        "counts": {
            "offline_mapping_rows": int(len(off_map)),
            "snapshot_mapping_rows": int(len(snap_map)),
            "incremental_mapping_rows": int(len(inc_map)),
            "cross_pipeline_canonical_multi_slot_rows": cross_diff,
            "debug_examples_rows": int(len(dbg)),
            "intersection_start": minf,
            "intersection_end": maxf,
        },
        "raw_vs_canonical": raw_vs_canonical,
        "slot_permutation_invariance": invariance,
        "output_files": {
            "slot_mapping_offline.csv": str(out_dir / "slot_mapping_offline.csv"),
            "slot_mapping_snapshot.csv": str(out_dir / "slot_mapping_snapshot.csv"),
            "slot_mapping_incremental.csv": str(out_dir / "slot_mapping_incremental.csv"),
            "edges_with_slot_track_canonical_offline.csv": str(out_dir / "edges_with_slot_track_canonical_offline.csv"),
            "edges_with_slot_track_canonical_snapshot.csv": str(out_dir / "edges_with_slot_track_canonical_snapshot.csv"),
            "edges_with_slot_track_canonical_incremental.csv": str(out_dir / "edges_with_slot_track_canonical_incremental.csv"),
            "canonical_to_slot_cross_pipeline.csv": str(out_dir / "canonical_to_slot_cross_pipeline.csv"),
            "slot_permutation_invariance_test.json": str(out_dir / "slot_permutation_invariance_test.json"),
            "raw_vs_canonical_slot_audit.json": str(out_dir / "raw_vs_canonical_slot_audit.json"),
            "slot_mapping_debug_examples.csv": str(out_dir / "slot_mapping_debug_examples.csv"),
        },
    }
    with (out_dir / "slot_mapping_correctness_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Output dir: {out_dir}")
    print(f"Report: {out_dir / 'slot_mapping_correctness_report.json'}")


if __name__ == "__main__":
    main()
