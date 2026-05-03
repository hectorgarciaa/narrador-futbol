"""
Auditoría de gaps y forced absorption en video_prueba (partido_tracks.json, 750 frames).

Exporta:
  - output/actions/pathcrf_compare/video_prueba_gap_absorption_debug/gap_summary.csv
  - output/actions/pathcrf_compare/video_prueba_gap_absorption_debug/gap_events.csv
  - output/actions/pathcrf_compare/video_prueba_gap_absorption_debug/forced_absorption_events.csv
  - output/actions/pathcrf_compare/video_prueba_gap_absorption_debug/final_report.md
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Rutas
# ---------------------------------------------------------------------------
TRACKS_PATH = Path("output/tracks_json/tracker/partido_tracks.json")
OUT_DIR = Path(
    "output/actions/pathcrf_compare/video_prueba_gap_absorption_debug"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAP_SUMMARY_CSV = OUT_DIR / "gap_summary.csv"
GAP_EVENTS_CSV = OUT_DIR / "gap_events.csv"
FA_EVENTS_CSV = OUT_DIR / "forced_absorption_events.csv"
REPORT_MD = OUT_DIR / "final_report.md"

# ---------------------------------------------------------------------------
# 1. Cargar tracking data
# ---------------------------------------------------------------------------
print("Cargando partido_tracks.json ...")
with open(TRACKS_PATH) as f:
    raw = json.load(f)

frames = raw["actions_incremental"]
N_FRAMES = len(frames)
print(f"  Total frames: {N_FRAMES}")

# ---------------------------------------------------------------------------
# 2. Construir timeline de observaciones por slot
# ---------------------------------------------------------------------------
# Cada frame tiene person_slot_assignments {track_id: slot_name}
# y referee_slot_assignments {track_id: slot_name}.
# Un slot está "observado" si aparece como valor en alguno de los dos.

total_frames = N_FRAMES

# Set de todos los slots vistos
all_slots: set[str] = set()
slot_observed_frames: dict[str, set[int]] = defaultdict(set)

for frame_idx, frame in enumerate(frames):
    for slot_name in frame.get("person_slot_assignments", {}).values():
        slot_observed_frames[slot_name].add(frame_idx)
        all_slots.add(slot_name)
    for slot_name in frame.get("referee_slot_assignments", {}).values():
        slot_observed_frames[slot_name].add(frame_idx)
        all_slots.add(slot_name)

all_slots = sorted(all_slots)
print(f"  Total slots: {len(all_slots)} -> {all_slots}")

# ---------------------------------------------------------------------------
# 3. Detectar gaps por slot
# ---------------------------------------------------------------------------
gap_events_rows = []
gap_summary_rows = []

for slot in all_slots:
    observed = sorted(slot_observed_frames[slot])
    if not observed:
        gap_summary_rows.append(
            {
                "slot": slot,
                "num_gaps": 0,
                "max_gap_frames": total_frames,
                "mean_gap_frames": float(total_frames),
                "median_gap_frames": float(total_frames),
                "p75_gap_frames": float(total_frames),
                "p95_gap_frames": float(total_frames),
                "total_missing_frames": total_frames,
                "total_observed_frames": 0,
                "first_observed_frame": None,
                "last_observed_frame": None,
                "gaps_le_4": 0,
                "gaps_le_8": 0,
                "gaps_le_12": 0,
                "gaps_le_16": 0,
                "gaps_le_24": 0,
                "gaps_gt_24": 0,
                "gaps_gt_50": 0,
                "gaps_gt_100": 0,
            }
        )
        continue

    # gaps = periodos sin observación
    # gap antes del primer frame observado
    gaps = []
    if observed[0] > 0:
        gaps.append((0, observed[0] - 1))

    for i in range(len(observed) - 1):
        gap_start = observed[i] + 1
        gap_end = observed[i + 1] - 1
        if gap_start <= gap_end:
            gaps.append((gap_start, gap_end))

    if observed[-1] < total_frames - 1:
        gaps.append((observed[-1] + 1, total_frames - 1))

    gap_lengths = [end - start + 1 for start, end in gaps]

    total_missing = sum(gap_lengths)
    total_observed = len(observed)

    # gap detail
    for (g_start, g_end), g_len in zip(gaps, gap_lengths):
        # find prev/next observed frames
        prev_obs = max((f for f in observed if f < g_start), default=None)
        next_obs = min((f for f in observed if f > g_end), default=None)
        # track ids (approximate: we track which raw_tracker_id was mapped to this slot)
        gap_events_rows.append(
            {
                "slot": slot,
                "gap_start_frame": g_start,
                "gap_end_frame": g_end,
                "gap_length_frames": g_len,
                "prev_observed_frame": prev_obs,
                "next_observed_frame": next_obs,
                "prev_track_id": None,
                "next_track_id": None,
                "can_be_interpolated_with_lag_12": (
                    next_obs is not None and (next_obs - g_end) <= 12
                ),
                "can_be_interpolated_with_lag_16": (
                    next_obs is not None and (next_obs - g_end) <= 16
                ),
                "can_be_interpolated_with_lag_24": (
                    next_obs is not None and (next_obs - g_end) <= 24
                ),
            }
        )

    # statistics
    if gap_lengths:
        max_gap = max(gap_lengths)
        mean_gap = sum(gap_lengths) / len(gap_lengths)
        sorted_lens = sorted(gap_lengths)
        n = len(sorted_lens)
        median_gap = sorted_lens[n // 2] if n % 2 == 1 else (
            sorted_lens[n // 2 - 1] + sorted_lens[n // 2]
        ) / 2.0
        p75 = sorted_lens[int(n * 0.75)] if int(n * 0.75) < n else sorted_lens[-1]
        p95 = sorted_lens[int(n * 0.95)] if int(n * 0.95) < n else sorted_lens[-1]
    else:
        max_gap = 0
        mean_gap = 0.0
        median_gap = 0.0
        p75 = 0
        p95 = 0

    gap_summary_rows.append(
        {
            "slot": slot,
            "num_gaps": len(gap_lengths),
            "max_gap_frames": max_gap,
            "mean_gap_frames": round(mean_gap, 2),
            "median_gap_frames": round(median_gap, 2),
            "p75_gap_frames": p75,
            "p95_gap_frames": p95,
            "total_missing_frames": total_missing,
            "total_observed_frames": total_observed,
            "first_observed_frame": int(observed[0]),
            "last_observed_frame": int(observed[-1]),
            "gaps_le_4": sum(1 for g in gap_lengths if g <= 4),
            "gaps_le_8": sum(1 for g in gap_lengths if g <= 8),
            "gaps_le_12": sum(1 for g in gap_lengths if g <= 12),
            "gaps_le_16": sum(1 for g in gap_lengths if g <= 16),
            "gaps_le_24": sum(1 for g in gap_lengths if g <= 24),
            "gaps_gt_24": sum(1 for g in gap_lengths if g > 24),
            "gaps_gt_50": sum(1 for g in gap_lengths if g > 50),
            "gaps_gt_100": sum(1 for g in gap_lengths if g > 100),
        }
    )
    print(f"  {slot}: max_gap={max_gap}, num_gaps={len(gap_lengths)}, missing={total_missing}/{total_frames}")

# ---------------------------------------------------------------------------
# 4. Export gap_summary.csv
# ---------------------------------------------------------------------------
df_gap_summary = pd.DataFrame(gap_summary_rows)
df_gap_summary.to_csv(GAP_SUMMARY_CSV, index=False)
print(f"\nGap summary saved: {GAP_SUMMARY_CSV} ({len(df_gap_summary)} slots)")

# ---------------------------------------------------------------------------
# 5. Export gap_events.csv
# ---------------------------------------------------------------------------
df_gap_events = pd.DataFrame(gap_events_rows)
df_gap_events.to_csv(GAP_EVENTS_CSV, index=False)
print(f"Gap events saved: {GAP_EVENTS_CSV} ({len(df_gap_events)} events)")

# ---------------------------------------------------------------------------
# 6. Extraer forced absorption events desde los tracks canónicos
# ---------------------------------------------------------------------------
# Los eventos de forced_absorption están dentro del estado de canonical tracks.
# En `partido_tracks.json` las detecciones canónicas están embebidas en la
# estructura de `actions_incremental`. Buscamos dentro de los payloads de cada
# frame las entradas con forced_absorption=true.
# La estructura exacta varía, pero sabemos que hay exactamente 2 eventos.
fa_rows = []

for frame_idx, frame in enumerate(frames):
    # Explorar todas las keys del frame en búsqueda de forced_absorption
    # Puede estar dentro de person_slot_assignments, canonical state data, etc.
    # Hacemos una búsqueda recursiva
    def _search(obj, path=""):
        results = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "forced_absorption" and v is True:
                    results.append((path, obj))
                results.extend(_search(v, f"{path}.{k}"))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                results.extend(_search(v, f"{path}[{i}]"))
        return results

    found = _search(frame)
    for path, payload in found:
        # Extract canonical_id from path or payload
        canonical_id = payload.get("player_id") or payload.get("canonical_id")
        # The canonical_id may be in the key name if it's under a dict
        path_parts = path.split(".")
        for part in reversed(path_parts):
            p = part.replace("[", ".").replace("]", "").split(".")[-1]
            try:
                cid = int(p)
                if 1 <= cid <= 25:
                    canonical_id = cid
                    break
            except (ValueError, TypeError):
                continue

        fa_rows.append(
            {
                "frame": frame_idx,
                "canonical_id": canonical_id,
                "old_track_id": payload.get("forced_absorption_source_raw_tracker_id"),
                "new_track_id": payload.get("source_raw_tracker_id"),
                "raw_tracker_streak_frames": payload.get(
                    "forced_absorption_raw_tracker_streak_frames"
                ),
                "canonical_lost_frames": payload.get(
                    "forced_absorption_canonical_lost_frames"
                ),
                "distance_sq": payload.get("forced_absorption_distance_sq"),
                "reference_frame": payload.get("forced_absorption_reference_frame"),
                "mode": payload.get("forced_absorption_mode"),
                "team": payload.get("team"),
                "confidence": payload.get("confidence"),
                "class_name": payload.get("class_tracker"),
                "bbox": payload.get("bbox"),
                "field_position_m": payload.get("field_position_m"),
            }
        )

# Si no encontramos eventos en los frames, intentar búsqueda más agresiva
if len(fa_rows) < 2:
    print("  Búsqueda detallada de forced_absorption en el JSON ...")
    # Recorrer todo el JSON sin restricción de claves
    def deep_search_forced_absorption(obj, path=""):
        results = []
        if isinstance(obj, dict):
            if obj.get("forced_absorption") is True:
                results.append((path, obj))
            for k, v in obj.items():
                results.extend(deep_search_forced_absorption(v, f"{path}.{k}"))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                results.extend(deep_search_forced_absorption(v, f"{path}[{i}]"))
        return results

    all_fa = deep_search_forced_absorption(raw)
    print(f"  Encontrados {len(all_fa)} objetos con forced_absorption=true")
    for path, payload in all_fa:
        canonical_id = None
        path_parts = path.split(".")
        for part in reversed(path_parts):
            p = part.replace("[", ".").replace("]", "").split(".")[-1]
            try:
                cid = int(p)
                if 1 <= cid <= 25:
                    canonical_id = cid
                    break
            except (ValueError, TypeError):
                continue
        fa_rows.append(
            {
                "frame": payload.get("frame"),
                "canonical_id": canonical_id,
                "old_track_id": payload.get("forced_absorption_source_raw_tracker_id"),
                "new_track_id": payload.get("source_raw_tracker_id"),
                "raw_tracker_streak_frames": payload.get(
                    "forced_absorption_raw_tracker_streak_frames"
                ),
                "canonical_lost_frames": payload.get(
                    "forced_absorption_canonical_lost_frames"
                ),
                "distance_sq": payload.get("forced_absorption_distance_sq"),
                "reference_frame": payload.get("forced_absorption_reference_frame"),
                "mode": payload.get("forced_absorption_mode"),
                "team": payload.get("team"),
                "confidence": payload.get("confidence"),
                "class_name": payload.get("class_tracker"),
                "bbox": payload.get("bbox"),
                "field_position_m": payload.get("field_position_m"),
            }
        )

# ---------------------------------------------------------------------------
# 7. Export forced_absorption_events.csv
# ---------------------------------------------------------------------------
df_fa = pd.DataFrame(fa_rows)
df_fa.to_csv(FA_EVENTS_CSV, index=False)
print(f"Forced absorption events saved: {FA_EVENTS_CSV} ({len(df_fa)} events)")

# ---------------------------------------------------------------------------
# 8. Análisis global y respuestas
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANÁLISIS GLOBAL DE GAPS")
print("=" * 70)

all_gap_lengths = [r["gap_length_frames"] for r in gap_events_rows]
all_slot_names = [r["slot"] for r in gap_summary_rows]

max_gap_global = max(r["max_gap_frames"] for r in gap_summary_rows)
max_gap_slot = max(
    (r["slot"], r["max_gap_frames"])
    for r in gap_summary_rows
    if r["total_observed_frames"] > 0
)
print(f"Max gap global: {max_gap_global} frames (slot: {max_gap_slot})")

total_missing_across_all = sum(r["total_missing_frames"] for r in gap_summary_rows)
total_frames_all_slots = total_frames * len(all_slots)
total_missing_in_gaps_gt_12 = sum(
    r["total_missing_frames"] - r["total_observed_frames"]
    for r in gap_summary_rows
)
print(f"Total missing frames across all slots: {total_missing_across_all}/{total_frames_all_slots}")

# Gaps > 12
all_gaps_gt_12 = [g for g in all_gap_lengths if g > 12]
pct_gaps_gt_12 = len(all_gaps_gt_12) / max(len(all_gap_lengths), 1) * 100
print(f"Gaps > 12 frames: {len(all_gaps_gt_12)}/{len(all_gap_lengths)} ({pct_gaps_gt_12:.1f}%)")

# Missing frames in gaps > 12
missing_in_gt12 = sum(g for g in all_gap_lengths if g > 12)
total_missing = sum(all_gap_lengths)
pct_missing_gt12 = missing_in_gt12 / max(total_missing, 1) * 100
print(f"Missing frames in gaps>12: {missing_in_gt12}/{total_missing} ({pct_missing_gt12:.1f}%)")

# Percentiles de gaps
sorted_gaps = sorted(all_gap_lengths)
if sorted_gaps:
    print(f"Percentiles de gaps (todos los slots):")
    for pct in [50, 75, 90, 95, 99, 100]:
        idx = int(len(sorted_gaps) * pct / 100) - 1
        idx = max(0, min(idx, len(sorted_gaps) - 1))
        val = sorted_gaps[idx]
        seconds = val / 25.0
        print(f"  p{pct:2d} = {val:4d} frames  ({seconds:.2f}s at 25fps)")

# Slot max gaps
print(f"\nTop 10 slots por max gap:")
slot_max = sorted(
    [(r["slot"], r["max_gap_frames"]) for r in gap_summary_rows],
    key=lambda x: -x[1],
)
for slot, mg in slot_max[:10]:
    print(f"  {slot}: max_gap={mg} frames ({mg/25:.2f}s)")

# ---------------------------------------------------------------------------
# 9. Análisis de forced_absorption
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FORCED ABSORPTION EN VIDEO_PRUEBA")
print("=" * 70)
for _, ev in df_fa.iterrows():
    print(f"  Frame {ev['frame']}:")
    print(f"    canonical_id={ev['canonical_id']}")
    print(f"    old_track_id={ev['old_track_id']} -> new_track_id={ev['new_track_id']}")
    print(f"    canonical_lost_frames={ev['canonical_lost_frames']}")
    print(f"    raw_tracker_streak_frames={ev['raw_tracker_streak_frames']}")
    print(f"    distance_sq={ev['distance_sq']}")
    print(f"    mode={ev['mode']}")
    print(f"    team={ev['team']}")
    print(f"    field_position={ev['field_position_m']}")

# ---------------------------------------------------------------------------
# 10. Análisis de slots problemáticos
# ---------------------------------------------------------------------------
problematic_slots = [
    "home_1", "home_2", "home_5", "home_10",
    "away_1", "away_7", "referee_2", "referee_3",
]
print(f"\n{'=' * 70}")
print("SLOTS PROBLEMÁTICOS")
print('=' * 70)
for slot in problematic_slots:
    row = next((r for r in gap_summary_rows if r["slot"] == slot), None)
    if row is None:
        print(f"  {slot}: NO DATA")
        continue
    fa_for_slot = df_fa[df_fa["canonical_id"].apply(
        lambda cid: _resolve_canonical_to_slot(cid) == slot
        if cid is not None else False
    )] if "canonical_id" in df_fa.columns else pd.DataFrame()

    # Get canonical IDs for this slot
    from football_ai.pathcrf_slot_mapping import person_slot_to_canonical_id, referee_slot_to_canonical_id
    cid = person_slot_to_canonical_id(slot) or referee_slot_to_canonical_id(slot)

    print(f"  {slot} (canonical_id={cid}):")
    print(f"    max_gap={row['max_gap_frames']} frames ({row['max_gap_frames']/25:.2f}s)")
    print(f"    missing={row['total_missing_frames']}/{total_frames}")
    print(f"    gaps_gt_12={row['gaps_gt_24']} (plus 24)")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _resolve_canonical_to_slot(cid):
    try:
        from football_ai.pathcrf_slot_mapping import (
            canonical_id_to_person_slot,
            canonical_id_to_referee_slot,
        )
        return (
            canonical_id_to_person_slot(cid)
            or canonical_id_to_referee_slot(cid)
        )
    except ImportError:
        return f"cid_{cid}"

print("\n¡Auditoría completada!")
