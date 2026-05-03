from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from football_ai.pathcrf_slot_mapping import (
    all_person_slots,
    all_referee_slots,
    canonical_id_to_person_slot,
    canonical_id_to_referee_slot,
)

TRACK_CLASSES = ("player", "goalkeeper", "referee", "ball")


@dataclass
class IncrementalBuilderConfig:
    fps: float = 25.0
    snapshot_window_frames: int = 500
    recompute_tail_frames: int = 75
    tail_overlap_frames: int = 100
    validate_every: int = 10
    validation_tolerance: float = 1e-3
    fallback_on_mismatch: bool = True
    mode: str = "manual"


class IncrementalTrackingBuilder:
    def __init__(self, config: IncrementalBuilderConfig | None = None):
        self.cfg = config or IncrementalBuilderConfig()
        self._df: pd.DataFrame | None = None
        self._origin: int = 0
        self._last_global: int = -1
        self._raw_to_slot: dict[str, str] = {}
        self._all_slots: list[str] = all_person_slots(11) + all_referee_slots(3) + ["ball"]
        self._checkpoints: int = 0
        self._fallback_count: int = 0
        self._validated: int = 0
        self._mismatches: int = 0

    def init_from_full(
        self,
        tracking_df: pd.DataFrame,
        snapshot_origin: int,
        tracks_buffer: Mapping[str, Any],
        full_frames: int,
    ) -> None:
        """Initialize from first full adapter checkpoint."""
        self._df = tracking_df.copy()
        # Reindex frame_id to 0-based
        self._df.index = range(len(self._df))
        self._df["frame_id"] = np.arange(len(self._df), dtype=np.int32)
        self._origin = snapshot_origin
        self._last_global = full_frames - 1
        self._checkpoints = 1

        # Build canonical slot mapping
        self._raw_to_slot = {}
        for cid in range(1, 23):
            slot = canonical_id_to_person_slot(cid)
            if slot:
                self._raw_to_slot[str(cid)] = slot
        for cid in range(23, 26):
            slot = canonical_id_to_referee_slot(cid)
            if slot:
                self._raw_to_slot[str(cid)] = slot

    @property
    def tracking_df(self) -> pd.DataFrame | None:
        return self._df

    @property
    def raw_to_slot(self) -> dict[str, str]:
        return self._raw_to_slot

    @property
    def origin(self) -> int:
        return self._origin

    @property
    def is_ready(self) -> bool:
        return self._df is not None

    def extend(self, tracks_buffer: Mapping[str, Any], full_frames: int) -> int:
        """Extend tracking with new frames. Returns number of rows added."""
        if self._df is None:
            return 0
        if self._last_global >= full_frames - 1:
            return 0

        new_start = self._last_global + 1
        new_end = full_frames - 1
        n_new = new_end - new_start + 1
        if n_new <= 0:
            return 0

        dt = 1.0 / float(self.cfg.fps)
        new_rows: list[dict[str, Any]] = []
        next_local = len(self._df)

        for f in range(new_start, new_end + 1):
            row: dict[str, Any] = {
                "frame_id": next_local,
                "period_id": 1,
                "timestamp": float(self._origin + next_local) / float(self.cfg.fps),
                "phase_id": 1,
                "episode_id": 1,
                "ball_state": "alive",
                "ball_owning_team_id": None,
                "player_id": None,
            }

            for cls in ("player", "goalkeeper"):
                frames = tracks_buffer.get(cls, [])
                if f >= len(frames):
                    continue
                for raw_id, payload in frames[f].items():
                    slot = self._raw_to_slot.get(str(raw_id))
                    if slot is None:
                        continue
                    fp = payload.get("field_position_m")
                    if fp and len(fp) >= 2:
                        row[f"{slot}_x"] = float(fp[0])
                        row[f"{slot}_y"] = float(fp[1])

            refs = tracks_buffer.get("referee", [])
            if f < len(refs):
                for raw_id, payload in refs[f].items():
                    slot = self._raw_to_slot.get(str(raw_id))
                    if slot is None:
                        continue
                    fp = payload.get("field_position_m")
                    if fp and len(fp) >= 2:
                        row[f"{slot}_x"] = float(fp[0])
                        row[f"{slot}_y"] = float(fp[1])

            balls = tracks_buffer.get("ball", [])
            if f < len(balls):
                for _, payload in balls[f].items():
                    bbox = payload.get("bbox")
                    if bbox and len(bbox) >= 4:
                        row["ball_x"] = (float(bbox[0]) + float(bbox[2])) / 2.0
                        row["ball_y"] = (float(bbox[1]) + float(bbox[3])) / 2.0
                    break

            new_rows.append(row)
            next_local += 1

        if not new_rows:
            return 0

        new_df = pd.DataFrame(new_rows)

        # Forward-fill missing x/y from last known position for each slot
        for slot in self._all_slots:
            xc, yc = f"{slot}_x", f"{slot}_y"
            if xc not in new_df.columns:
                new_df[xc] = np.nan
            if yc not in new_df.columns:
                new_df[yc] = np.nan

        # Get last known x/y per slot from existing tracking
        last_vals: dict[str, tuple[float, float]] = {}
        for slot in self._all_slots:
            xc, yc = f"{slot}_x", f"{slot}_y"
            if xc in self._df.columns and yc in self._df.columns:
                xs = self._df[xc].dropna()
                ys = self._df[yc].dropna()
                if len(xs) > 0 and len(ys) > 0:
                    last_vals[slot] = (float(xs.iloc[-1]), float(ys.iloc[-1]))

        # Forward-fill new rows
        for i in range(len(new_df)):
            for slot in self._all_slots:
                xc, yc = f"{slot}_x", f"{slot}_y"
                if pd.isna(new_df.iloc[i][xc]):
                    if i > 0 and not pd.isna(new_df.iloc[i-1][xc]):
                        new_df.iloc[i, new_df.columns.get_loc(xc)] = new_df.iloc[i-1][xc]
                        new_df.iloc[i, new_df.columns.get_loc(yc)] = new_df.iloc[i-1][yc]
                    elif slot in last_vals:
                        new_df.iloc[i, new_df.columns.get_loc(xc)] = last_vals[slot][0]
                        new_df.iloc[i, new_df.columns.get_loc(yc)] = last_vals[slot][1]
                last_vals[slot] = (float(new_df.iloc[i][xc]) if not pd.isna(new_df.iloc[i][xc]) else last_vals.get(slot, (0.0, 0.0))[0],
                                   float(new_df.iloc[i][yc]) if not pd.isna(new_df.iloc[i][yc]) else last_vals.get(slot, (0.0, 0.0))[1])

        # Compute motion features on tail of old + all new
        tail_len = min(self.cfg.recompute_tail_frames, len(self._df))
        old_tail = self._df.iloc[-tail_len:] if tail_len > 0 else self._df.iloc[:0]
        combined = pd.concat([old_tail, new_df], ignore_index=True)

        for slot in self._all_slots:
            for axis in ("x", "y", "vx", "vy", "speed", "accel"):
                col = f"{slot}_{axis}"
                if col not in combined.columns:
                    combined[col] = np.nan

            xc, yc = f"{slot}_x", f"{slot}_y"
            vxc, vyc = f"{slot}_vx", f"{slot}_vy"
            sc = f"{slot}_speed"
            ac = f"{slot}_accel"

            combined[vxc] = combined[xc].diff().fillna(0.0) / dt
            combined[vyc] = combined[yc].diff().fillna(0.0) / dt
            combined[sc] = np.sqrt(combined[vxc] ** 2 + combined[vyc] ** 2)
            combined[ac] = combined[sc].diff().fillna(0.0) / dt

            # Restore tail frame values from original (they were already correct)
            for i in range(tail_len):
                col_names = [vxc, vyc, sc, ac]
                for cn in col_names:
                    if cn in self._df.columns:
                        combined.iloc[i, combined.columns.get_loc(cn)] = self._df[cn].iloc[-tail_len + i]

        # Keep only new portion
        new_result = combined.iloc[tail_len:].reset_index(drop=True)

        # Fill missing columns
        for col in self._df.columns:
            if col not in new_result.columns:
                new_result[col] = np.nan
        new_result = new_result[self._df.columns]

        self._df = pd.concat([self._df, new_result], ignore_index=True)
        self._df.index = range(len(self._df))
        self._df["frame_id"] = np.arange(len(self._df), dtype=np.int32)
        self._last_global = new_end

        # Window
        window = self.cfg.snapshot_window_frames
        if len(self._df) > window:
            drop = len(self._df) - window
            self._df = self._df.iloc[drop:].reset_index(drop=True)
            self._df["frame_id"] = np.arange(len(self._df), dtype=np.int32)
            self._origin += drop

        return n_new

    def prepare_tail_snapshot(
        self, tracks_buffer: Mapping[str, Any], full_frames: int
    ) -> tuple[dict[str, Any], int, int]:
        """Build tracks dict for tail window. Returns (snapshot, window_origin, update_start_global)."""
        window = self.cfg.snapshot_window_frames
        tail_frames = self.cfg.recompute_tail_frames
        overlap = self.cfg.tail_overlap_frames
        window_end = full_frames - 1
        window_start = max(0, window_end - window + 1)
        update_start = max(window_start, window_end - tail_frames + 1)
        snapshot_start = max(0, update_start - overlap)

        trimmed: dict[str, Any] = {}
        for cls in TRACK_CLASSES:
            data = tracks_buffer.get(cls, [])
            trimmed[cls] = data[snapshot_start:window_end + 1]
        return trimmed, snapshot_start, update_start

    def merge_tail_from_adapter(
        self,
        adapter_tracking_df: pd.DataFrame,
        snapshot_origin: int,
        update_start_global: int,
        full_frames: int,
    ) -> None:
        """Replace tail rows in cached tracking with adapter output. Keeps head intact."""
        window = self.cfg.snapshot_window_frames
        frame_count = len(adapter_tracking_df)
        if frame_count == 0 or self._df is None:
            return

        # Adapter tracking covers frames [snapshot_origin, snapshot_origin + frame_count - 1]
        # We want adapter rows for frames >= update_start_global
        tail_start_local = max(0, update_start_global - snapshot_origin)

        # Keep only head from cached tracking (frames < update_start_global)
        cached_head = self._df[self._df["frame_id"] < update_start_global - self._origin].copy() if self._origin <= update_start_global else self._df.iloc[:0]

        # Take tail from adapter
        adapter_tail = adapter_tracking_df.iloc[tail_start_local:].copy()
        adapter_tail["frame_id"] = np.arange(len(cached_head), len(cached_head) + len(adapter_tail), dtype=np.int32)

        # Merge
        self._origin = min(self._origin, snapshot_origin + tail_start_local)
        self._df = pd.concat([cached_head, adapter_tail], ignore_index=True)
        self._df["frame_id"] = np.arange(len(self._df), dtype=np.int32)
        self._last_global = full_frames - 1

        # Window
        if len(self._df) > window:
            drop = len(self._df) - window
            self._df = self._df.iloc[drop:].reset_index(drop=True)
            self._df["frame_id"] = np.arange(len(self._df), dtype=np.int32)
            self._origin += drop

    def validate_and_fallback(
        self,
        full_tracking_df: pd.DataFrame,
        origin: int,
        tracks_buffer: Mapping[str, Any],
        full_frames: int,
    ) -> dict[str, Any]:
        """Compare against full adapter output. Returns mismatch stats."""
        self._validated += 1
        result: dict[str, Any] = {"validated": True, "mismatch": False, "differences": 0, "max_abs_diff": 0.0}

        if full_tracking_df.shape != self._df.shape:
            result["mismatch"] = True
            result["differences"] = -1
            result["shape_diff"] = f"{full_tracking_df.shape} vs {self._df.shape}"
            if self.cfg.fallback_on_mismatch:
                self._fallback_reset(full_tracking_df, origin, full_frames)
            return result

        tol = self.cfg.validation_tolerance
        max_diff = 0.0
        n_diffs = 0
        for col in self._df.columns:
            if col == "frame_id":
                continue
            if self._df[col].dtype in ("float32", "float64", "int32", "int64"):
                diff = (self._df[col].fillna(0) - full_tracking_df[col].fillna(0)).abs()
                col_max = float(diff.max())
                if col_max > tol:
                    n_diffs += 1
                    max_diff = max(max_diff, col_max)
            else:
                if not (self._df[col].astype(str).fillna("") == full_tracking_df[col].astype(str).fillna("")).all():
                    n_diffs += 1

        result["differences"] = n_diffs
        result["max_abs_diff"] = round(max_diff, 6)
        if n_diffs > 0:
            result["mismatch"] = True
            self._mismatches += 1
            if self.cfg.fallback_on_mismatch:
                self._fallback_reset(full_tracking_df, origin, full_frames)

        return result

    def _fallback_reset(
        self,
        full_tracking_df: pd.DataFrame,
        origin: int,
        full_frames: int,
    ) -> None:
        """Reset from full adapter output."""
        self._df = full_tracking_df.copy()
        self._df.index = range(len(self._df))
        self._df["frame_id"] = np.arange(len(self._df), dtype=np.int32)
        self._origin = origin
        self._last_global = full_frames - 1
        self._fallback_count += 1

    def stats(self) -> dict[str, Any]:
        return {
            "tracking_rows": len(self._df) if self._df is not None else 0,
            "tracking_origin": self._origin,
            "last_global_frame": self._last_global,
            "checkpoints": self._checkpoints,
            "fallbacks": self._fallback_count,
            "validations": self._validated,
            "mismatches": self._mismatches,
        }
