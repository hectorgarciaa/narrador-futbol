from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import json
import numpy as np
import pandas as pd


PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
OUTSIDE_NODE_POINTS = {
    "out_left": (0.0, PITCH_WIDTH_M / 2.0),
    "out_right": (PITCH_LENGTH_M, PITCH_WIDTH_M / 2.0),
    "out_bottom": (PITCH_LENGTH_M / 2.0, 0.0),
    "out_top": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M),
}


class PathCRFDrawer:
    def __init__(self) -> None:
        self.pitch_color = (25, 105, 45)
        self.line_color = (235, 235, 235)
        self.home_color = (46, 134, 193)
        self.away_color = (39, 174, 96)
        self.referee_color = (210, 210, 210)
        self.ball_color = (245, 245, 245)
        self.outside_color = (70, 70, 70)
        self.active_edge_color = (0, 215, 255)
        self.active_src_color = (0, 165, 255)
        self.active_dst_color = (0, 255, 255)
        self.text_color = (245, 245, 245)
        self.node_border_color = (20, 20, 20)
        self.ball_border_color = (20, 20, 20)

    @staticmethod
    def _can_show_gui() -> bool:
        if sys.platform != "linux":
            return True
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    @staticmethod
    def _normalize_frame_df(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "frame_id" in result.columns:
            result = result.set_index("frame_id")
        result.index.name = "frame_id"
        return result.sort_index()

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            casted = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(casted):
            return None
        return casted

    @staticmethod
    def _player_label(node_id: str) -> str:
        if node_id.startswith("home_"):
            return f"H{node_id.split('_')[-1]}"
        if node_id.startswith("away_"):
            return f"A{node_id.split('_')[-1]}"
        if node_id.startswith("referee_"):
            return f"R{node_id.split('_')[-1]}"
        return str(node_id)

    def _to_canvas_xy(self, point_m: tuple[float, float], frame_w: int, frame_h: int) -> tuple[int, int]:
        margin_x = int(frame_w * 0.07)
        margin_y = int(frame_h * 0.08)
        usable_w = max(1, frame_w - (2 * margin_x))
        usable_h = max(1, frame_h - (2 * margin_y))
        x_m = float(np.clip(point_m[0], 0.0, PITCH_LENGTH_M))
        y_m = float(np.clip(point_m[1], 0.0, PITCH_WIDTH_M))
        x = margin_x + int((x_m / PITCH_LENGTH_M) * usable_w)
        y = margin_y + int((y_m / PITCH_WIDTH_M) * usable_h)
        return int(np.clip(x, 0, frame_w - 1)), int(np.clip(y, 0, frame_h - 1))

    def _draw_pitch_background(self, frame: np.ndarray) -> None:
        frame[:] = self.pitch_color
        h, w = frame.shape[:2]
        margin_x = int(w * 0.07)
        margin_y = int(h * 0.08)
        left = margin_x
        right = w - margin_x
        top = margin_y
        bottom = h - margin_y
        mid_x = (left + right) // 2
        mid_y = (top + bottom) // 2

        cv2.rectangle(frame, (left, top), (right, bottom), self.line_color, 2)
        cv2.line(frame, (mid_x, top), (mid_x, bottom), self.line_color, 2)
        cv2.circle(frame, (mid_x, mid_y), max(16, int(min(w, h) * 0.08)), self.line_color, 2)

        box_w = int((16.5 / PITCH_LENGTH_M) * (right - left))
        box_h = int((40.32 / PITCH_WIDTH_M) * (bottom - top))
        six_w = int((5.5 / PITCH_LENGTH_M) * (right - left))
        six_h = int((18.32 / PITCH_WIDTH_M) * (bottom - top))

        top_box = mid_y - (box_h // 2)
        bottom_box = mid_y + (box_h // 2)
        cv2.rectangle(frame, (left, top_box), (left + box_w, bottom_box), self.line_color, 2)
        cv2.rectangle(frame, (right - box_w, top_box), (right, bottom_box), self.line_color, 2)

        top_six = mid_y - (six_h // 2)
        bottom_six = mid_y + (six_h // 2)
        cv2.rectangle(frame, (left, top_six), (left + six_w, bottom_six), self.line_color, 2)
        cv2.rectangle(frame, (right - six_w, top_six), (right, bottom_six), self.line_color, 2)

        penalty_left = left + int((11.0 / PITCH_LENGTH_M) * (right - left))
        penalty_right = right - int((11.0 / PITCH_LENGTH_M) * (right - left))
        cv2.circle(frame, (penalty_left, mid_y), 3, self.line_color, -1)
        cv2.circle(frame, (penalty_right, mid_y), 3, self.line_color, -1)

        goal_half_h = max(10, int((7.32 / PITCH_WIDTH_M) * (bottom - top) * 0.5))
        cv2.line(frame, (left, mid_y - goal_half_h), (left - 10, mid_y - goal_half_h), self.line_color, 2)
        cv2.line(frame, (left, mid_y + goal_half_h), (left - 10, mid_y + goal_half_h), self.line_color, 2)
        cv2.line(frame, (left - 10, mid_y - goal_half_h), (left - 10, mid_y + goal_half_h), self.line_color, 2)
        cv2.line(frame, (right, mid_y - goal_half_h), (right + 10, mid_y - goal_half_h), self.line_color, 2)
        cv2.line(frame, (right, mid_y + goal_half_h), (right + 10, mid_y + goal_half_h), self.line_color, 2)
        cv2.line(frame, (right + 10, mid_y - goal_half_h), (right + 10, mid_y + goal_half_h), self.line_color, 2)

    def _node_position(self, tracking_row: pd.Series, node_id: str) -> tuple[float, float] | None:
        if node_id in OUTSIDE_NODE_POINTS:
            return OUTSIDE_NODE_POINTS[node_id]
        x_value = self._safe_float(tracking_row.get(f"{node_id}_x"))
        y_value = self._safe_float(tracking_row.get(f"{node_id}_y"))
        if x_value is None or y_value is None:
            return None
        return x_value, y_value

    def _draw_text_box(
        self,
        frame: np.ndarray,
        text: str,
        origin: tuple[int, int],
        font_scale: float = 0.55,
        bg_color: tuple[int, int, int] = (20, 20, 20),
        text_color: tuple[int, int, int] = (245, 245, 245),
    ) -> None:
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        x, y = origin
        top_left = (max(0, x - 6), max(0, y - text_h - 8))
        bottom_right = (
            min(frame.shape[1] - 1, x + text_w + 6),
            min(frame.shape[0] - 1, y + baseline + 6),
        )
        cv2.rectangle(frame, top_left, bottom_right, bg_color, -1)
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_node(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        node_id: str,
        role: str,
        active_src: bool,
        active_dst: bool,
        label_override: str | None = None,
    ) -> None:
        if role == "home":
            fill_color = self.home_color
            radius = 18
        elif role == "away":
            fill_color = self.away_color
            radius = 18
        elif role == "referee":
            fill_color = self.referee_color
            radius = 12
        else:
            fill_color = self.outside_color
            radius = 11

        x, y = center
        if role == "outside":
            points = np.array(
                [
                    [x, y - radius],
                    [x + radius, y],
                    [x, y + radius],
                    [x - radius, y],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(frame, points, fill_color)
            cv2.polylines(frame, [points], isClosed=True, color=self.node_border_color, thickness=2)
        else:
            cv2.circle(frame, center, radius, fill_color, -1)
            cv2.circle(frame, center, radius, self.node_border_color, 2)

        if active_src:
            cv2.circle(frame, center, radius + 5, self.active_src_color, 2)
        if active_dst:
            cv2.circle(frame, center, radius + 9 if active_src else radius + 5, self.active_dst_color, 2)

        base_label = self._player_label(node_id)
        label = label_override if label_override else base_label
        cv2.putText(
            frame,
            label,
            (x - 12, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.text_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_ball(self, frame: np.ndarray, tracking_row: pd.Series, frame_w: int, frame_h: int) -> None:
        ball_x = self._safe_float(tracking_row.get("ball_x"))
        ball_y = self._safe_float(tracking_row.get("ball_y"))
        if ball_x is None or ball_y is None:
            return
        center = self._to_canvas_xy((ball_x, ball_y), frame_w, frame_h)
        cv2.circle(frame, center, 7, self.ball_color, -1)
        cv2.circle(frame, center, 7, self.ball_border_color, 2)

    def _draw_active_edge(
        self,
        frame: np.ndarray,
        edge_row: pd.Series | None,
        node_positions: Mapping[str, tuple[int, int]],
    ) -> tuple[str | None, str | None]:
        if edge_row is None:
            return None, None

        edge_src = edge_row.get("edge_src")
        edge_dst = edge_row.get("edge_dst")
        if not isinstance(edge_src, str) or not isinstance(edge_dst, str):
            return None, None

        src_xy = node_positions.get(edge_src)
        dst_xy = node_positions.get(edge_dst)
        if src_xy is None or dst_xy is None:
            return edge_src, edge_dst

        if edge_src == edge_dst:
            cv2.circle(frame, src_xy, 30, self.active_edge_color, 2)
            return edge_src, edge_dst

        cv2.arrowedLine(
            frame,
            src_xy,
            dst_xy,
            self.active_edge_color,
            3,
            cv2.LINE_AA,
            tipLength=0.12,
        )
        return edge_src, edge_dst

    @staticmethod
    def _safe_bbox(bbox: Any) -> tuple[int, int, int, int] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        except (TypeError, ValueError):
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_center(bbox: tuple[int, int, int, int] | None) -> tuple[int, int] | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))

    @staticmethod
    def _normalize_tracks_payload(tracks_payload: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
        if not isinstance(tracks_payload, Mapping):
            return {}
        normalized: dict[str, list[dict[str, Any]]] = {}
        for class_name in ("player", "goalkeeper", "referee", "ball"):
            frames = tracks_payload.get(class_name, [])
            if isinstance(frames, list):
                normalized[class_name] = frames
        return normalized

    @staticmethod
    def _segment_role_label(payload: Mapping[str, Any]) -> str | None:
        role = (
            payload.get("display_role_slot")
            or payload.get("expected_role_slot")
            or payload.get("segment_majority_expected_role_slot")
            or payload.get("segment_majority_role")
            or payload.get("predicted_role")
            or payload.get("predicted_role_frame")
        )
        if not role:
            return None
        token = str(role).strip().upper().replace("-", "_").replace(" ", "_")
        token = "_".join(part for part in token.split("_") if part)
        if token in {"MC_IZQ", "MC_DCHO", "DC_IZQ", "DC_DCHO"}:
            return token
        return str(role)

    def _frame_node_label_overrides(
        self,
        frame_tracks: Mapping[str, dict[str, Mapping[str, Any]]],
        raw_to_slot: Mapping[str, str],
    ) -> dict[str, str]:
        overrides: dict[str, str] = {}
        for class_name in ("player", "goalkeeper", "referee"):
            for raw_id, payload in frame_tracks.get(class_name, {}).items():
                if not isinstance(payload, Mapping):
                    continue
                slot = raw_to_slot.get(str(raw_id))
                if not slot:
                    continue
                base = self._player_label(str(slot))
                player_name = str(payload.get("player_name") or "").strip()
                role = self._segment_role_label(payload)
                parts = [base]
                if player_name:
                    parts.append(player_name)
                if role:
                    parts.append(role)
                overrides[str(slot)] = " ".join(parts)
        return overrides

    def _resolve_slot_lookup(self, conversion_summary: Mapping[str, Any] | None) -> tuple[dict[str, str], dict[str, str]]:
        if not isinstance(conversion_summary, Mapping):
            return {}, {}
        raw_to_slot: dict[str, str] = {}
        for mapping_name in ("person_slot_assignments", "referee_slot_assignments"):
            mapping = conversion_summary.get(mapping_name)
            if not isinstance(mapping, Mapping):
                continue
            for raw_id, slot_name in mapping.items():
                raw_to_slot[str(raw_id)] = str(slot_name)
        slot_to_raw = {slot_name: raw_id for raw_id, slot_name in raw_to_slot.items()}
        return raw_to_slot, slot_to_raw

    def _frame_tracks_payload(
        self,
        tracks_payload: Mapping[str, list[dict[str, Any]]],
        frame_id: int,
    ) -> dict[str, dict[str, Mapping[str, Any]]]:
        result = {class_name: {} for class_name in ("player", "goalkeeper", "referee", "ball")}
        for class_name, frames in tracks_payload.items():
            if frame_id < 0 or frame_id >= len(frames):
                continue
            frame_map = frames[frame_id]
            if not isinstance(frame_map, Mapping):
                continue
            result[class_name] = {str(raw_id): payload for raw_id, payload in frame_map.items() if isinstance(payload, Mapping)}
        return result

    def _role_color(self, slot_name: str | None, class_name: str) -> tuple[int, int, int]:
        if isinstance(slot_name, str) and slot_name.startswith("home_"):
            return self.home_color
        if isinstance(slot_name, str) and slot_name.startswith("away_"):
            return self.away_color
        if isinstance(slot_name, str) and slot_name.startswith("referee_"):
            return self.referee_color
        if class_name == "ball":
            return self.ball_color
        if class_name == "referee":
            return self.referee_color
        return self.home_color if class_name == "goalkeeper" else self.away_color

    def _draw_bbox_annotation(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        label: str,
        color: tuple[int, int, int],
        active_src: bool,
        active_dst: bool,
    ) -> None:
        x1, y1, x2, y2 = bbox
        thickness = 2
        if active_src:
            cv2.rectangle(
                frame,
                (max(0, x1 - 4), max(0, y1 - 4)),
                (min(frame.shape[1] - 1, x2 + 4), min(frame.shape[0] - 1, y2 + 4)),
                self.active_src_color,
                2,
            )
        if active_dst:
            cv2.rectangle(
                frame,
                (max(0, x1 - 7), max(0, y1 - 7)),
                (min(frame.shape[1] - 1, x2 + 7), min(frame.shape[0] - 1, y2 + 7)),
                self.active_dst_color,
                2,
            )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if label:
            self._draw_text_box(
                frame,
                label,
                (max(6, x1), max(18, y1 - 8)),
                font_scale=0.45,
                bg_color=(18, 18, 18),
            )

    def _draw_tracks_on_video(
        self,
        frame: np.ndarray,
        frame_tracks: Mapping[str, dict[str, Mapping[str, Any]]],
        raw_to_slot: Mapping[str, str],
        edge_src: str | None,
        edge_dst: str | None,
        node_label_overrides: dict[str, str] | None = None,
    ) -> None:
        overrides = node_label_overrides or {}
        ordered_classes = ("player", "goalkeeper", "referee", "ball")
        for class_name in ordered_classes:
            for raw_id, payload in frame_tracks.get(class_name, {}).items():
                bbox = self._safe_bbox(payload.get("bbox"))
                if bbox is None:
                    continue
                slot_name = raw_to_slot.get(str(raw_id))
                if class_name == "ball":
                    label = "BALL"
                elif slot_name is not None:
                    label = overrides.get(slot_name)
                    if not label:
                        base_label = self._player_label(slot_name)
                        role = self._segment_role_label(payload)
                        label = f"{base_label} {role}" if role else base_label
                else:
                    label = f"{class_name[:1].upper()}{raw_id}"
                self._draw_bbox_annotation(
                    frame=frame,
                    bbox=bbox,
                    label=label,
                    color=self._role_color(slot_name, class_name),
                    active_src=(slot_name == edge_src),
                    active_dst=(slot_name == edge_dst),
                )

        ball_payload = next(iter(frame_tracks.get("ball", {}).values()), None)
        if isinstance(ball_payload, Mapping):
            bbox = self._safe_bbox(ball_payload.get("bbox"))
            center = self._bbox_center(bbox)
            if center is not None:
                cv2.circle(frame, center, 5, self.ball_color, -1)
                cv2.circle(frame, center, 5, self.ball_border_color, 2)

    def _draw_video_active_edge(
        self,
        frame: np.ndarray,
        frame_tracks: Mapping[str, dict[str, Mapping[str, Any]]],
        slot_to_raw: Mapping[str, str],
        edge_src: str | None,
        edge_dst: str | None,
    ) -> None:
        if not isinstance(edge_src, str) or not isinstance(edge_dst, str):
            return
        if edge_src in OUTSIDE_NODE_POINTS or edge_dst in OUTSIDE_NODE_POINTS:
            return

        def _slot_center(slot_name: str) -> tuple[int, int] | None:
            raw_id = slot_to_raw.get(slot_name)
            if raw_id is None:
                return None
            for class_name in ("player", "goalkeeper", "referee"):
                payload = frame_tracks.get(class_name, {}).get(str(raw_id))
                if not isinstance(payload, Mapping):
                    continue
                return self._bbox_center(self._safe_bbox(payload.get("bbox")))
            return None

        src_center = _slot_center(edge_src)
        dst_center = _slot_center(edge_dst)
        if src_center is None or dst_center is None:
            return
        if edge_src == edge_dst:
            cv2.circle(frame, src_center, 22, self.active_edge_color, 2)
            return
        cv2.arrowedLine(frame, src_center, dst_center, self.active_edge_color, 3, cv2.LINE_AA, tipLength=0.15)

    def _build_pitch_inset(
        self,
        tracking_row: pd.Series,
        edge_row: pd.Series | None,
        event_row: pd.Series | None,
        frame_size: tuple[int, int],
        node_label_overrides: dict[str, str] | None = None,
    ) -> np.ndarray:
        return self._draw_frame(
            tracking_row=tracking_row,
            edge_row=edge_row,
            event_row=event_row,
            frame_size=frame_size,
            node_label_overrides=node_label_overrides,
        )

    def _blend_inset(
        self,
        base_frame: np.ndarray,
        inset_frame: np.ndarray,
        opacity: float = 0.78,
        margin_px: int = 18,
    ) -> None:
        inset_h, inset_w = inset_frame.shape[:2]
        frame_h, frame_w = base_frame.shape[:2]
        x1 = max(0, frame_w - inset_w - margin_px)
        y1 = max(0, margin_px)
        x2 = min(frame_w, x1 + inset_w)
        y2 = min(frame_h, y1 + inset_h)
        roi = base_frame[y1:y2, x1:x2]
        inset_crop = inset_frame[: y2 - y1, : x2 - x1]
        cv2.addWeighted(inset_crop, float(opacity), roi, float(1.0 - opacity), 0.0, dst=roi)
        cv2.rectangle(base_frame, (x1, y1), (x2 - 1, y2 - 1), self.line_color, 2)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        tracking_row: pd.Series,
        edge_src: str | None,
        edge_dst: str | None,
        event_row: pd.Series | None,
    ) -> None:
        frame_id = int(tracking_row.name) if tracking_row.name is not None else int(tracking_row.get("frame_id", 0))
        timestamp = self._safe_float(tracking_row.get("timestamp")) or 0.0
        self._draw_text_box(frame, f"frame {frame_id} | t={timestamp:0.2f}s", (24, 32))

        if edge_src is not None and edge_dst is not None:
            edge_label = f"EDGE: {edge_src} -> {edge_dst}" if edge_src != edge_dst else f"EDGE: {edge_src} control"
            self._draw_text_box(frame, edge_label, (24, 66), bg_color=(30, 45, 60))

        if event_row is not None and not event_row.empty:
            event_type = str(event_row.get("event_type") or "").strip() or "evento"
            player_id = str(event_row.get("player_id") or "").strip()
            receiver_id = str(event_row.get("receiver_id") or "").strip()
            support = event_row.get("support_frames")
            ratio = event_row.get("support_ratio")
            run_len = event_row.get("longest_consecutive_run")
            if receiver_id:
                event_label = f"EVENT: {event_type} {player_id} -> {receiver_id}"
            else:
                event_label = f"EVENT: {event_type} {player_id}"
            if support is not None:
                parts = [event_label]
                if run_len is not None:
                    parts.append(f"run={int(run_len)}")
                if ratio is not None:
                    parts.append(f"ratio={float(ratio):.2f}")
                if support is not None:
                    n_edges = event_row.get("emit_block_last_frame", event_row.get("end_frame", 0)) - event_row.get("emit_block_first_frame", event_row.get("start_frame", 0)) + 1
                    parts.append(f"sup={int(support)}")
                event_label = "  ".join(parts)
            self._draw_text_box(
                frame,
                event_label,
                (24, frame.shape[0] - 22),
                bg_color=(52, 73, 94),
            )

    def _draw_frame(
        self,
        tracking_row: pd.Series,
        edge_row: pd.Series | None,
        event_row: pd.Series | None,
        frame_size: tuple[int, int],
        node_label_overrides: dict[str, str] | None = None,
    ) -> np.ndarray:
        frame_w, frame_h = frame_size
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        self._draw_pitch_background(canvas)

        node_positions: dict[str, tuple[int, int]] = {}
        edge_src = edge_row.get("edge_src") if edge_row is not None else None
        edge_dst = edge_row.get("edge_dst") if edge_row is not None else None

        for node_id in [f"home_{idx}" for idx in range(1, 12)]:
            point = self._node_position(tracking_row, node_id)
            if point is None:
                continue
            node_positions[node_id] = self._to_canvas_xy(point, frame_w, frame_h)

        for node_id in [f"away_{idx}" for idx in range(1, 12)]:
            point = self._node_position(tracking_row, node_id)
            if point is None:
                continue
            node_positions[node_id] = self._to_canvas_xy(point, frame_w, frame_h)

        for node_id in [f"referee_{idx}" for idx in range(1, 4)]:
            point = self._node_position(tracking_row, node_id)
            if point is None:
                continue
            node_positions[node_id] = self._to_canvas_xy(point, frame_w, frame_h)

        for node_id, point in OUTSIDE_NODE_POINTS.items():
            node_positions[node_id] = self._to_canvas_xy(point, frame_w, frame_h)

        self._draw_active_edge(canvas, edge_row, node_positions)

        overrides = node_label_overrides or {}

        for node_id in [f"home_{idx}" for idx in range(1, 12)]:
            center = node_positions.get(node_id)
            if center is None:
                continue
            self._draw_node(canvas, center, node_id, "home", node_id == edge_src, node_id == edge_dst, label_override=overrides.get(node_id))

        for node_id in [f"away_{idx}" for idx in range(1, 12)]:
            center = node_positions.get(node_id)
            if center is None:
                continue
            self._draw_node(canvas, center, node_id, "away", node_id == edge_src, node_id == edge_dst, label_override=overrides.get(node_id))

        for node_id in [f"referee_{idx}" for idx in range(1, 4)]:
            center = node_positions.get(node_id)
            if center is None:
                continue
            self._draw_node(canvas, center, node_id, "referee", False, False, label_override=overrides.get(node_id))

        for node_id in OUTSIDE_NODE_POINTS:
            center = node_positions[node_id]
            self._draw_node(canvas, center, node_id, "outside", node_id == edge_src, node_id == edge_dst)

        self._draw_ball(canvas, tracking_row, frame_w, frame_h)
        self._draw_overlay(canvas, tracking_row, edge_src, edge_dst, event_row)
        return canvas

    def render_tracking_and_edges(
        self,
        tracking: pd.DataFrame,
        edge_sequence: pd.DataFrame,
        output_path: str | Path,
        events: pd.DataFrame | None = None,
        fps: float = 25.0,
        frame_size: tuple[int, int] = (1280, 720),
        video_path: str | Path | None = None,
        tracks_path: str | Path | None = None,
        conversion_summary: Mapping[str, Any] | None = None,
        show: bool = False,
        window_name: str = "PathCRF",
    ) -> Path:
        tracking_df = self._normalize_frame_df(tracking)
        edge_df = self._normalize_frame_df(edge_sequence)
        edge_df = edge_df.reindex(tracking_df.index)  # sin ffill/bfill: solo edges reales por frame

        events_df = None
        if events is not None and not events.empty:
            events_df = events.copy()
            if "frame_id" in events_df.columns:
                events_df = events_df.set_index("frame_id")

        tracks_payload: dict[str, list[dict[str, Any]]] = {}
        raw_to_slot: dict[str, str] = {}
        slot_to_raw: dict[str, str] = {}
        if tracks_path is not None:
            tracks_path = Path(tracks_path).expanduser().resolve()
            if tracks_path.exists():
                with tracks_path.open("r", encoding="utf-8") as f:
                    tracks_payload = self._normalize_tracks_payload(json.load(f))
        raw_to_slot, slot_to_raw = self._resolve_slot_lookup(conversion_summary)

        video_cap = None
        effective_frame_size = frame_size
        use_video_background = False
        if video_path is not None:
            candidate_video_path = Path(video_path).expanduser().resolve()
            if candidate_video_path.exists():
                video_cap = cv2.VideoCapture(str(candidate_video_path))
                if video_cap.isOpened():
                    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if width > 0 and height > 0:
                        effective_frame_size = (width, height)
                        use_video_background = bool(tracks_payload)
                else:
                    video_cap.release()
                    video_cap = None

        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), effective_frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"No se pudo crear el video de salida: {output_path}")

        show_window = bool(show and self._can_show_gui())
        try:
            for frame_id in tracking_df.index:
                tracking_row = tracking_df.loc[frame_id]
                edge_row = edge_df.loc[frame_id] if frame_id in edge_df.index else None
                event_row = events_df.loc[frame_id] if events_df is not None and frame_id in events_df.index else None
                if isinstance(event_row, pd.DataFrame):
                    event_row = event_row.iloc[0]
                if video_cap is not None:
                    ok, base_frame = video_cap.read()
                    if not ok:
                        break
                else:
                    base_frame = None
                frame_tracks = self._frame_tracks_payload(tracks_payload, int(frame_id))
                node_label_overrides = self._frame_node_label_overrides(frame_tracks, raw_to_slot)

                if use_video_background and base_frame is not None:
                    frame = base_frame.copy()
                    edge_src = edge_row.get("edge_src") if edge_row is not None else None
                    edge_dst = edge_row.get("edge_dst") if edge_row is not None else None
                    self._draw_tracks_on_video(
                        frame, frame_tracks, raw_to_slot, edge_src, edge_dst,
                        node_label_overrides=node_label_overrides,
                    )
                    self._draw_video_active_edge(frame, frame_tracks, slot_to_raw, edge_src, edge_dst)
                    inset_w = max(280, int(frame.shape[1] * 0.28))
                    inset_h = max(180, int(frame.shape[0] * 0.28))
                    inset_frame = self._build_pitch_inset(
                        tracking_row=tracking_row,
                        edge_row=edge_row,
                        event_row=event_row,
                        frame_size=(inset_w, inset_h),
                        node_label_overrides=node_label_overrides,
                    )
                    self._blend_inset(frame, inset_frame, opacity=0.8, margin_px=18)
                    self._draw_overlay(frame, tracking_row, edge_src, edge_dst, event_row)
                else:
                    frame = self._draw_frame(
                        tracking_row=tracking_row,
                        edge_row=edge_row,
                        event_row=event_row,
                        frame_size=effective_frame_size,
                        node_label_overrides=node_label_overrides,
                    )
                writer.write(frame)
                if show_window:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(max(1, int(round(1000.0 / max(fps, 1e-6))))) & 0xFF == ord("q"):
                        break
        finally:
            writer.release()
            if video_cap is not None:
                video_cap.release()
            if show_window:
                cv2.destroyWindow(window_name)

        return output_path
