from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
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

        label = self._player_label(node_id)
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
            edge_label = f"edge: {edge_src} -> {edge_dst}" if edge_src != edge_dst else f"edge: {edge_src} control"
            self._draw_text_box(frame, edge_label, (24, 66), bg_color=(30, 45, 60))

        if event_row is not None and not event_row.empty:
            event_type = str(event_row.get("event_type") or "").strip() or "evento"
            player_id = str(event_row.get("player_id") or "").strip()
            receiver_id = str(event_row.get("receiver_id") or "").strip()
            if receiver_id:
                event_label = f"{event_type}: {player_id} -> {receiver_id}"
            else:
                event_label = f"{event_type}: {player_id}"
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

        for node_id in [f"home_{idx}" for idx in range(1, 12)]:
            center = node_positions.get(node_id)
            if center is None:
                continue
            self._draw_node(canvas, center, node_id, "home", node_id == edge_src, node_id == edge_dst)

        for node_id in [f"away_{idx}" for idx in range(1, 12)]:
            center = node_positions.get(node_id)
            if center is None:
                continue
            self._draw_node(canvas, center, node_id, "away", node_id == edge_src, node_id == edge_dst)

        for node_id in [f"referee_{idx}" for idx in range(1, 4)]:
            center = node_positions.get(node_id)
            if center is None:
                continue
            self._draw_node(canvas, center, node_id, "referee", False, False)

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
        show: bool = False,
        window_name: str = "PathCRF",
    ) -> Path:
        tracking_df = self._normalize_frame_df(tracking)
        edge_df = self._normalize_frame_df(edge_sequence)
        edge_df = edge_df.reindex(tracking_df.index).ffill().bfill()

        events_df = None
        if events is not None and not events.empty:
            events_df = events.copy()
            if "frame_id" in events_df.columns:
                events_df = events_df.set_index("frame_id")

        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), frame_size)
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
                frame = self._draw_frame(
                    tracking_row=tracking_row,
                    edge_row=edge_row,
                    event_row=event_row,
                    frame_size=frame_size,
                )
                writer.write(frame)
                if show_window:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(max(1, int(round(1000.0 / max(fps, 1e-6))))) & 0xFF == ord("q"):
                        break
        finally:
            writer.release()
            if show_window:
                cv2.destroyWindow(window_name)

        return output_path
