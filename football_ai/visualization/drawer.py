import os
import sys
import logging
from copy import deepcopy

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0


def _format_role_overlay_label(value):
    token = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    token = "_".join(part for part in token.split("_") if part)
    if token in {"MC_IZQ", "MC_DCHO", "DC_IZQ", "DC_DCHO"}:
        return token
    return str(value)


class Drawer:
    def __init__(self, colors, default_color=(255, 255, 255), visualization_conf=None):
        self.colors = colors
        self.DEFAULT_COLOR = default_color
        self.visualization_conf = visualization_conf or {}
        self.tracked_draw_classes = ("player", "goalkeeper", "referee", "ball")
        self.line_thickness = int(self.visualization_conf.get("line_thickness", 2))
        self.font_scale = float(self.visualization_conf.get("font_scale", 0.5))
        self.compact_font_scale = float(
            self.visualization_conf.get("compact_font_scale", max(0.30, self.font_scale * 0.7))
        )
        self.possession_highlight_color = (0, 255, 255)
        self.continuity_keep_all_seen_ids = bool(
            self.visualization_conf.get("continuity_keep_all_seen_ids", True)
        )
        self.pitch_marker_radius = max(
            8,
            int(self.visualization_conf.get("pitch_marker_radius", 12)),
        )
        configured_team_colors = self.visualization_conf.get("team_colors", {})
        team_color_space = str(
            self.visualization_conf.get("team_colors_space", "bgr")
        ).strip().lower()
        self.team_fill_colors = {}
        if isinstance(configured_team_colors, dict):
            for team_name, color in configured_team_colors.items():
                parsed = self._parse_team_color(color, team_color_space)
                if parsed is not None:
                    self.team_fill_colors[str(team_name).strip().lower()] = parsed

    @staticmethod
    def _can_show_gui():
        if sys.platform != "linux":
            return True
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    @staticmethod
    def _short_class_label(class_name):
        mapping = {
            "player": "p",
            "goalkeeper": "gk",
            "referee": "ref",
            "ball": "",
        }
        return mapping.get(class_name, class_name)

    def _short_debug_class_label(self, class_name):
        short_label = self._short_class_label(class_name)
        if short_label == "ref":
            return "r"
        return short_label

    @staticmethod
    def _parse_bgr_color(color):
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            return None
        try:
            b, g, r = int(color[0]), int(color[1]), int(color[2])
        except (TypeError, ValueError):
            return None
        return (int(np.clip(b, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(r, 0, 255)))

    @classmethod
    def _parse_team_color(cls, color, color_space):
        if color_space == "lab_opencv":
            if not isinstance(color, (list, tuple)) or len(color) < 3:
                return None
            try:
                l_ch = float(color[0])
                a_ch = float(color[1])
                b_ch = float(color[2])
            except (TypeError, ValueError):
                return None
            lab_pixel = np.array([[[l_ch, a_ch, b_ch]]], dtype=np.uint8)
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            bgr = bgr_pixel.reshape(-1).tolist()
            return cls._parse_bgr_color(bgr)
        return cls._parse_bgr_color(color)

    @staticmethod
    def _coerce_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_track_identifier(track_id):
        if track_id is None:
            return None
        try:
            return int(track_id)
        except (TypeError, ValueError):
            return str(track_id)

    @classmethod
    def _track_id_matches(cls, left_track_id, right_track_id):
        if left_track_id is None or right_track_id is None:
            return False
        return (
            cls._normalize_track_identifier(left_track_id)
            == cls._normalize_track_identifier(right_track_id)
        )

    @staticmethod
    def _extract_role_text(data):
        segment_majority_expected_role_slot = data.get("segment_majority_expected_role_slot")
        if segment_majority_expected_role_slot:
            return _format_role_overlay_label(segment_majority_expected_role_slot)
        segment_majority_role = data.get("segment_majority_role")
        if segment_majority_role:
            return _format_role_overlay_label(segment_majority_role)
        display_role_slot = data.get("display_role_slot")
        if display_role_slot:
            return _format_role_overlay_label(display_role_slot)
        expected_role_slot = data.get("expected_role_slot")
        if expected_role_slot:
            return _format_role_overlay_label(expected_role_slot)
        predicted_role = data.get("predicted_role")
        if predicted_role:
            return str(predicted_role)
        predicted_role_frame = data.get("predicted_role_frame")
        if predicted_role_frame:
            return str(predicted_role_frame)
        return None

    def _resolve_debug_detection_class_name(self, det):
        for key in ("class_name", "class_tracker", "class_name_td", "class_relabel", "class_team_detector", "class_yolo", "class"):
            value = det.get(key)
            if value:
                return value
        return ""

    def _resolve_detection_source_classes(self, data):
        class_tracker = (
            data.get("class_tracker")
            or data.get("class_name")
        )
        class_yolo = (
            data.get("class_yolo")
        )
        class_team_detector = (
            data.get("class_name_td")
            or data.get("class_relabel")
            or data.get("class_team_detector")
            or data.get("class")
        )
        return class_tracker, class_yolo, class_team_detector

    def _format_detection_source_label(self, data):
        class_tracker, class_yolo, class_team_detector = self._resolve_detection_source_classes(data)
        label_parts = []
        tracker_token = (
            self._short_debug_class_label(class_tracker)
            if class_tracker
            else "-"
        )
        label_parts.append(f"tr:{tracker_token}")
        if class_yolo:
            label_parts.append(f"y:{self._short_debug_class_label(class_yolo)}")
        else:
            label_parts.append("y:-")
        if class_team_detector:
            label_parts.append(f"td:{self._short_debug_class_label(class_team_detector)}")
        else:
            label_parts.append("td:-")
        return " ".join(label_parts)

    @staticmethod
    def _short_discard_reason(reason):
        token = str(reason or "").strip()
        if not token:
            return ""
        short_map = {
            "canonical_state_missing": "st_miss",
            "canonical_id_used_in_frame": "id_used",
            "canonical_class_unsupported": "cls_unsup",
            "canonical_gate_failed": "gate_fail",
            "raw_tracker_id_unmapped_or_rejected": "bt_unmap",
            "canonical_class_limit_reached": "cls_full",
            "canonical_no_free_id": "no_id",
            "candidate_class_missing": "cand_cls",
            "detection_class_missing": "det_cls",
            "referee_class_incompatible": "ref_cls",
            "team_incompatible": "team",
            "referee_candidate_zone_unknown": "ref_z?",
            "referee_detection_zone_unknown": "det_z?",
            "referee_zone_incompatible": "ref_zone",
            "referee_detection_outside_player_lane": "ref_lane",
            "continuity_motion_incompatible": "mot",
            "continuity_gate_failed": "cont_fail",
            "motion_incompatible_special_seed": "seed_mot",
            "motion_incompatible_reserved_seed": "resv_mot",
            "motion_incompatible": "mot",
            "class_incompatible": "cls",
        }
        compact_parts = []
        for part in token.split("|"):
            part = part.strip()
            if not part:
                continue
            compact_parts.append(short_map.get(part, part[:10]))
        return "|".join(compact_parts)

    def _detection_draw_color(self, class_name, data, fallback_color):
        if class_name in {"player", "goalkeeper"}:
            team_color = self._team_fill_color(self._team_name_for_color(data))
            if team_color is not None:
                return team_color
        return fallback_color

    @staticmethod
    def _nearest_outfield_team_from_distances(distances):
        if not isinstance(distances, dict):
            return None
        valid = []
        for team_name, distance in distances.items():
            if team_name in {None, "referee"}:
                continue
            try:
                numeric_distance = float(distance)
            except (TypeError, ValueError):
                continue
            valid.append((str(team_name), numeric_distance))
        if not valid:
            return None
        valid.sort(key=lambda item: (item[1], item[0]))
        return valid[0][0]

    def _team_name_for_color(self, data):
        if not isinstance(data, dict):
            return None
        team_name = data.get("team")
        if team_name not in {None, ""}:
            return team_name
        return self._nearest_outfield_team_from_distances(data.get("distances"))

    def create_writer(self, video, output_path, output_size=None):
        if not video:
            raise ValueError("Video path is empty.")
        if not os.path.isfile(video):
            raise FileNotFoundError(f"Video not found: {video}")

        root, ext = os.path.splitext(output_path)
        if ext == "":
            output_path = root + ".mp4"

        try:
            cap = cv2.VideoCapture(video)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video}")

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if output_size is None:
                output_width, output_height = width, height
            else:
                output_width, output_height = int(output_size[0]), int(output_size[1])

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not create video writer for: {output_path}")

            return cap, writer

        except Exception as e:
            if "cap" in locals() and cap is not None:
                cap.release()
            raise RuntimeError(f"Error creating video writer: {e}") from e

    def draw_detection(
        self,
        frame,
        class_name,
        data,
        color,
        track_id,
        possession_player_id=None,
        compact=False,
        show_identity_segment=False,
    ):
        if data.get("synthetic_seed"):
            return
        if not isinstance(data.get("bbox"), (list, tuple)) or len(data["bbox"]) < 4:
            return
        x1, y1, x2, y2 = map(int, data["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
        is_possession_player = (
            class_name in {"player", "goalkeeper"}
            and (
                bool(data.get("is_possession_player"))
                or self._track_id_matches(track_id, possession_player_id)
            )
        )
        if is_possession_player:
            pad = max(2, int(self.line_thickness))
            cv2.rectangle(
                frame,
                (max(0, x1 - pad), max(0, y1 - pad)),
                (min(frame.shape[1] - 1, x2 + pad), min(frame.shape[0] - 1, y2 + pad)),
                self.possession_highlight_color,
                max(1, self.line_thickness),
            )

        if compact:
            cls_text = self._short_class_label(class_name)
            label = f"{cls_text}#{track_id}" if cls_text else ""
            if label:
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.compact_font_scale,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            info_lines = []
            source_label = self._format_detection_source_label(data)
            if source_label:
                info_lines.append(source_label)
            role_text = self._extract_role_text(data)
            if role_text:
                info_lines.append(role_text)
            if show_identity_segment:
                identity_segment_id = data.get("identity_segment_id")
                if identity_segment_id is not None:
                    info_lines.append(
                        f"seg:{self._coerce_int(identity_segment_id, identity_segment_id)}"
                    )
            if class_name == "player":
                field_position = data.get("field_position_m")
                if (
                    isinstance(field_position, (list, tuple))
                    and len(field_position) >= 2
                    and field_position[0] is not None
                    and field_position[1] is not None
                ):
                    info_lines.append(
                        f"{self._coerce_int(round(field_position[0]))}, {self._coerce_int(round(field_position[1]))}"
                    )
            for idx, info_label in enumerate(info_lines):
                baseline_y = min(y2 + 13 + (idx * 12), frame.shape[0] - 4)
                is_segment_label = (
                    show_identity_segment
                    and isinstance(info_label, str)
                    and info_label.startswith("seg:")
                )
                highlight_segment_label = bool(
                    is_segment_label and data.get("identity_reset")
                )
                if highlight_segment_label:
                    (text_w, text_h), text_base = cv2.getTextSize(
                        info_label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.compact_font_scale,
                        1,
                    )
                    text_x = int(x1)
                    text_y = int(baseline_y)
                    top_left = (
                        max(0, text_x - 2),
                        max(0, text_y - text_h - 2),
                    )
                    bottom_right = (
                        min(frame.shape[1] - 1, text_x + text_w + 2),
                        min(frame.shape[0] - 1, text_y + text_base + 2),
                    )
                    cv2.rectangle(
                        frame,
                        top_left,
                        bottom_right,
                        (20, 20, 20),
                        -1,
                    )
                    text_color = (0, 255, 255)
                else:
                    text_color = color
                cv2.putText(
                    frame,
                    info_label,
                    (x1, baseline_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.compact_font_scale,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )
            return

        label = f"{class_name} #{track_id}"
        distances = data.get("distances")
        team = data.get("team")
        if distances is not None and team is not None:
            d_str = ", ".join(f"{t}: {d:.1f}" for t, d in distances.items())
            label += f" [{team}] ({d_str})"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

        info_lines = []
        player_name = data.get("player_name")
        lineup_slot = data.get("lineup_slot")
        if player_name:
            if lineup_slot:
                info_lines.append(
                    f"{player_name} | {_format_role_overlay_label(lineup_slot)}"
                )
            else:
                info_lines.append(str(player_name))
        predicted_role_frame = data.get("predicted_role_frame")
        predicted_role = data.get("predicted_role")
        assignment_method = data.get("assignment_method")
        segment_majority_expected_role_slot = data.get("segment_majority_expected_role_slot")
        segment_majority_role = data.get("segment_majority_role")
        expected_role_slot = data.get("expected_role_slot")
        display_role_slot = data.get("display_role_slot")
        constrained_expected_role = (
            bool(assignment_method) or bool(expected_role_slot) or bool(display_role_slot)
        )

        if segment_majority_expected_role_slot:
            info_lines.append(
                f"role: {_format_role_overlay_label(segment_majority_expected_role_slot)}"
            )
        elif segment_majority_role:
            info_lines.append(
                f"role: {_format_role_overlay_label(segment_majority_role)}"
            )
        elif constrained_expected_role and display_role_slot:
            info_lines.append(f"role: {_format_role_overlay_label(display_role_slot)}")
        elif constrained_expected_role and predicted_role:
            info_lines.append(f"role: {predicted_role}")
        elif predicted_role_frame and predicted_role:
            if str(predicted_role_frame) == str(predicted_role):
                info_lines.append(f"role: {predicted_role_frame}")
            else:
                info_lines.append(f"role: {predicted_role_frame} | stable: {predicted_role}")
        elif predicted_role_frame:
            info_lines.append(f"role: {predicted_role_frame}")
        elif predicted_role:
            info_lines.append(f"role: {predicted_role}")

        if class_name == "player":
            field_position = data.get("field_position_m")
            if (
                isinstance(field_position, (list, tuple))
                and len(field_position) >= 2
                and field_position[0] is not None
                and field_position[1] is not None
            ):
                info_lines.append(f"pos(m): {field_position[0]:.1f}, {field_position[1]:.1f}")

        for idx, info_label in enumerate(info_lines):
            baseline_y = min(y2 + 15 + (idx * 14), frame.shape[0] - 5)
            cv2.putText(
                frame,
                info_label,
                (x1, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale * 0.9,
                color,
                1,
                cv2.LINE_AA,
            )

    def draw_all_detections_in_frame(
        self,
        frame,
        class_name,
        class_tracks,
        frame_id,
        possession_player_id=None,
        compact=False,
    ):
        frame_data = class_tracks[frame_id]
        color = self.colors.get(class_name, self.DEFAULT_COLOR)
        for track_id, data in frame_data.items():
            self.draw_detection(
                frame,
                class_name,
                data,
                color,
                track_id,
                possession_player_id=possession_player_id,
                compact=compact,
            )

    def _extract_frame_tracks(self, tracks, frame_id):
        frame_tracks = {class_name: {} for class_name in self.tracked_draw_classes}

        for class_name in self.tracked_draw_classes:
            class_tracks = tracks.get(class_name, [])
            if frame_id < len(class_tracks):
                frame_tracks[class_name] = class_tracks[frame_id]
        return frame_tracks

    def _frame_possession_info(self, frame_tracks):
        info = {
            "team_id": None,
            "player_id": None,
            "reason": None,
        }

        for class_name in ("ball", "player", "goalkeeper", "referee"):
            frame_map = frame_tracks.get(class_name, {})
            if not isinstance(frame_map, dict):
                continue
            for track_id, payload in frame_map.items():
                if not isinstance(payload, dict):
                    continue
                if info["team_id"] is None and payload.get("ball_owning_team_id") is not None:
                    info["team_id"] = payload.get("ball_owning_team_id")
                if info["player_id"] is None:
                    if payload.get("ball_owning_player_id") is not None:
                        info["player_id"] = payload.get("ball_owning_player_id")
                    elif payload.get("player_id") is not None:
                        info["player_id"] = payload.get("player_id")
                if info["reason"] is None and payload.get("possession_reason") is not None:
                    info["reason"] = payload.get("possession_reason")
                if (
                    class_name in {"player", "goalkeeper"}
                    and payload.get("is_possession_player")
                ):
                    info["player_id"] = track_id
                    if info["team_id"] is None:
                        info["team_id"] = payload.get("team")

        if info["team_id"] is None and info["player_id"] is not None:
            for class_name in ("player", "goalkeeper"):
                frame_map = frame_tracks.get(class_name, {})
                if not isinstance(frame_map, dict):
                    continue
                for track_id, payload in frame_map.items():
                    if not isinstance(payload, dict):
                        continue
                    if self._track_id_matches(track_id, info["player_id"]):
                        info["team_id"] = payload.get("team")
                        break
                if info["team_id"] is not None:
                    break

        return info

    @staticmethod
    def _frame_possession_from_tracks(tracks, frame_id):
        if not isinstance(tracks, dict):
            return {}
        possession_frames = tracks.get("possession")
        if not isinstance(possession_frames, list) or frame_id >= len(possession_frames):
            return {}
        frame_payload = possession_frames[frame_id]
        return frame_payload if isinstance(frame_payload, dict) else {}

    @staticmethod
    def _merge_possession_info(primary, fallback):
        merged = {}
        if isinstance(fallback, dict):
            merged.update(fallback)
        if isinstance(primary, dict):
            for key, value in primary.items():
                if value is not None:
                    merged[key] = value
        return merged

    @staticmethod
    def _frame_actions_from_tracks(tracks, frame_id):
        if not isinstance(tracks, dict):
            return {}
        actions_frames = tracks.get("actions_incremental")
        if not isinstance(actions_frames, list) or frame_id >= len(actions_frames):
            return {}
        payload = actions_frames[frame_id]
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _safe_track_id(value):
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _bbox_center_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        bbox = payload.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        except (TypeError, ValueError):
            return None
        return int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))

    def _find_track_center(self, frame_tracks, track_id):
        target = self._safe_track_id(track_id)
        if target is None:
            return None
        for class_name in ("player", "goalkeeper", "referee"):
            frame_map = frame_tracks.get(class_name, {})
            if not isinstance(frame_map, dict):
                continue
            for candidate_id, payload in frame_map.items():
                if self._safe_track_id(candidate_id) != target:
                    continue
                return self._bbox_center_from_payload(payload)
        return None

    def _draw_actions_overlay(self, frame, frame_tracks, actions_info):
        if not isinstance(actions_info, dict):
            return
        raw = actions_info.get("raw_edge") if isinstance(actions_info.get("raw_edge"), dict) else {}
        post = actions_info.get("confirmed_action") if isinstance(actions_info.get("confirmed_action"), dict) else {}
        slot_map = actions_info.get("person_slot_assignments") if isinstance(actions_info.get("person_slot_assignments"), dict) else {}
        ref_slot_map = actions_info.get("referee_slot_assignments") if isinstance(actions_info.get("referee_slot_assignments"), dict) else {}

        raw_src = raw.get("edge_src")
        raw_dst = raw.get("edge_dst")
        raw_src_tid = raw.get("edge_src_track_id")
        raw_dst_tid = raw.get("edge_dst_track_id")

        post_label = str(post.get("event_type") or "None")
        lines = [
            f"PathCRF raw: {raw_src}->{raw_dst}" if raw_src is not None and raw_dst is not None else "PathCRF raw: None",
            f"PathCRF post: {post_label}",
            f"raw src/dst track: {raw_src_tid}->{raw_dst_tid}",
            f"raw src/dst slot: {raw_src}->{raw_dst}",
        ]
        if isinstance(post, dict) and post:
            lines.append(
                f"post src/dst track: {post.get('player_track_id')}->{post.get('receiver_track_id')}"
            )
            lines.append(
                f"post src/dst slot: {post.get('player_slot_id')}->{post.get('receiver_slot_id')}"
            )
        y = 24
        for text in lines:
            (tw, th), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (10, y - th - 8), (22 + tw, y + b + 3), (25, 25, 25), -1)
            cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
            y += 24

        # Flecha amarilla del raw PathCRF sobre el broadcast usando track_ids reales.
        src_center = self._find_track_center(frame_tracks, raw_src_tid)
        dst_center = self._find_track_center(frame_tracks, raw_dst_tid)
        if src_center is not None and dst_center is not None:
            cv2.arrowedLine(frame, src_center, dst_center, (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.18)

        # Etiquetas slot sobre cada track cuando exista mapping.
        merged_slot_map = {}
        merged_slot_map.update({str(k): str(v) for k, v in slot_map.items()})
        merged_slot_map.update({str(k): str(v) for k, v in ref_slot_map.items()})
        for class_name in ("player", "goalkeeper", "referee"):
            for track_id, payload in frame_tracks.get(class_name, {}).items():
                slot = merged_slot_map.get(str(track_id))
                center = self._bbox_center_from_payload(payload)
                if slot is None or center is None:
                    continue
                cv2.putText(frame, slot, (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_possession_banner(self, frame, possession_info, compact=False, top_margin_px=6):
        if not isinstance(possession_info, dict):
            return
        team_id = possession_info.get("team_id")
        player_id = possession_info.get("player_id")

        if team_id is None and player_id is None:
            label = "POS: unknown"
        else:
            label = f"POS: {team_id if team_id is not None else 'unknown'}"
            if player_id is not None and not compact:
                label += f" | ID {player_id}"

        font_scale = self.compact_font_scale if compact else self.font_scale
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        x = 8
        y = max(text_h + 6, int(top_margin_px) + text_h)
        x2 = min(frame.shape[1] - 2, x + text_w + 8)
        y2 = min(frame.shape[0] - 2, y + baseline + 4)
        cv2.rectangle(frame, (x - 4, y - text_h - 4), (x2, y2), (20, 20, 20), -1)
        cv2.putText(
            frame,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self.possession_highlight_color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_tracks_frame(self, frame, frame_tracks, compact=False, possession_info=None):
        return self._draw_tracks_frame_with_options(
            frame,
            frame_tracks,
            compact=compact,
            possession_info=possession_info,
            show_identity_segment=False,
        )

    def _draw_tracks_frame_with_options(
        self,
        frame,
        frame_tracks,
        *,
        compact=False,
        possession_info=None,
        show_identity_segment=False,
    ):
        resolved_possession = (
            possession_info
            if isinstance(possession_info, dict)
            else self._frame_possession_info(frame_tracks)
        )
        possession_player_id = resolved_possession.get("player_id")
        for class_name, frame_data in frame_tracks.items():
            if not isinstance(frame_data, dict):
                continue
            class_color = self.colors.get(class_name, self.DEFAULT_COLOR)
            for track_id, data in frame_data.items():
                if not isinstance(data, dict):
                    continue
                draw_color = self._detection_draw_color(class_name, data, class_color)
                self.draw_detection(
                    frame,
                    class_name,
                    data,
                    draw_color,
                    track_id,
                    possession_player_id=possession_player_id,
                    compact=compact,
                    show_identity_segment=show_identity_segment,
                )
        return frame

    def _draw_panel_title(self, panel, title):
        cv2.putText(
            panel,
            title,
            (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.compact_font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _team_fill_color(self, team_name):
        if team_name is None:
            return None
        team_key = str(team_name).strip().lower()
        configured = self.team_fill_colors.get(team_key)
        if configured is not None:
            return configured
        return None

    @staticmethod
    def _class_border_color(class_name):
        mapping = {
            "player": (30, 30, 30),
            "goalkeeper": (0, 215, 255),
            "referee": (255, 255, 255),
        }
        return mapping.get(class_name, (120, 120, 120))

    @staticmethod
    def _track_pitch_point(data, frame_shape_hw):
        field_position = data.get("field_position_m")
        if (
            isinstance(field_position, (list, tuple))
            and len(field_position) >= 2
            and field_position[0] is not None
            and field_position[1] is not None
        ):
            x_m = float(np.clip(field_position[0], 0.0, PITCH_LENGTH_M))
            y_m = float(np.clip(field_position[1], 0.0, PITCH_WIDTH_M))
            return x_m, y_m
        bbox = data.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and frame_shape_hw is not None:
            frame_h, frame_w = frame_shape_hw
            if frame_h > 0 and frame_w > 0:
                x1, _, x2, y2 = bbox[:4]
                x_ratio = float(np.clip((x1 + x2) * 0.5 / frame_w, 0.0, 1.0))
                y_ratio = float(np.clip(y2 / frame_h, 0.0, 1.0))
                return x_ratio * PITCH_LENGTH_M, y_ratio * PITCH_WIDTH_M
        return None

    def _to_panel_xy(self, point_m, panel_w, panel_h):
        margin_x = int(panel_w * 0.07)
        margin_y = int(panel_h * 0.07)
        usable_w = max(1, panel_w - 2 * margin_x)
        usable_h = max(1, panel_h - 2 * margin_y)
        x_m, y_m = point_m
        x = margin_x + int((x_m / PITCH_LENGTH_M) * usable_w)
        y = margin_y + int((y_m / PITCH_WIDTH_M) * usable_h)
        return int(np.clip(x, 0, panel_w - 1)), int(np.clip(y, 0, panel_h - 1))

    def _draw_pitch_background(self, panel):
        panel[:] = (25, 105, 45)
        h, w = panel.shape[:2]
        margin_x = int(w * 0.07)
        margin_y = int(h * 0.07)
        cv2.rectangle(panel, (margin_x, margin_y), (w - margin_x, h - margin_y), (230, 230, 230), 1)
        mid_x = (margin_x + (w - margin_x)) // 2
        cv2.line(panel, (mid_x, margin_y), (mid_x, h - margin_y), (230, 230, 230), 1)
        center = (mid_x, (margin_y + (h - margin_y)) // 2)
        radius = max(6, int(min(w, h) * 0.08))
        cv2.circle(panel, center, radius, (230, 230, 230), 1)

    def _draw_pitch_panel(self, frame_tracks, frame_shape_hw, panel_shape_hw, possession_info=None):
        panel_h, panel_w = panel_shape_hw
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        self._draw_pitch_background(panel)
        resolved_possession = (
            possession_info
            if isinstance(possession_info, dict)
            else self._frame_possession_info(frame_tracks)
        )
        possession_player_id = resolved_possession.get("player_id")

        for class_name, frame_data in frame_tracks.items():
            if not isinstance(frame_data, dict):
                continue
            for track_id, data in frame_data.items():
                if not isinstance(data, dict):
                    continue
                if data.get("synthetic_seed"):
                    continue
                point_m = self._track_pitch_point(data, frame_shape_hw)
                if point_m is None:
                    continue
                x, y = self._to_panel_xy(point_m, panel_w, panel_h)
                team_fill = self._detection_draw_color(
                    class_name,
                    data,
                    fallback_color=(180, 180, 180),
                )
                border_color = self._class_border_color(class_name)
                if class_name == "ball":
                    side = 8
                    cv2.rectangle(panel, (x - side // 2, y - side // 2), (x + side // 2, y + side // 2), (0, 0, 255), -1)
                    continue

                cv2.circle(panel, (x, y), self.pitch_marker_radius, team_fill, -1)
                cv2.circle(panel, (x, y), self.pitch_marker_radius, border_color, 2)
                if (
                    class_name in {"player", "goalkeeper"}
                    and (
                        bool(data.get("is_possession_player"))
                        or self._track_id_matches(track_id, possession_player_id)
                    )
                ):
                    cv2.circle(panel, (x, y), self.pitch_marker_radius + 3, self.possession_highlight_color, 2)
                cv2.putText(
                    panel,
                    str(track_id),
                    (x - 7, y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.compact_font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                role_text = self._extract_role_text(data)
                if role_text:
                    cv2.putText(
                        panel,
                        role_text,
                        (x - 12, min(panel_h - 5, y + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.compact_font_scale * 0.9,
                        (245, 245, 245),
                        1,
                        cv2.LINE_AA,
                    )
        self._draw_possession_banner(panel, resolved_possession, compact=True, top_margin_px=20)
        return panel

    def _draw_discarded_panel(self, base_frame, debug_frame):
        panel = base_frame.copy()
        debug_frame = debug_frame or {}
        show_reasons = bool(self.visualization_conf.get("discarded_panel_show_reasons", False))
        color_not_tracked = self._parse_bgr_color(
            self.visualization_conf.get("discarded_panel_color_not_tracked", (0, 165, 255))
        ) or (0, 165, 255)
        color_tracked_no_canonical = self._parse_bgr_color(
            self.visualization_conf.get("discarded_panel_color_tracked_no_canonical", (255, 0, 255))
        ) or (255, 0, 255)

        # Backwards compatible: if new keys aren't present, use the legacy list.
        yolo_not_tracked = debug_frame.get("discarded_yolo_not_tracked")
        bytetrack_no_canonical = debug_frame.get("discarded_bytetrack_not_canonical")
        legacy = debug_frame.get("discarded_detections", [])
        if not isinstance(yolo_not_tracked, list) and not isinstance(bytetrack_no_canonical, list):
            yolo_not_tracked = legacy
            bytetrack_no_canonical = []

        def draw_list(items, color):
            for det in items or []:
                bbox = det.get("bbox")
                if not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                class_name = self._resolve_debug_detection_class_name(det)
                draw_color = color
                cls = self._short_debug_class_label(class_name)
                conf = float(det.get("confidence", 0.0))
                source_label = self._format_detection_source_label(det)
                label_parts = [part for part in (source_label, cls) if part]
                label = " ".join(label_parts).strip()
                label = f"{label} {conf:.2f}".strip()
                bt_id = det.get("bytetrack_id")
                if bt_id is not None and bt_id != -1:
                    label = f"{label} bt#{bt_id}"
                if show_reasons:
                    reason = str(det.get("discard_reason") or "").strip()
                    if reason:
                        # Keep label compact to avoid unreadable overlays.
                        compact_reason = self._short_discard_reason(reason)
                        if compact_reason:
                            label = f"{label} {compact_reason[:28]}"
                cv2.rectangle(panel, (x1, y1), (x2, y2), draw_color, 1)
                cv2.putText(
                    panel,
                    label,
                    (x1, max(12, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.compact_font_scale,
                    draw_color,
                    1,
                    cv2.LINE_AA,
                )

        draw_list(yolo_not_tracked, color_not_tracked)
        draw_list(bytetrack_no_canonical, color_tracked_no_canonical)
        return panel

    @staticmethod
    def _sorted_track_ids(frame_map):
        def key_fn(track_id):
            try:
                return (0, int(track_id))
            except (TypeError, ValueError):
                return (1, str(track_id))
        return sorted(frame_map.keys(), key=key_fn)

    def _build_stabilized_frame_tracks(self, frame_tracks, carry_state, expected_counts):
        # Si un ID aparece en una clase en el frame actual, lo retiramos del carry
        # de otras clases para evitar duplicados cross-class en continuidad.
        current_owner_class = {}
        for class_name, frame_map in frame_tracks.items():
            if not isinstance(frame_map, dict):
                continue
            for track_id in frame_map.keys():
                current_owner_class[track_id] = class_name

        for track_id, owner_class in current_owner_class.items():
            for class_name, class_state in carry_state.items():
                if class_name == owner_class:
                    continue
                if isinstance(class_state, dict):
                    class_state.pop(track_id, None)

        for class_name, frame_map in frame_tracks.items():
            class_state = carry_state.setdefault(class_name, {})
            for track_id, data in frame_map.items():
                class_state[track_id] = deepcopy(data)

        stabilized = {class_name: deepcopy(frame_map) for class_name, frame_map in frame_tracks.items()}
        default_expected = {"player": 22, "goalkeeper": 2, "referee": 3, "ball": 1}
        expected = dict(default_expected)
        if isinstance(expected_counts, dict):
            expected.update(expected_counts)

        fill_classes = ("player", "goalkeeper", "referee", "ball")
        for class_name in fill_classes:
            target_count = self._coerce_int(
                expected.get(class_name),
                default_expected.get(class_name, 0),
            )
            class_map = stabilized.setdefault(class_name, {})
            class_state = carry_state.get(class_name, {})
            for track_id in self._sorted_track_ids(class_state):
                if (
                    not self.continuity_keep_all_seen_ids
                    and len(class_map) >= target_count
                ):
                    break
                if track_id in class_map:
                    continue
                carry_data = deepcopy(class_state[track_id])
                carry_data["_carry_forward"] = True
                class_map[track_id] = carry_data

        return stabilized

    def _compose_four_panel_frame(
        self,
        frame,
        tracks,
        frame_id,
        debug_frames,
        carry_state,
        expected_counts,
    ):
        h, w = frame.shape[:2]
        frame_tracks = self._extract_frame_tracks(tracks, frame_id)
        possession_info = self._merge_possession_info(
            self._frame_possession_from_tracks(tracks, frame_id),
            self._frame_possession_info(frame_tracks),
        )
        panel_a = self._draw_tracks_frame(
            frame.copy(),
            frame_tracks,
            compact=True,
            possession_info=possession_info,
        )
        panel_b = self._draw_pitch_panel(
            frame_tracks,
            (h, w),
            (h, w),
            possession_info=possession_info,
        )
        debug_frame = debug_frames[frame_id] if debug_frames and frame_id < len(debug_frames) else None
        panel_c = self._draw_discarded_panel(frame.copy(), debug_frame)
        stabilized_tracks = self._build_stabilized_frame_tracks(frame_tracks, carry_state, expected_counts)
        panel_d = self._draw_tracks_frame_with_options(
            frame.copy(),
            stabilized_tracks,
            compact=True,
            possession_info=possession_info,
            show_identity_segment=True,
        )

        self._draw_panel_title(panel_a, "A) Tracking compact")
        self._draw_panel_title(panel_b, "B) Campo + IDs + rol")
        self._draw_panel_title(panel_c, "C) YOLO descartadas")
        self._draw_panel_title(panel_d, "D) Tracking con continuidad")
        self._draw_possession_banner(panel_a, possession_info, compact=True, top_margin_px=24)
        self._draw_possession_banner(panel_c, possession_info, compact=True, top_margin_px=24)
        self._draw_possession_banner(panel_d, possession_info, compact=True, top_margin_px=24)

        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = panel_a
        canvas[0:h, w:2 * w] = panel_b
        canvas[h:2 * h, 0:w] = panel_c
        canvas[h:2 * h, w:2 * w] = panel_d
        return canvas

    def draw_tracks(
        self,
        tracks,
        video,
        output_path,
        show=False,
        window_name="Tracking",
        four_panel=False,
        debug_frames=None,
        expected_counts=None,
    ):
        cap = None
        out = None
        show_window = show
        if show and not self._can_show_gui():
            logger.warning(
                "show=True pero no hay entorno gráfico (DISPLAY/WAYLAND). "
                "Se desactiva la visualización en tiempo real y solo se guardará el video de salida."
            )
            show_window = False

        try:
            temp_cap = cv2.VideoCapture(video)
            if not temp_cap.isOpened():
                raise RuntimeError(f"Could not open video: {video}")
            width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            temp_cap.release()

            output_size = (width * 2, height * 2) if four_panel else (width, height)
            cap, out = self.create_writer(video, output_path, output_size=output_size)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            carry_state = {}

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame_id >= num_frames:
                    break

                if four_panel:
                    output_frame = self._compose_four_panel_frame(
                        frame,
                        tracks,
                        frame_id,
                        debug_frames,
                        carry_state,
                        expected_counts,
                    )
                else:
                    output_frame = frame
                    frame_tracks = self._extract_frame_tracks(tracks, frame_id)
                    actions_info = self._frame_actions_from_tracks(tracks, frame_id)
                    possession_info = self._merge_possession_info(
                        self._frame_possession_from_tracks(tracks, frame_id),
                        self._frame_possession_info(frame_tracks),
                    )
                    output_frame = self._draw_tracks_frame(
                        output_frame,
                        frame_tracks,
                        compact=False,
                        possession_info=possession_info,
                    )
                    self._draw_possession_banner(
                        output_frame,
                        possession_info,
                        compact=False,
                        top_margin_px=8,
                    )
                    self._draw_actions_overlay(output_frame, frame_tracks, actions_info)

                out.write(output_frame)
                if show_window:
                    cv2.imshow(window_name, output_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_id += 1

        except Exception as e:
            raise RuntimeError(f"Error drawing tracks: {e}") from e

        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()
            if show_window:
                cv2.destroyAllWindows()
