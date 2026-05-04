from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from football_ai.core import PHASE_CANONICALTRACK, Phase, make_phase_packet

from .assignment_commit import CanonicalAssignmentCommitMixin
from .assignment_ingest import CanonicalAssignmentIngestMixin
from .assignment_process import CanonicalAssignmentProcessMixin
from .logic.ball import CanonicalBallMixin
from .logic.common import CanonicalCommonMixin
from .logic.forced_absorption import CanonicalForcedAbsorptionMixin
from .logic.motion import CanonicalMotionMixin
from .logic.pending import CanonicalPendingAssignmentMixin
from .logic.referee import CanonicalRefereeMixin
from .logic.seeds import CanonicalSeedMixin
from .state import CanonicalTrackState
from .utils import serialize_for_trace


class CanonicalTrackPhase(
    CanonicalPendingAssignmentMixin,
    CanonicalBallMixin,
    CanonicalMotionMixin,
    CanonicalSeedMixin,
    CanonicalForcedAbsorptionMixin,
    CanonicalRefereeMixin,
    CanonicalCommonMixin,
    CanonicalAssignmentProcessMixin,
    CanonicalAssignmentCommitMixin,
    CanonicalAssignmentIngestMixin,
    Phase,
):
    def __init__(
        self,
        max_tracks_per_class,
        tracker_conf,
        projector_conf,
        ball_conf,
    ):
        self.max_tracks_per_class = dict(max_tracks_per_class or {})
        self.max_total_tracks = sum(
            int(limit)
            for class_name, limit in self.max_tracks_per_class.items()
            if str(class_name) != "ball"
        )
        self.player_class_limit = int(self.max_tracks_per_class.get("player", 0) or 0)

        self.ball_min_conf = ball_conf["ball_min_conf"]
        self.ball_expected_position_gate_px = ball_conf["ball_expected_position_gate_px"]
        self.ball_expected_position_gate_growth_per_frame = ball_conf[
            "ball_expected_position_gate_growth_per_frame"
        ]
        self.ball_expected_position_confidence_relax = ball_conf[
            "ball_expected_position_confidence_relax"
        ]
        self.ball_size_ratio_per_frame = ball_conf["ball_size_ratio_per_frame"]
        self.ball_size_min_samples = ball_conf["ball_size_min_samples"]
        self.ball_size_std_factor = ball_conf["ball_size_std_factor"]
        self.ball_size_std_floor = ball_conf["ball_size_std_floor"]
        self.ball_max_reassign_lost_frames = ball_conf["ball_max_reassign_lost_frames"]
        self.ball_high_conf_override = ball_conf["ball_high_conf_override"]

        self.reassign_motion_factor = tracker_conf["reassign_motion_factor"]
        self.reassign_min_distance = tracker_conf["reassign_min_distance"]
        self.reassign_min_samples = tracker_conf["reassign_min_samples"]
        self.reassign_motion_growth_cap_frames = tracker_conf[
            "reassign_motion_growth_cap_frames"
        ]
        self.motion_std_gate_enabled = tracker_conf["motion_std_gate_enabled"]
        self.motion_std_factor = tracker_conf["motion_std_factor"]
        self.motion_std_min_samples = tracker_conf["motion_std_min_samples"]
        self.motion_std_floor = tracker_conf["motion_std_floor"]

        self.reserve_penalty_spot_seed_players = tracker_conf[
            "reserve_penalty_spot_seed_players"
        ]
        self.reserve_penalty_spot_seed_match_distance_m = tracker_conf[
            "reserve_penalty_spot_seed_match_distance_m"
        ]

        configured_special_seed_ids = tracker_conf.get("special_seed_canonical_ids", [1, 2])
        self.special_seed_canonical_ids = tuple(
            sorted(
                {
                    int(canonical_id)
                    for canonical_id in configured_special_seed_ids
                    if self._normalize_optional_positive_int(canonical_id) is not None
                }
            )
        )
        if not self.special_seed_canonical_ids:
            self.special_seed_canonical_ids = (1, 2)

        configured_referee_ids = tracker_conf.get("referee_canonical_ids", [23, 24, 25])
        self.referee_canonical_ids = tuple(
            sorted(
                {
                    int(canonical_id)
                    for canonical_id in configured_referee_ids
                    if self._normalize_optional_positive_int(canonical_id) is not None
                }
            )
        )
        if not self.referee_canonical_ids:
            self.referee_canonical_ids = (23, 24, 25)

        reserved_person_ids = set(self.special_seed_canonical_ids) | set(
            self.referee_canonical_ids
        )
        player_canonical_ids = tuple(
            canonical_id
            for canonical_id in range(1, self.max_total_tracks + 1)
            if canonical_id not in reserved_person_ids
        )
        if self.player_class_limit and len(player_canonical_ids) != self.player_class_limit:
            raise ValueError(
                "CanonicalTrackPhase player canonical range mismatch: "
                f"expected {self.player_class_limit} ids but found {len(player_canonical_ids)}"
            )
        if len(player_canonical_ids) % 2 != 0:
            raise ValueError(
                "CanonicalTrackPhase requires an even number of player canonical ids "
                "to split them across the two teams."
            )
        half_player_ids = len(player_canonical_ids) // 2
        self.player_team_canonical_id_groups = (
            tuple(player_canonical_ids[:half_player_ids]),
            tuple(player_canonical_ids[half_player_ids:]),
        )
        self.player_team_capacity = half_player_ids

        self.referee_sideline_band_distance_m = float(
            tracker_conf.get("referee_sideline_band_distance_m", 3.0)
        )
        self.referee_field_width_m = float(
            projector_conf.get("constructor", {}).get("field_width_m", 68.0)
        )

        self.use_field_positions = projector_conf["enabled"]
        self.field_position_classes = projector_conf["classes"]
        self.field_distance_gate_m = projector_conf["match_distance_gate_m"]
        self.field_distance_gate_max_lost_frames = projector_conf[
            "match_distance_max_lost_frames"
        ]

        raw_field_distance_gate_cap_m = projector_conf["match_distance_cap_m"]
        if raw_field_distance_gate_cap_m is None:
            self.field_distance_gate_cap_m = None
        else:
            gate_cap = float(raw_field_distance_gate_cap_m)
            self.field_distance_gate_cap_m = (
                gate_cap if np.isfinite(gate_cap) and gate_cap > 0.0 else None
            )

        self.field_distance_growth_mode = str(
            projector_conf["match_distance_growth_mode"].strip().lower()
        )
        if self.field_distance_growth_mode not in {"power", "linear_decay"}:
            self.field_distance_growth_mode = "power"

        self.field_distance_lost_exponent = projector_conf["match_distance_lost_exponent"]
        if (
            not np.isfinite(self.field_distance_lost_exponent)
            or self.field_distance_lost_exponent <= 0.0
        ):
            self.field_distance_lost_exponent = 1.0

        self.field_distance_decay_per_frame = projector_conf["match_distance_decay_per_frame"]
        if (
            not np.isfinite(self.field_distance_decay_per_frame)
            or self.field_distance_decay_per_frame < 0.0
        ):
            self.field_distance_decay_per_frame = 0.0

        self.strict_person_class_separation = tracker_conf["strict_person_class_separation"]
        self.max_reassign_lost_frames = tracker_conf["max_reassign_lost_frames"]

        raw_max_reassign_lost_frames_by_class = tracker_conf[
            "max_reassign_lost_frames_by_class"
        ]
        self.max_reassign_lost_frames_by_class = {}
        for class_name, class_limit in raw_max_reassign_lost_frames_by_class.items():
            self.max_reassign_lost_frames_by_class[str(class_name)] = class_limit

        self.forced_absorption_enabled = bool(
            tracker_conf.get("forced_absorption_enabled", True)
        )
        self.forced_absorption_player_min_lost_frames = max(
            1,
            int(tracker_conf.get("forced_absorption_player_min_lost_frames", 20)),
        )
        self.forced_absorption_player_min_consistent_frames = max(
            1,
            int(tracker_conf.get("forced_absorption_player_min_consistent_frames", 10)),
        )

        self.field_projector = None
        if projector_conf.get("enabled"):
            constructor_conf = dict(projector_conf.get("constructor", {}))
            geometry = SimpleNamespace(
                field_length_m=float(constructor_conf.get("field_length_m", 106.0)),
                field_width_m=float(constructor_conf.get("field_width_m", 68.0)),
                center_y_m=float(
                    constructor_conf.get(
                        "center_y_m",
                        constructor_conf.get("field_width_m", 68.0) / 2.0,
                    )
                ),
                penalty_mark_distance_m=float(
                    constructor_conf.get("penalty_mark_distance_m", 11.0)
                ),
            )
            self.field_projector = SimpleNamespace(geometry=geometry)

        self.reset()

    def reset(self):
        self.state = CanonicalTrackState()
        self.class_motion_stats = {
            "player": {"count": 0, "mean": 0.0, "m2": 0.0},
            "goalkeeper": {"count": 0, "mean": 0.0, "m2": 0.0},
            "referee": {"count": 0, "mean": 0.0, "m2": 0.0},
            "ball": {"count": 0, "mean": 0.0, "m2": 0.0},
        }
        self._current_referee_central_x_bounds = None
        self._initialize_reserved_penalty_spot_players(self.state.canonical_state)

    def _ensure_frame_slot(self, n_frame):
        for class_name, class_history in self.state.tracks_history.items():
            while len(class_history) <= n_frame:
                class_history.append({})
            class_history[n_frame] = {}

    def _tracks_frame_from_history(self, n_frame):
        return {
            "player": dict(self.state.tracks_history["player"][n_frame]),
            "goalkeeper": dict(self.state.tracks_history["goalkeeper"][n_frame]),
            "referee": dict(self.state.tracks_history["referee"][n_frame]),
            "ball": dict(self.state.tracks_history["ball"][n_frame]),
        }

    @staticmethod
    def _emit_reserved_seed_tracks(
        canonical_state,
        used_canonical_ids_in_frame,
        tracks,
        n_frame,
        reference_packet,
        build_reserved_seed_track_payload_fn,
    ):
        for canonical_id, state in canonical_state.items():
            if canonical_id in used_canonical_ids_in_frame or not state.get("reserved_seed"):
                continue
            tracks["goalkeeper"][n_frame][canonical_id] = build_reserved_seed_track_payload_fn(
                state,
                reference_packet,
            )

    def _collect_raw_ball_candidates(
        self,
        detection_bbox_xyxy,
        detection_confidence,
        detection_class_labels,
        detection_det_ids,
        field_positions,
        ground_points_projected,
        ball_candidates,
    ):
        for raw_idx, (bbox, score, class_name, det_id) in enumerate(
            zip(
                detection_bbox_xyxy,
                detection_confidence,
                detection_class_labels,
                detection_det_ids,
            )
        ):
            if class_name != "ball" or score < self.ball_min_conf:
                continue
            x1, y1, x2, y2 = bbox.tolist()
            ball_candidates.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(score),
                    "metadata": {
                        "team": None,
                        "distances": None,
                        "shirt_color": None,
                        "bbox_size": float((x2 - x1) * (y2 - y1)),
                        "field_position": field_positions[raw_idx],
                        "ground_point_image": ground_points_projected[raw_idx],
                    },
                    "source": "raw",
                    "raw_det_idx": int(det_id),
                }
            )

    def _select_and_update_ball_track(
        self,
        ball_candidates,
        ball_state,
        n_frame,
        frame_size,
        tracks,
        collect_visual_debug,
        accepted_raw_detection_indexes,
    ):
        selected_ball = self._select_ball_candidate(
            ball_candidates,
            ball_state,
            n_frame,
            frame_size=frame_size,
        )
        if selected_ball is None:
            return ball_state, None

        tracks["ball"][n_frame][0] = self._build_ball_track_payload(
            selected_ball["bbox"],
            selected_ball["confidence"],
            selected_ball["metadata"],
        )
        if collect_visual_debug and selected_ball.get("raw_det_idx") is not None:
            accepted_raw_detection_indexes.add(int(selected_ball["raw_det_idx"]))

        updated_state = self._update_ball_state(
            ball_state if self._ball_state_is_active(ball_state, n_frame) else None,
            selected_ball["bbox"],
            n_frame,
        )
        return updated_state, selected_ball

    def execute(self, bytetrack_packet, collect_visual_debug=False):
        n_frame = int(bytetrack_packet["frame_index"])
        self._ensure_frame_slot(n_frame)

        clean_in = bytetrack_packet["clean"]
        frame_size = (
            int(bytetrack_packet["image_width"]),
            int(bytetrack_packet["image_height"]),
        )

        sorted_tracked_detections = self._sort_tracked_detections(
            self._tracked_detections_from_bytetrack_packet(bytetrack_packet)
        )
        self._update_referee_absorption_context(sorted_tracked_detections)

        (
            bytetrack_raw_detection_indexes,
            bytetrack_id_by_raw_idx,
        ) = self._collect_bytetrack_detection_info(
            sorted_tracked_detections,
            collect_visual_debug,
        )

        bytetrack_not_tracked_reason_by_raw_idx = (
            self._bytetrack_debug_map_from_packet(bytetrack_packet)
            if collect_visual_debug
            else {}
        )

        used_canonical_ids_in_frame = set()
        bytetrack_discard_reason_by_raw_idx = {}
        accepted_raw_detection_indexes = set()
        pending_detections = []
        ball_candidates = []

        self._phase_process_tracked_detections(
            sorted_tracked_detections,
            self.state.canonical_state,
            self.state.raw_to_canonical_id,
            self.state.canonical_to_raw_id,
            used_canonical_ids_in_frame,
            self.state.tracks_history,
            accepted_raw_detection_indexes,
            collect_visual_debug,
            n_frame,
            pending_detections,
            ball_candidates,
            bytetrack_discard_reason_by_raw_idx,
        )

        self._phase_assign_pending_detections(
            pending_detections,
            self.state.canonical_state,
            used_canonical_ids_in_frame,
            n_frame,
            collect_visual_debug,
            bytetrack_discard_reason_by_raw_idx,
            self.state.canonical_to_raw_id,
            self.state.raw_to_canonical_id,
            self.state.tracks_history,
            accepted_raw_detection_indexes,
            self.state.forced_absorption_state,
        )

        self._emit_reserved_seed_tracks(
            self.state.canonical_state,
            used_canonical_ids_in_frame,
            self.state.tracks_history,
            n_frame,
            bytetrack_packet,
            self._build_reserved_seed_track_payload,
        )

        detection_bbox_xyxy = np.asarray(clean_in["bbox_xyxy"], dtype=np.float32).reshape(-1, 4)
        detection_confidence = np.asarray(clean_in["confidence"], dtype=np.float32).reshape(-1)
        detection_class_labels = [str(class_name) for class_name in clean_in["class_name"]]
        detection_det_ids = np.asarray(clean_in["det_id"], dtype=np.int32).reshape(-1)
        field_positions = np.asarray(clean_in["field_positions_m"], dtype=np.float32).reshape(-1, 2)
        ground_points_projected = np.asarray(
            clean_in["ground_points_image_original"],
            dtype=np.float32,
        ).reshape(-1, 2)

        self._collect_raw_ball_candidates(
            detection_bbox_xyxy,
            detection_confidence,
            detection_class_labels,
            detection_det_ids,
            field_positions,
            ground_points_projected,
            ball_candidates,
        )

        self.state.ball_state, selected_ball = self._select_and_update_ball_track(
            ball_candidates,
            self.state.ball_state,
            n_frame,
            frame_size,
            self.state.tracks_history,
            collect_visual_debug,
            accepted_raw_detection_indexes,
        )

        tracks_frame = self._tracks_frame_from_history(n_frame)
        canonical_ids_in_frame = sorted(
            {
                int(canonical_id)
                for class_name in ("player", "goalkeeper", "referee")
                for canonical_id in tracks_frame[class_name].keys()
            }
        )

        pending_assignments_debug = []
        dropped_count = 0
        for pending_idx, pending in enumerate(pending_detections):
            raw_idx = pending.get("raw_detection_idx")
            is_accepted = (
                raw_idx is not None
                and int(raw_idx) in accepted_raw_detection_indexes
            )
            if not is_accepted:
                dropped_count += 1

            pending_assignments_debug.append(
                {
                    "pending_idx": int(pending_idx),
                    "raw_tracker_id": int(pending["raw_tracker_id"]),
                    "raw_detection_idx": raw_idx,
                    "preferred_class_name": pending.get("preferred_class_name"),
                    "assigned": bool(is_accepted),
                    "assigned_canonical_id": self.state.raw_to_canonical_id.get(
                        int(pending["raw_tracker_id"])
                    ),
                    "discard_reason": (
                        bytetrack_discard_reason_by_raw_idx.get(int(raw_idx))
                        if raw_idx is not None
                        else None
                    ),
                }
            )

        forced_absorption_assignments = []
        for class_name in ("player", "goalkeeper", "referee"):
            for canonical_id, payload in tracks_frame[class_name].items():
                if not bool(payload.get("forced_absorption", False)):
                    continue
                forced_absorption_assignments.append(
                    {
                        "class_name": class_name,
                        "canonical_id": int(canonical_id),
                        "source_raw_tracker_id": payload.get(
                            "forced_absorption_source_raw_tracker_id"
                        ),
                        "raw_tracker_streak_frames": payload.get(
                            "forced_absorption_raw_tracker_streak_frames"
                        ),
                        "canonical_lost_frames": payload.get(
                            "forced_absorption_canonical_lost_frames"
                        ),
                        "mode": payload.get("forced_absorption_mode"),
                        "reference_frame": payload.get(
                            "forced_absorption_reference_frame"
                        ),
                        "distance_sq": payload.get("forced_absorption_distance_sq"),
                    }
                )

        ball_selection_debug = {
            "candidate_count": int(len(ball_candidates)),
            "candidate_count_tracked": int(
                sum(
                    1
                    for candidate in ball_candidates
                    if str(candidate.get("source")) == "tracked"
                )
            ),
            "candidate_count_raw": int(
                sum(
                    1 for candidate in ball_candidates if str(candidate.get("source")) == "raw"
                )
            ),
            "selected": selected_ball is not None,
            "selected_source": selected_ball.get("source") if selected_ball else None,
            "selected_raw_det_idx": (
                selected_ball.get("raw_det_idx") if selected_ball else None
            ),
            "selected_confidence": (
                float(selected_ball.get("confidence", 0.0)) if selected_ball else None
            ),
            "selected_bbox": (selected_ball.get("bbox") if selected_ball else None),
        }

        summary = {
            "tracked_input_count": int(len(sorted_tracked_detections)),
            "canonical_assigned_count": int(len(canonical_ids_in_frame)),
            "pending_count": int(len(pending_detections)),
            "dropped_count": int(dropped_count),
        }

        discarded_yolo_not_tracked = []
        discarded_bytetrack_not_canonical = []
        if collect_visual_debug:
            detection_payload_by_raw_idx = {}
            for det_index, raw_det_idx in enumerate(clean_in.get("det_id", [])):
                try:
                    raw_idx_int = int(raw_det_idx)
                except (TypeError, ValueError):
                    continue
                bbox = clean_in["bbox_xyxy"][det_index] if det_index < len(clean_in["bbox_xyxy"]) else None
                confidence = clean_in["confidence"][det_index] if det_index < len(clean_in["confidence"]) else None
                class_yolo = clean_in["class_name"][det_index] if det_index < len(clean_in["class_name"]) else None
                class_td = clean_in["class_td"][det_index] if det_index < len(clean_in["class_td"]) else None
                team = clean_in["team"][det_index] if det_index < len(clean_in["team"]) else None
                distances = clean_in["distances"][det_index] if det_index < len(clean_in["distances"]) else None
                detection_payload_by_raw_idx[raw_idx_int] = {
                    "raw_det_idx": raw_idx_int,
                    "bbox": self._bbox_to_list(bbox),
                    "confidence": (float(confidence) if confidence is not None else None),
                    "class_name": class_yolo,
                    "class_yolo": class_yolo,
                    "class_td": class_td,
                    "class_relabel": class_td,
                    "team": team,
                    "distances": distances,
                }

            accepted_set = {int(raw_idx) for raw_idx in accepted_raw_detection_indexes}
            bytetrack_set = {int(raw_idx) for raw_idx in bytetrack_raw_detection_indexes}

            for raw_idx in sorted(detection_payload_by_raw_idx.keys()):
                if raw_idx in accepted_set:
                    continue
                entry = dict(detection_payload_by_raw_idx[raw_idx])
                discard_reason = bytetrack_discard_reason_by_raw_idx.get(raw_idx, {})
                if isinstance(discard_reason, dict):
                    reason_pre = discard_reason.get("reason_pre")
                    reason_post = discard_reason.get("reason_post")
                    combined_reason = "|".join(
                        str(token)
                        for token in (reason_pre, reason_post)
                        if token
                    )
                    if combined_reason:
                        entry["discard_reason"] = combined_reason
                    class_tracker = discard_reason.get("class_tracker")
                    if class_tracker:
                        entry["class_tracker"] = class_tracker
                if raw_idx in bytetrack_set:
                    entry["bytetrack_id"] = bytetrack_id_by_raw_idx.get(raw_idx)
                    discarded_bytetrack_not_canonical.append(entry)
                else:
                    not_tracked_reason = bytetrack_not_tracked_reason_by_raw_idx.get(raw_idx, {})
                    if isinstance(not_tracked_reason, dict):
                        for extra_key in (
                            "bytetrack_reason",
                            "bytetrack_stage",
                            "reject_code",
                            "reject_label",
                            "class_tracker",
                        ):
                            extra_value = not_tracked_reason.get(extra_key)
                            if extra_value is not None:
                                entry[extra_key] = extra_value
                    discarded_yolo_not_tracked.append(entry)

        clean_out = {
            "tracks_frame": tracks_frame,
            "summary": summary,
            "canonical_ids_in_frame": canonical_ids_in_frame,
        }

        trace = {
            "pending_assignments_debug": serialize_for_trace(pending_assignments_debug),
            "discard_reason_by_raw_idx": serialize_for_trace(
                bytetrack_discard_reason_by_raw_idx
            ),
            "forced_absorption_debug": serialize_for_trace(
                {
                    "assignments": forced_absorption_assignments,
                    "active_candidates_state": self.state.forced_absorption_state,
                }
            ),
            "canonical_state_debug_snapshot": serialize_for_trace(
                self.state.canonical_state
            ),
            "ball_selection_debug": serialize_for_trace(ball_selection_debug),
            "accepted_raw_detection_indexes": sorted(
                int(raw_idx) for raw_idx in accepted_raw_detection_indexes
            ),
            "bytetrack_raw_detection_indexes": sorted(
                int(raw_idx) for raw_idx in bytetrack_raw_detection_indexes
            ),
            "bytetrack_id_by_raw_idx": serialize_for_trace(bytetrack_id_by_raw_idx),
            "bytetrack_not_tracked_reason_by_raw_idx": serialize_for_trace(
                bytetrack_not_tracked_reason_by_raw_idx
            ),
            "unconfirmed_association_debug": serialize_for_trace(
                bytetrack_packet["trace"].get("unconfirmed_association_debug", [])
            ),
            "discarded_yolo_not_tracked": serialize_for_trace(
                discarded_yolo_not_tracked
            ),
            "discarded_bytetrack_not_canonical": serialize_for_trace(
                discarded_bytetrack_not_canonical
            ),
        }

        return make_phase_packet(
            phase_name=PHASE_CANONICALTRACK,
            frame_index=bytetrack_packet["frame_index"],
            frame_time_ms=bytetrack_packet["frame_time_ms"],
            image_width=bytetrack_packet["image_width"],
            image_height=bytetrack_packet["image_height"],
            clean=clean_out,
            trace=trace,
        )

    def canonicalize_packet(self, bytetrack_packet, collect_visual_debug=False):
        return self.execute(
            bytetrack_packet,
            collect_visual_debug=collect_visual_debug,
        )


__all__ = ["CanonicalTrackPhase"]
