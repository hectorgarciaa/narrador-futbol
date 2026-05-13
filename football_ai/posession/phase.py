from __future__ import annotations

from typing import Any

from football_ai.core import PHASE_POSESSION, Phase, make_phase_packet

from .estimator import PossessionConfig, TeamPossessionEstimator


class PosessionPhase(Phase):
    def __init__(self, config_mapping: dict[str, Any] | None = None):
        self.config = PossessionConfig.from_mapping(config_mapping)
        self.reset()

    def reset(self):
        self.estimator = TeamPossessionEstimator(self.config)

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
        return cls._normalize_track_identifier(left_track_id) == cls._normalize_track_identifier(
            right_track_id
        )

    def _annotate_tracks_frame(self, tracks_frame, possession_info):
        owning_team_id = possession_info["team_id"]
        owning_player_id = possession_info["player_id"]
        for class_name in ("player", "goalkeeper", "referee", "ball"):
            for track_id, payload in tracks_frame[class_name].items():
                is_owner = class_name in {"player", "goalkeeper"} and self._track_id_matches(
                    track_id, owning_player_id
                )
                payload["is_possession_player"] = bool(is_owner)
                payload["ball_owning_team_id"] = owning_team_id
                payload["ball_owning_player_id"] = owning_player_id
                payload["player_id"] = owning_player_id
                payload["possession_reason"] = possession_info["reason"]
                payload["possession_ball_detected"] = bool(possession_info["ball_detected"])
                payload["possession_nearest_track_id"] = possession_info["nearest_track_id"]
                payload["possession_nearest_team_id"] = possession_info["nearest_team_id"]

    def execute(self, canonical_packet, *, execution_mode="runtime"):
        collect_debug = str(execution_mode).strip().lower() == "debug"
        tracks_frame_in = canonical_packet["clean"]["tracks_frame"]
        tracks_frame = {
            "player": {
                track_id: dict(payload) for track_id, payload in tracks_frame_in["player"].items()
            },
            "goalkeeper": {
                track_id: dict(payload)
                for track_id, payload in tracks_frame_in["goalkeeper"].items()
            },
            "referee": {
                track_id: dict(payload) for track_id, payload in tracks_frame_in["referee"].items()
            },
            "ball": {track_id: dict(payload) for track_id, payload in tracks_frame_in["ball"].items()},
        }

        frame_tracks_for_possession = {
            "player": tracks_frame["player"],
            "goalkeeper": tracks_frame["goalkeeper"],
            "referee": tracks_frame["referee"],
            "ball": tracks_frame["ball"],
        }
        possession_info = self.estimator.update_frame(
            int(canonical_packet["frame_index"]),
            frame_tracks_for_possession,
        )
        self._annotate_tracks_frame(tracks_frame, possession_info)

        clean_out = {
            "tracks_frame": tracks_frame,
            "possession": dict(possession_info),
        }

        trace_out = {}
        if collect_debug:
            trace_out = dict(canonical_packet["trace"])
            trace_out["possession"] = dict(possession_info)

        return make_phase_packet(
            phase_name=PHASE_POSESSION,
            frame_index=canonical_packet["frame_index"],
            frame_time_ms=canonical_packet["frame_time_ms"],
            image_width=canonical_packet["image_width"],
            image_height=canonical_packet["image_height"],
            clean=clean_out,
            trace=trace_out,
        )

    def process_packet(self, canonical_packet, *, execution_mode="runtime"):
        return self.execute(canonical_packet, execution_mode=execution_mode)


__all__ = ["PosessionPhase"]
