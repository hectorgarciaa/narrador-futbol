from __future__ import annotations

from football_ai.core import PHASE_POSITION_INFERING, Phase, make_phase_packet

from .assigner import OnlineSpecialSeedRoleAssigner


class PositionInferingPhase(Phase):
    def __init__(
        self,
        config,
        video_path,
        logger,
        *,
        expected_roles_by_team_override=None,
        lineup_matcher=None,
    ):
        self.assigner = OnlineSpecialSeedRoleAssigner(
            config,
            video_path,
            logger,
            expected_roles_by_team_override=expected_roles_by_team_override,
            lineup_matcher=lineup_matcher,
        )

    def __getattr__(self, name):
        return getattr(self.assigner, name)

    def reset(self):
        self.assigner.reset()

    def execute(self, canonical_packet):
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
            "ball": {
                track_id: dict(payload) for track_id, payload in tracks_frame_in["ball"].items()
            },
        }

        frame_summary = self.assigner.process_tracks_frame(
            tracks_frame,
            int(canonical_packet["frame_index"]),
        )

        clean_out = dict(canonical_packet["clean"])
        clean_out["tracks_frame"] = tracks_frame
        clean_out["position_infering"] = dict(frame_summary)

        trace_out = dict(canonical_packet["trace"])
        trace_out["position_infering"] = {
            "frame_summary": dict(frame_summary),
            "stats": self.assigner.summary(),
        }

        return make_phase_packet(
            phase_name=PHASE_POSITION_INFERING,
            frame_index=canonical_packet["frame_index"],
            frame_time_ms=canonical_packet["frame_time_ms"],
            image_width=canonical_packet["image_width"],
            image_height=canonical_packet["image_height"],
            clean=clean_out,
            trace=trace_out,
        )
