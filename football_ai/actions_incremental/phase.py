from __future__ import annotations

from football_ai.core import PHASE_ACTIONS_DETECTOR, Phase, make_phase_packet

from .runtime import ActionsRuntime, ActionsRuntimeConfig


class ActionsDetectorPhase(Phase):
    """Phase wrapper around the incremental actions runtime."""

    def __init__(self, runtime_config: ActionsRuntimeConfig | None = None):
        self.runtime = ActionsRuntime(runtime_config)

    def reset(self) -> None:
        self.runtime.reset()

    def execute(self, position_packet):
        clean_in = dict(position_packet["clean"])
        trace_in = dict(position_packet["trace"])

        runtime_result = self.runtime.process_frame(
            int(position_packet["frame_index"]),
            clean_in,
        )

        clean_out = dict(clean_in)
        clean_out["actions_incremental"] = {
            "raw_edge": runtime_result.raw_edge,
            "raw_edge_batch": runtime_result.raw_edge_batch,
            "confirmed_action": runtime_result.confirmed_action,
            "action_metadata": runtime_result.action_metadata,
        }
        trace_out = dict(trace_in)
        trace_out["actions_incremental"] = {
            "track_count_by_class": dict(runtime_result.summary.track_count_by_class),
            "person_slot_assignments": dict(runtime_result.summary.person_slot_assignments),
            "referee_slot_assignments": dict(runtime_result.summary.referee_slot_assignments),
            "synthetic_home_slots": list(runtime_result.summary.synthetic_home_slots),
            "synthetic_away_slots": list(runtime_result.summary.synthetic_away_slots),
            "synthetic_referee_slots": list(runtime_result.summary.synthetic_referee_slots),
            "frames": int(runtime_result.summary.frames),
        }

        return make_phase_packet(
            phase_name=PHASE_ACTIONS_DETECTOR,
            frame_index=position_packet["frame_index"],
            frame_time_ms=position_packet["frame_time_ms"],
            image_width=position_packet["image_width"],
            image_height=position_packet["image_height"],
            clean=clean_out,
            trace=trace_out,
        )
