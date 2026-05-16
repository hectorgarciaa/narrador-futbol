"""
Tracker - orquesta las 6 subfases del pipeline de tracking.

Subfases (en orden de ejecucion):
  1. DetectionPhase       -> detector_packet
  2. ProjectionPhase      -> reference_packet
  3. FilteringPhase       -> filtering_packet
  4. IdentificationPhase  -> identification_packet
  5. ByteTrackPhase       -> bytetrack_packet
  6. CanonicalTrackPhase  -> canonical_packet  (salida publica)
"""
from __future__ import annotations

from football_ai.core import make_phase_packet
from football_ai.tracking.phases.bytetrack import ByteTrackPhase
from football_ai.tracking.phases.canonicaltrack import CanonicalTrackPhase
from football_ai.tracking.phases.detection import DetectionPhase
from football_ai.tracking.phases.filtering import FilteringPhase
from football_ai.tracking.phases.identification import IdentificationPhase
from football_ai.tracking.phases.reference_points import ProjectionPhase

PHASE_TRACKING = "TRACKING"


def _normalize_execution_mode(raw_mode):
    mode = str(raw_mode or "runtime").strip().lower()
    return "debug" if mode == "debug" else "runtime"


class Tracker:
    """
    Orquesta las 6 subfases de tracking para un unico frame.

    Uso:
        tracker = Tracker(...)
        tracker.reset()
        packet = tracker.process_frame(frame_bgr, frame_index, frame_time_ms)
    """

    def __init__(self, tracking_runtime_conf):
        tracking_runtime_conf = dict(tracking_runtime_conf or {})
        model_path = tracking_runtime_conf["model_path"]
        detector_conf = tracking_runtime_conf.get("detector_conf", {})
        team_detector_conf = tracking_runtime_conf.get("team_detector_conf", {})
        bytetracker_conf = tracking_runtime_conf.get("bytetracker_conf", {})
        tracker_conf = tracking_runtime_conf.get("tracker_conf", {})
        canonical_conf = tracking_runtime_conf.get("canonical_conf", {})
        positions_conf = tracking_runtime_conf.get("positions_conf", {})
        projector_conf = tracking_runtime_conf.get("projector_conf", {})
        project_root = tracking_runtime_conf["project_root"]
        detector_runtime_conf = dict(detector_conf or {})
        detector_runtime_conf["model_path"] = model_path
        bytetracker_runtime_conf = dict(bytetracker_conf or {})

        projector_runtime_conf = dict(projector_conf or {})
        projector_runtime_conf["project_root"] = project_root
        projector_constructor_conf = dict(
            projector_runtime_conf.get("constructor", {}) or {}
        )

        referee_field_width_m = float(
            projector_constructor_conf.get("field_width_m", 68.0)
        )
        referee_sideline_band_distance_m = float(
            (canonical_conf or {}).get("referee_sideline_band_distance_m", 3.0)
        )

        identification_runtime_conf = {
            "team_detector_conf": dict(team_detector_conf or {}),
            "referee_field_width_m": referee_field_width_m,
            "referee_sideline_band_distance_m": referee_sideline_band_distance_m,
        }

        canonical_runtime_conf = dict(canonical_conf or {})
        canonical_runtime_conf["use_field_positions"] = bool(
            projector_runtime_conf.get("enabled", False)
        )
        canonical_runtime_conf["referee_field_width_m"] = referee_field_width_m
        canonical_runtime_conf["special_seed_canonical_ids"] = list(
            dict(positions_conf or {}).get("special_seed_canonical_ids", [1, 2])
        )
        canonical_runtime_conf["field_geometry"] = {
            "field_length_m": float(
                projector_constructor_conf.get("field_length_m", 106.0)
            ),
            "field_width_m": referee_field_width_m,
            "center_y_m": float(
                projector_constructor_conf.get(
                    "center_y_m",
                    referee_field_width_m / 2.0,
                )
            ),
            "penalty_mark_distance_m": float(
                projector_constructor_conf.get("penalty_mark_distance_m", 11.0)
            ),
        }

        self.detection_phase = DetectionPhase(detector_runtime_conf)
        self.projection_phase = ProjectionPhase(projector_runtime_conf)
        self.field_projector = self.projection_phase.projector
        self.filtering_phase = FilteringPhase()
        self.identification_phase = IdentificationPhase(identification_runtime_conf)
        self.execution_mode = _normalize_execution_mode(
            tracker_conf.get("execution_mode", "runtime")
        )
        self.bytetrack_phase = ByteTrackPhase(bytetracker_runtime_conf)
        self.canonical_phase = CanonicalTrackPhase(canonical_runtime_conf)

    def reset(self) -> None:
        """Reinicia el estado interno de las subfases con estado propio."""
        self.bytetrack_phase.reset()
        self.canonical_phase.reset()

    def _phase_build_visual_debug_frame(
        self, frame_index, detector_packet, bytetrack_packet, canonical_packet
    ):
        trace_detector = detector_packet.get("trace", {})
        raw_detections = trace_detector.get("raw_detections", [])
        if not isinstance(raw_detections, list):
            raw_detections = trace_detector.get("detections", [])

        trace_bytetrack = bytetrack_packet.get("trace", {})
        bytetrack_matching_debug = trace_bytetrack.get("matching_debug", {})

        trace_canonical = canonical_packet.get("trace", {})
        discarded = trace_canonical.get("discarded_detections", [])
        discarded_not_tracked = trace_canonical.get("discarded_yolo_not_tracked", [])
        discarded_tracked_no_canonical = trace_canonical.get(
            "discarded_bytetrack_not_canonical", []
        )

        return {
            "frame_num": int(frame_index),
            "raw_detections": raw_detections,
            "discarded_detections": discarded,
            "discarded_yolo_not_tracked": discarded_not_tracked,
            "discarded_bytetrack_not_canonical": discarded_tracked_no_canonical,
            "bytetrack_matching_debug": bytetrack_matching_debug,
        }

    def process_frame(
        self,
        frame_bgr,
        frame_index: int,
        frame_time_ms: float,
        *,
        show_kmeans: bool = False,
        collect_visual_debug: bool = False,
        execution_mode: str | None = None,
    ) -> dict:
        """
        Ejecuta las 6 subfases sobre un unico frame y devuelve el
        canonical_packet listo para ser consumido por fases posteriores.

        Retorna un packet con el esquema estandar:
            {
                "phase_name":    "TRACKING",
                "frame_index":   int,
                "frame_time_ms": float,
                "image_width":   int,
                "image_height":  int,
                "clean": {
                    "tracks_frame":            dict,   # {class: {id: payload}}
                },
                "trace": { ... },   # datos de depuracion internos
            }
        """
        effective_execution_mode = _normalize_execution_mode(
            execution_mode if execution_mode is not None else self.execution_mode
        )

        detector_packet, detector_ms = self.detection_phase.process(
            frame_bgr,
            frame_index=frame_index,
            frame_time_ms=frame_time_ms,
            execution_mode=effective_execution_mode,
        )

        reference_packet, proj_ms = self.projection_phase.process(
            frame_bgr,
            detector_packet,
            execution_mode=effective_execution_mode,
        )

        filtering_packet, filter_ms = self.filtering_phase.process(
            reference_packet,
            active_track_boxes_xyxy=self.bytetrack_phase.active_track_boxes_xyxy(),
            geometry=self.projection_phase.geometry,
            execution_mode=effective_execution_mode,
        )

        identification_packet, id_ms = self.identification_phase.process(
            frame_bgr,
            filtering_packet,
            show_kmeans=show_kmeans,
            execution_mode=effective_execution_mode,
        )

        bytetrack_packet, byte_ms = self.bytetrack_phase.process(
            identification_packet,
            execution_mode=effective_execution_mode,
        )

        canonical_packet, canon_ms = self.canonical_phase.process(
            bytetrack_packet,
            execution_mode=effective_execution_mode,
        )

        canonical_packet.setdefault("trace", {})["profile_ms"] = {
            "detector_ms": detector_ms,
            "proj_ms": proj_ms,
            "filter_ms": filter_ms,
            "id_ms": id_ms,
            "byte_ms": byte_ms,
            "canon_ms": canon_ms,
        }

        if collect_visual_debug and effective_execution_mode == "debug":
            canonical_packet["trace"][
                "visual_debug"
            ] = self._phase_build_visual_debug_frame(
                frame_index, detector_packet, bytetrack_packet, canonical_packet
            )

        return make_phase_packet(
            phase_name=PHASE_TRACKING,
            frame_index=canonical_packet["frame_index"],
            frame_time_ms=canonical_packet["frame_time_ms"],
            image_width=canonical_packet["image_width"],
            image_height=canonical_packet["image_height"],
            clean=canonical_packet["clean"],
            trace=canonical_packet["trace"],
        )


__all__ = ["Tracker", "PHASE_TRACKING"]
