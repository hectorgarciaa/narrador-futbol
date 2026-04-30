"""
Tracker — orquesta las 6 subfases del pipeline de tracking.

Subfases (en orden de ejecución):
  1. DetectionPhase       → detector_packet
  2. ProjectionPhase      → reference_packet
  3. FilteringPhase       → filtering_packet
  4. IdentificationPhase  → identification_packet
  5. ByteTrackPhase       → bytetrack_packet
  6. CanonicalTrackPhase  → canonical_packet  (salida pública)
"""
from __future__ import annotations

from football_ai.core import make_phase_packet
from football_ai.tracking.phases.bytetrack import ByteTrackPhase
from football_ai.tracking.phases.canonicaltrack import CanonicalTrackPhase
from football_ai.tracking.phases.detection import DetectionPhase
from football_ai.tracking.phases.filtering import FilteringPhase
from football_ai.tracking.phases.identification import IdentificationPhase
from football_ai.tracking.phases.reference_points import PnLCalibFieldProjector, ProjectionPhase

PHASE_TRACKING = "TRACKING"


class Tracker:
    """
    Orquesta las 6 subfases de tracking para un único frame.

    Uso:
        tracker = Tracker(...)
        tracker.reset()
        packet = tracker.process_frame(frame_bgr, frame_index, frame_time_ms)
    """

    def __init__(
        self,
        model_path,
        detector_conf,
        team_detector_conf,
        bytetracker_conf,
        ball_conf,
        tracker_conf,
        projector_conf,
        project_root,
    ):
        # ── subfase 1: detección YOLO ─────────────────────────────────────
        self.detection_phase = DetectionPhase(model_path, detector_conf)

        # ── subfase 2: proyección de campo ───────────────────────────────
        field_projector = None
        if projector_conf["enabled"]:
            field_projector = PnLCalibFieldProjector(
                project_root=project_root,
                **projector_conf["constructor"],
            )
        # Guardamos referencia explícita para métricas/runtime introspection.
        self.field_projector = field_projector
        self.projection_phase = ProjectionPhase(field_projector)

        # ── subfase 3: filtrado post-proyección ──────────────────────────
        self.filtering_phase = FilteringPhase()

        # ── subfase 4: identificación de equipos/clases ──────────────────
        referee_field_width_m = float(
            projector_conf.get("constructor", {}).get("field_width_m", 68.0)
        )
        referee_sideline_band_distance_m = float(
            tracker_conf.get("referee_sideline_band_distance_m", 3.0)
        )
        self.identification_phase = IdentificationPhase(
            team_detector_conf,
            referee_field_width_m=referee_field_width_m,
            referee_sideline_band_distance_m=referee_sideline_band_distance_m,
        )

        # ── subfase 5: ByteTrack ─────────────────────────────────────────
        bytetracker_runtime_conf = dict(bytetracker_conf or {})
        bytetracker_runtime_conf.pop("max_tracks_per_class", None)
        self.bytetrack_phase = ByteTrackPhase(**bytetracker_runtime_conf)

        # ── subfase 6: canonización ──────────────────────────────────────
        self.canonical_phase = CanonicalTrackPhase(
            max_tracks_per_class=bytetracker_conf["max_tracks_per_class"],
            tracker_conf=tracker_conf,
            projector_conf=projector_conf,
            ball_conf=ball_conf,
        )

    # ──────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────

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
        bytetrack_unconfirmed_association_debug = trace_bytetrack.get(
            "unconfirmed_association_debug", []
        )
        
        trace_canonical = canonical_packet.get("trace", {})
        discarded = trace_canonical.get("discarded_detections", [])
        discarded_not_tracked = trace_canonical.get("discarded_yolo_not_tracked", [])
        discarded_tracked_no_canonical = trace_canonical.get("discarded_bytetrack_not_canonical", [])
        
        return {
            "frame_num": int(frame_index),
            "raw_detections": raw_detections,
            "discarded_detections": discarded,
            "discarded_yolo_not_tracked": discarded_not_tracked,
            "discarded_bytetrack_not_canonical": discarded_tracked_no_canonical,
            "bytetrack_unconfirmed_association": bytetrack_unconfirmed_association_debug,
        }

    def process_frame(
        self,
        frame_bgr,
        frame_index: int,
        frame_time_ms: float,
        *,
        show_kmeans: bool = False,
        collect_visual_debug: bool = False,
    ) -> dict:
        """
        Ejecuta las 6 subfases sobre un único frame y devuelve el
        canonical_packet listo para ser consumido por fases posteriores.

        Retorna un packet con el esquema estándar:
            {
                "phase_name":    "TRACKING",
                "frame_index":   int,
                "frame_time_ms": float,
                "image_width":   int,
                "image_height":  int,
                "clean": {
                    "tracks_frame":            dict,   # {class: {id: payload}}
                    "canonical_ids_in_frame":  list,
                    "summary":                 dict,
                },
                "trace": { ... },   # datos de depuración internos
            }
        """
        # 1. Detección
        detector_packet, detector_ms = self.detection_phase.process(
            frame_bgr,
            frame_index=frame_index,
            frame_time_ms=frame_time_ms,
        )

        # 2. Proyección
        reference_packet, proj_ms = self.projection_phase.process(
            frame_bgr,
            detector_packet,
        )

        # 3. Filtrado
        filtering_packet, filter_ms = self.filtering_phase.process(
            reference_packet,
            active_track_boxes_xyxy=self.bytetrack_phase.active_track_boxes_xyxy(),
            geometry=self.projection_phase.geometry,
        )

        # 4. Identificación de equipos / clases
        identification_packet, id_ms = self.identification_phase.process(
            frame_bgr,
            filtering_packet,
            show_kmeans=show_kmeans,
        )

        # 5. ByteTrack
        bytetrack_packet, byte_ms = self.bytetrack_phase.process(
            identification_packet,
            collect_visual_debug=collect_visual_debug,
        )

        # 6. Canonización → salida pública
        canonical_packet, canon_ms = self.canonical_phase.process(
            bytetrack_packet,
            collect_visual_debug=collect_visual_debug,
        )

        canonical_packet.setdefault("trace", {})["profile_ms"] = {
            "detector_ms": detector_ms,
            "proj_ms": proj_ms,
            "filter_ms": filter_ms,
            "id_ms": id_ms,
            "byte_ms": byte_ms,
            "canon_ms": canon_ms,
        }

        if collect_visual_debug:
            canonical_packet["trace"]["visual_debug"] = self._phase_build_visual_debug_frame(
                frame_index, detector_packet, bytetrack_packet, canonical_packet
            )

        # Re-etiquetamos la fase para que los consumidores externos vean
        # "TRACKING" en lugar de "CANONICALTRACK".
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
