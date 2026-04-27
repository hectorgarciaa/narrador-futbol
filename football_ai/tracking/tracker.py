from time import perf_counter

import cv2
import numpy as np

from football_ai.bytetrack import ByteTrackPhase
from football_ai.canonicaltrack import CanonicalTrackPhase
from football_ai.detection import Detector
from football_ai.filtering import filter_reference_points
from football_ai.identification import TeamDetector
from football_ai.posession import PosessionPhase
from football_ai.reference_points import (
    PnLCalibFieldProjector,
    build_reference_points_packet_without_homography,
)


class Tracker:
    @staticmethod
    def _measure_phase_execution(fn, *args, **kwargs):
        start = perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (perf_counter() - start) * 1000.0
        return result, elapsed_ms

    @staticmethod
    def _print_frame_phase_profile(phase_rows):
        total_ms = sum(float(row[1]) for row in phase_rows)
        for phase_index, row in enumerate(phase_rows, start=1):
            phase_title, phase_ms = row[0], row[1]
            print(f"fase {phase_index}: {phase_title}: {phase_ms:.2f} ms", flush=True)
            subphase_rows = row[2] if len(row) > 2 else None
            if subphase_rows:
                for subphase_title, subphase_ms in subphase_rows:
                    print(
                        f"  subfase: {subphase_title}: {subphase_ms:.2f} ms",
                        flush=True,
                    )
        print(f"FRAME: TIEMPO TOTAL: {total_ms:.2f} ms", flush=True)
        print("#########################################", flush=True)

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
        print("*" * 100, model_path)

        self.model = Detector(model_path, **detector_conf)
        self.team_detector = TeamDetector(**team_detector_conf)

        bytetracker_runtime_conf = dict(bytetracker_conf or {})
        bytetracker_runtime_conf.pop("max_tracks_per_class", None)
        self.bytetrack_phase = ByteTrackPhase(**bytetracker_runtime_conf)
        self.canonical_phase = CanonicalTrackPhase(
            max_tracks_per_class=bytetracker_conf["max_tracks_per_class"],
            tracker_conf=tracker_conf,
            projector_conf=projector_conf,
            ball_conf=ball_conf,
        )

        self.max_tracks_per_class = dict(bytetracker_conf["max_tracks_per_class"])

        self.referee_field_width_m = float(
            projector_conf.get("constructor", {}).get("field_width_m", 68.0)
        )
        self.referee_sideline_band_distance_m = float(
            tracker_conf.get("referee_sideline_band_distance_m", 3.0)
        )

        self.field_projector = None
        if projector_conf["enabled"]:
            self.field_projector = PnLCalibFieldProjector(
                project_root=project_root,
                **projector_conf["constructor"],
            )

        self.visualization_debug_frames = []
        self.posession_phase = PosessionPhase(tracker_conf.get("possession"))

    def _phase_prepare_frame_inputs(self, frame_bgr, detector_packet, collect_visual_debug):
        subphase_rows = []

        t0 = perf_counter()
        raw_detections = self._get_raw_detections(
            detector_packet["trace"],
            collect_visual_debug,
        )
        subphase_rows.append(("Extraer detecciones RAW", (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        reference_packet = (
            self.field_projector.project_frame(frame_bgr, detector_packet)
            if self.field_projector is not None
            else build_reference_points_packet_without_homography(detector_packet)
        )
        subphase_rows.append(("Proyección de campo (PnLCalib)", (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        filtering_packet = filter_reference_points(
            reference_packet,
            active_track_boxes_xyxy=self.bytetrack_phase.active_track_boxes_xyxy(),
            geometry=getattr(self.field_projector, "geometry", None)
        )
        filtered_det_ids = {
            int(det_id)
            for det_id in np.asarray(
                filtering_packet["clean"]["det_id"],
                dtype=np.int32,
            ).reshape(-1)
        }
        if raw_detections:
            raw_detections = [
                det
                for det in raw_detections
                if int(det["raw_det_idx"]) in filtered_det_ids
            ]
        subphase_rows.append(
            ("Filtrar detecciones fuera del campo proyectado", (perf_counter() - t0) * 1000.0)
        )

        return {
            "payload": (raw_detections, filtering_packet),
            "subphase_rows": subphase_rows,
        }

    def _phase_assign_team_and_enrich_metadata(
        self,
        frame_bgr,
        raw_detections,
        filtering_packet,
        show_kmeans,
    ):
        subphase_rows = []

        t0 = perf_counter()
        identification_packet = self.team_detector.identify_packet(
            frame_bgr,
            filtering_packet,
            self.referee_field_width_m,
            self.referee_sideline_band_distance_m,
            show_kmeans,
        )
        identification_clean = identification_packet["clean"]
        subphase_rows.append(("TeamDetector.detect_teams", (perf_counter() - t0) * 1000.0))

        t0 = perf_counter()
        if raw_detections:
            yolo_class_labels = list(identification_clean["class_name"])
            class_labels = list(identification_clean["class_name_td"])
            for raw_idx, raw_detection in enumerate(raw_detections):
                raw_detection["class_yolo"] = (
                    str(yolo_class_labels[raw_idx])
                    if raw_idx < len(yolo_class_labels)
                    else raw_detection.get("class_yolo")
                )
                class_relabel_value = (
                    str(class_labels[raw_idx]) if raw_idx < len(class_labels) else None
                )
                raw_detection["class_name_td"] = class_relabel_value
                raw_detection["class_relabel"] = class_relabel_value
                raw_detection["class_team_detector"] = class_relabel_value
                if raw_idx < int(identification_clean["num_detections"]):
                    raw_detection["referee_reassign_gate"] = identification_clean[
                        "referee_reassign_gate"
                    ][raw_idx]
                    raw_detection["goalkeeper_reassign_gate"] = identification_clean[
                        "goalkeeper_reassign_gate"
                    ][raw_idx]
        subphase_rows.append(("Construir arrays de labels", (perf_counter() - t0) * 1000.0))

        return {
            "payload": identification_packet,
            "subphase_rows": subphase_rows,
        }

    @staticmethod
    def _phase_build_visual_debug_frame(
        frame_num,
        raw_detections,
        accepted_raw_detection_indexes,
        bytetrack_raw_detection_indexes,
        bytetrack_id_by_raw_idx,
        bytetrack_discard_reason_by_raw_idx,
        bytetrack_not_tracked_reason_by_raw_idx,
        bytetrack_unconfirmed_association_debug,
        visual_debug_frames,
    ):
        discarded = [
            raw_detection
            for raw_detection in raw_detections
            if int(raw_detection["raw_det_idx"]) not in accepted_raw_detection_indexes
        ]

        discarded_not_tracked = []
        discarded_tracked_no_canonical = []
        for det in discarded:
            raw_idx = int(det["raw_det_idx"])
            if raw_idx in bytetrack_raw_detection_indexes:
                payload = dict(det)
                payload["bytetrack_id"] = bytetrack_id_by_raw_idx.get(raw_idx)
                reason_info = bytetrack_discard_reason_by_raw_idx.get(raw_idx, {})
                payload["class_tracker"] = reason_info.get("class_tracker")
                reason_pre = reason_info.get("reason_pre")
                reason_post = reason_info.get("reason_post")
                if reason_pre and reason_post:
                    payload["discard_reason"] = f"{reason_pre}|{reason_post}"
                elif reason_post:
                    payload["discard_reason"] = str(reason_post)
                elif reason_pre:
                    payload["discard_reason"] = str(reason_pre)
                discarded_tracked_no_canonical.append(payload)
            else:
                payload = dict(det)
                reason_info = bytetrack_not_tracked_reason_by_raw_idx.get(raw_idx, {})
                payload["bytetrack_reason"] = reason_info.get("reason")
                payload["bytetrack_stage"] = reason_info.get("stage")
                payload["class_tracker"] = reason_info.get("class_name")
                payload["bytetrack_id"] = reason_info.get("tracker_id")
                discarded_not_tracked.append(payload)

        visual_debug_frames.append(
            {
                "frame_num": int(frame_num),
                "raw_detections": raw_detections,
                "discarded_detections": discarded,
                "discarded_yolo_not_tracked": discarded_not_tracked,
                "discarded_bytetrack_not_canonical": discarded_tracked_no_canonical,
                "bytetrack_unconfirmed_association": bytetrack_unconfirmed_association_debug or [],
            }
        )

    def get_tracks(
        self,
        video,
        show_kmeans=False,
        frame_hook=None,
        collect_visual_debug=False,
        profile_phases=False,
    ):
        self.bytetrack_phase.reset()
        self.canonical_phase.reset()
        self.posession_phase.reset()

        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": [], "possession": []}
        visual_debug_frames = [] if collect_visual_debug else None

        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval_ms = 1000.0 / float(fps if fps > 0 else 25.0)
        n_frame = 0

        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frame_phase_rows = []
                detector_packet = self.model.predict_frame(
                    frame_bgr,
                    frame_index=n_frame,
                    frame_time_ms=n_frame * frame_interval_ms,
                )

                prepare_phase_result, elapsed_ms = self._measure_phase_execution(
                    self._phase_prepare_frame_inputs,
                    frame_bgr,
                    detector_packet,
                    collect_visual_debug,
                )
                raw_detections, filtering_packet = prepare_phase_result["payload"]
                frame_phase_rows.append(
                    (
                        "Preparar detecciones y proyección",
                        elapsed_ms,
                        prepare_phase_result.get("subphase_rows"),
                    )
                )

                assign_phase_result, elapsed_ms = self._measure_phase_execution(
                    self._phase_assign_team_and_enrich_metadata,
                    frame_bgr,
                    raw_detections,
                    filtering_packet,
                    show_kmeans,
                )
                identification_packet = assign_phase_result["payload"]
                frame_phase_rows.append(
                    (
                        "Asignar equipo y enriquecer metadata",
                        elapsed_ms,
                        assign_phase_result.get("subphase_rows"),
                    )
                )

                bytetrack_packet, elapsed_ms = self._measure_phase_execution(
                    self.bytetrack_phase.track_packet,
                    identification_packet,
                    collect_visual_debug=collect_visual_debug,
                )
                frame_phase_rows.append(("Asociación ByteTrack", elapsed_ms))

                canonical_packet, elapsed_ms = self._measure_phase_execution(
                    self.canonical_phase.canonicalize_packet,
                    bytetrack_packet,
                    collect_visual_debug=collect_visual_debug,
                )
                frame_phase_rows.append(("Canonización (CanonicalTrack)", elapsed_ms))

                posession_packet, elapsed_ms = self._measure_phase_execution(
                    self.posession_phase.process_packet,
                    canonical_packet,
                )
                frame_phase_rows.append(("Posesión (Posession)", elapsed_ms))

                tracks_frame = posession_packet["clean"]["tracks_frame"]
                tracks["player"].append(dict(tracks_frame["player"]))
                tracks["goalkeeper"].append(dict(tracks_frame["goalkeeper"]))
                tracks["referee"].append(dict(tracks_frame["referee"]))
                tracks["ball"].append(dict(tracks_frame["ball"]))
                tracks["possession"].append(dict(posession_packet["clean"]["possession"]))

                if collect_visual_debug:
                    canonical_trace = canonical_packet["trace"]
                    _, elapsed_ms = self._measure_phase_execution(
                        self._phase_build_visual_debug_frame,
                        n_frame,
                        raw_detections,
                        {int(raw_idx) for raw_idx in canonical_trace["accepted_raw_detection_indexes"]},
                        {int(raw_idx) for raw_idx in canonical_trace["bytetrack_raw_detection_indexes"]},
                        {
                            int(raw_idx): tracker_id
                            for raw_idx, tracker_id in canonical_trace["bytetrack_id_by_raw_idx"].items()
                        },
                        {
                            int(raw_idx): reason
                            for raw_idx, reason in canonical_trace["discard_reason_by_raw_idx"].items()
                        },
                        {
                            int(raw_idx): reason
                            for raw_idx, reason in canonical_trace["bytetrack_not_tracked_reason_by_raw_idx"].items()
                        },
                        canonical_trace["unconfirmed_association_debug"],
                        visual_debug_frames,
                    )
                    frame_phase_rows.append(("Construir debug visual del frame", elapsed_ms))

                hook_elapsed_ms = 0.0
                if callable(frame_hook):
                    t0 = perf_counter()
                    frame_hook(tracks, n_frame)
                    hook_elapsed_ms = (perf_counter() - t0) * 1000.0
                frame_phase_rows.append(("Ejecutar frame hook", hook_elapsed_ms))

                if profile_phases:
                    self._print_frame_phase_profile(frame_phase_rows)
                n_frame += 1
        finally:
            capture.release()

        self.visualization_debug_frames = visual_debug_frames if collect_visual_debug else []
        return tracks

    @staticmethod
    def _get_raw_detections(detector_trace, collect_visual_debug):
        if not collect_visual_debug:
            return []

        return [
            {
                "raw_det_idx": int(trace_item["det_id"]),
                "class_yolo": trace_item["class_name"],
                "bbox": list(trace_item["bbox_xyxy"]),
                "confidence": float(trace_item["confidence"]),
            }
            for trace_item in detector_trace["detections"]
        ]
