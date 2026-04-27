from time import perf_counter

import cv2
import numpy as np

from football_ai.bytetrack import ByteTrackPhase
from football_ai.canonicaltrack import CanonicalTrackPhase
from football_ai.detection import DetectionPhase
from football_ai.filtering import FilteringPhase
from football_ai.identification import IdentificationPhase
from football_ai.positions import PositionInferingPhase
from football_ai.posession import PosessionPhase
from football_ai.reference_points import (
    PnLCalibFieldProjector,
    ProjectionPhase,
)


class Tracker:
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
        *,
        config=None,
        video_path=None,
        logger=None,
        expected_roles_by_team_override=None,
        lineup_matcher=None,
    ):
        print("*" * 100, model_path)

        bytetracker_runtime_conf = dict(bytetracker_conf or {})
        bytetracker_runtime_conf.pop("max_tracks_per_class", None)

        self.detection_phase = DetectionPhase(model_path, detector_conf)
        self.model = self.detection_phase.detector

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
        self.projection_phase = ProjectionPhase(self.field_projector)
        self.filtering_phase = FilteringPhase()
        self.identification_phase = IdentificationPhase(
            team_detector_conf,
            referee_field_width_m=self.referee_field_width_m,
            referee_sideline_band_distance_m=self.referee_sideline_band_distance_m,
        )
        self.team_detector = self.identification_phase.team_detector

        self.bytetrack_phase = ByteTrackPhase(**bytetracker_runtime_conf)
        self.canonical_phase = CanonicalTrackPhase(
            max_tracks_per_class=bytetracker_conf["max_tracks_per_class"],
            tracker_conf=tracker_conf,
            projector_conf=projector_conf,
            ball_conf=ball_conf,
        )

        self.max_tracks_per_class = dict(bytetracker_conf["max_tracks_per_class"])

        self.visualization_debug_frames = []
        self.posession_phase = PosessionPhase(tracker_conf.get("possession"))
        self.position_infering_phase = (
            PositionInferingPhase(
                config,
                video_path,
                logger,
                expected_roles_by_team_override=expected_roles_by_team_override,
                lineup_matcher=lineup_matcher,
            )
            if config is not None and video_path is not None and logger is not None
            else None
        )

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

    @staticmethod
    def _filter_raw_detections_by_packet(raw_detections, filtering_packet):
        if not raw_detections:
            return []
        filtered_det_ids = {
            int(det_id)
            for det_id in np.asarray(
                filtering_packet["clean"]["det_id"],
                dtype=np.int32,
            ).reshape(-1)
        }
        return [
            det
            for det in raw_detections
            if int(det["raw_det_idx"]) in filtered_det_ids
        ]

    @staticmethod
    def _enrich_raw_detections_with_identification(
        raw_detections,
        identification_packet,
    ):
        if not raw_detections:
            return
        identification_clean = identification_packet["clean"]
        yolo_class_labels = list(identification_clean["class_name"])
        class_labels = list(identification_clean["class_name_td"])
        num_detections = int(identification_clean["num_detections"])
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
            if raw_idx < num_detections:
                raw_detection["referee_reassign_gate"] = identification_clean[
                    "referee_reassign_gate"
                ][raw_idx]
                raw_detection["goalkeeper_reassign_gate"] = identification_clean[
                    "goalkeeper_reassign_gate"
                ][raw_idx]

    @staticmethod
    def _merge_parallel_phase_packets(
        canonical_packet,
        posession_packet,
        position_packet,
    ):
        canonical_tracks_frame = canonical_packet["clean"]["tracks_frame"]
        posession_tracks_frame = posession_packet["clean"]["tracks_frame"]
        position_tracks_frame = (
            position_packet["clean"]["tracks_frame"] if position_packet is not None else {}
        )

        merged_tracks_frame = {}
        for class_name in ("player", "goalkeeper", "referee", "ball"):
            merged_tracks_frame[class_name] = {}
            track_ids = set(canonical_tracks_frame.get(class_name, {}).keys())
            track_ids.update(posession_tracks_frame.get(class_name, {}).keys())
            track_ids.update(position_tracks_frame.get(class_name, {}).keys())
            for track_id in track_ids:
                merged_payload = {}
                for source in (
                    canonical_tracks_frame.get(class_name, {}),
                    posession_tracks_frame.get(class_name, {}),
                    position_tracks_frame.get(class_name, {}),
                ):
                    payload = source.get(track_id)
                    if isinstance(payload, dict):
                        merged_payload.update(payload)
                merged_tracks_frame[class_name][track_id] = merged_payload

        clean_out = dict(canonical_packet["clean"])
        clean_out["tracks_frame"] = merged_tracks_frame
        clean_out["possession"] = dict(posession_packet["clean"]["possession"])

        summary = dict(clean_out.get("summary", {}))
        posession_summary = dict(posession_packet["clean"].get("summary", {}))
        summary.update(posession_summary)
        if position_packet is not None:
            position_summary = dict(position_packet["clean"].get("position_infering", {}))
            if position_summary:
                summary["position_infering_frame_predictions"] = int(
                    position_summary.get("frame_predictions", 0)
                )
                summary["position_infering_player_predictions"] = int(
                    position_summary.get("player_predictions", 0)
                )
            clean_out["position_infering"] = position_summary
        clean_out["summary"] = summary

        trace_out = dict(canonical_packet["trace"])
        trace_out["possession"] = dict(posession_packet["trace"].get("possession", {}))
        if position_packet is not None:
            trace_out["position_infering"] = dict(
                position_packet["trace"].get("position_infering", {})
            )

        return {
            "phase_name": "TRACKING_OUTPUT",
            "frame_index": canonical_packet["frame_index"],
            "frame_time_ms": canonical_packet["frame_time_ms"],
            "image_width": canonical_packet["image_width"],
            "image_height": canonical_packet["image_height"],
            "clean": clean_out,
            "trace": trace_out,
        }

    def get_tracks(
        self,
        video,
        show_kmeans=False,
        frame_hook=None,
        collect_visual_debug=False,
        profile_phases=False,
    ):
        if self.position_infering_phase is not None:
            self.position_infering_phase.reset()
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
                detector_packet, elapsed_ms = self.detection_phase.process(
                    frame_bgr,
                    frame_index=n_frame,
                    frame_time_ms=n_frame * frame_interval_ms,
                )
                frame_phase_rows.append(("Detección (YOLO)", elapsed_ms))

                raw_detections = self._get_raw_detections(
                    detector_packet["trace"],
                    collect_visual_debug,
                )

                reference_packet, elapsed_ms = self.projection_phase.process(
                    frame_bgr,
                    detector_packet,
                )
                frame_phase_rows.append(
                    ("Proyección de campo (PnLCalib)", elapsed_ms)
                )

                filtering_packet, elapsed_ms = self.filtering_phase.process(
                    reference_packet,
                    active_track_boxes_xyxy=self.bytetrack_phase.active_track_boxes_xyxy(),
                    geometry=self.projection_phase.geometry,
                )
                frame_phase_rows.append(
                    ("Filtrar detecciones fuera del campo proyectado", elapsed_ms)
                )
                raw_detections = self._filter_raw_detections_by_packet(
                    raw_detections,
                    filtering_packet,
                )

                identification_packet, elapsed_ms = self.identification_phase.process(
                    frame_bgr,
                    filtering_packet,
                    show_kmeans=show_kmeans,
                )
                frame_phase_rows.append(("Asignar equipo e identificar clase", elapsed_ms))
                self._enrich_raw_detections_with_identification(
                    raw_detections,
                    identification_packet,
                )

                bytetrack_packet, elapsed_ms = self.bytetrack_phase.process(
                    identification_packet,
                    collect_visual_debug=collect_visual_debug,
                )
                frame_phase_rows.append(("Asociación ByteTrack", elapsed_ms))

                canonical_packet, elapsed_ms = self.canonical_phase.process(
                    bytetrack_packet,
                    collect_visual_debug=collect_visual_debug,
                )
                frame_phase_rows.append(("Canonización (CanonicalTrack)", elapsed_ms))

                posession_packet, elapsed_ms = self.posession_phase.process(
                    canonical_packet,
                )
                frame_phase_rows.append(("Posesión (Posession)", elapsed_ms))

                position_packet = None
                if self.position_infering_phase is not None:
                    position_packet, elapsed_ms = self.position_infering_phase.process(
                        canonical_packet,
                    )
                    frame_phase_rows.append(
                        ("Posiciones (PositionInfering)", elapsed_ms)
                    )

                tracking_packet = self._merge_parallel_phase_packets(
                    canonical_packet,
                    posession_packet,
                    position_packet,
                )

                tracks_frame = tracking_packet["clean"]["tracks_frame"]
                tracks["player"].append(dict(tracks_frame["player"]))
                tracks["goalkeeper"].append(dict(tracks_frame["goalkeeper"]))
                tracks["referee"].append(dict(tracks_frame["referee"]))
                tracks["ball"].append(dict(tracks_frame["ball"]))
                tracks["possession"].append(dict(tracking_packet["clean"]["possession"]))

                if collect_visual_debug:
                    canonical_trace = tracking_packet["trace"]
                    self._phase_build_visual_debug_frame(
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

                if callable(frame_hook):
                    frame_hook(tracks, n_frame)

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
