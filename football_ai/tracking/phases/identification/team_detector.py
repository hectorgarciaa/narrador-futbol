from __future__ import annotations

import numpy as np

from football_ai.core import PHASE_IDENTIFICATION, make_phase_packet
from football_ai.evaluation.cluster_visualizer import visualize_shirt_clusters

from .shirt_detector import ShirtDetector
from .team_color_model import TeamColorModel
from .team_detector_utils import (
    CANDIDATE_CLASSES,
    bbox_area,
    extract_shirt_crop,
    field_position_to_tuple,
    goalkeeper_position_gate,
    person_axis_positions,
    referee_position_gate,
    serialize_color,
    serialize_distances,
)


class TeamDetector:
    def __init__(
        self,
        team_color_model_conf=None,
        shirt_detector_conf=None,
    ):
        self.shirt_detector = ShirtDetector(**dict(shirt_detector_conf or {}))
        self.color_model = TeamColorModel(**dict(team_color_model_conf or {}))
        self.candidate_classes = CANDIDATE_CLASSES
        self.n_frame = -1

    @property
    def team_colors(self):
        return self.color_model.team_colors

    @property
    def updated(self):
        return self.color_model.updated

    @property
    def class_samples(self):
        return self.color_model.class_samples

    @property
    def outfield_team_distance_stats(self):
        return self.color_model.outfield_team_distance_stats

    @property
    def referee_distance_stats(self):
        return self.color_model.referee_distance_stats

    def identify_packet(
        self,
        frame_bgr,
        filtering_packet,
        field_width_m,
        sideline_band_distance_m,
        show_plot=False,
        *,
        execution_mode="runtime",
    ):
        filtering_clean = filtering_packet["clean"]
        collect_debug = str(execution_mode).strip().lower() == "debug"
        entries, cluster_events = self._run_detection_pass(
            frame_bgr=frame_bgr,
            bbox_xyxy=filtering_clean["bbox_xyxy"],
            confidence=filtering_clean["confidence"],
            yolo_class_labels=filtering_clean["class_name"],
            field_positions=filtering_clean["field_positions_m"],
            field_width_m=field_width_m,
            sideline_band_distance_m=sideline_band_distance_m,
            show_plot=show_plot,
            collect_debug=collect_debug,
        )
        clean = {
            "num_detections": int(filtering_clean["num_detections"]),
            "det_id": list(filtering_clean["det_id"]),
            "bbox_xyxy": [list(bbox) for bbox in filtering_clean["bbox_xyxy"]],
            "confidence": list(filtering_clean["confidence"]),
            "class_name": list(filtering_clean["class_name"]),
            "field_positions_m": [list(point) for point in filtering_clean["field_positions_m"]],
            "ground_points_image_original": [
                list(point) for point in filtering_clean["ground_points_image_original"]
            ],
            "class_td": [entry["class_td"] for entry in entries],
            "team": [entry["team"] for entry in entries],
            "shirt_color": [entry["shirt_color"] for entry in entries],
            "distances": [entry["distances"] for entry in entries],
            "bbox_size": [entry["bbox_size"] for entry in entries],
        }
        return make_phase_packet(
            phase_name=PHASE_IDENTIFICATION,
            frame_index=filtering_packet["frame_index"],
            frame_time_ms=filtering_packet["frame_time_ms"],
            image_width=filtering_packet["image_width"],
            image_height=filtering_packet["image_height"],
            clean=clean,
            trace=(
                {
                    "detections": [entry["trace"] for entry in entries],
                    "clusters": self.color_model.trace_snapshot(cluster_events),
                    "summary": self._summary(entries),
                }
                if collect_debug
                else {}
            ),
        )

    def _run_detection_pass(
        self,
        frame_bgr,
        bbox_xyxy,
        confidence,
        yolo_class_labels,
        field_positions,
        field_width_m,
        sideline_band_distance_m,
        show_plot,
        collect_debug,
    ):
        self.n_frame += 1
        boxes = np.asarray(bbox_xyxy, dtype=np.float32).reshape(-1, 4)
        confidences = np.asarray(confidence, dtype=np.float32).reshape(-1)
        yolo_class_labels = np.asarray(yolo_class_labels, dtype=object).reshape(-1)

        shirts_for_plot, shirt_colors = self._detect_shirt_colors(
            frame_bgr,
            boxes,
            yolo_class_labels,
            show_plot,
        )
        x_positions = person_axis_positions(field_positions, yolo_class_labels, 0)
        y_positions = person_axis_positions(field_positions, yolo_class_labels, 1)

        entries = []
        cluster_events = []
        for index, class_name in enumerate(yolo_class_labels):
            entry, cluster_event = self._build_detection_entry(
                det_index=index,
                class_name=str(class_name),
                bbox_xyxy=boxes[index],
                confidence=float(confidences[index]),
                field_position=field_positions[index],
                shirt_color=shirt_colors.get(index),
                x_positions=x_positions,
                y_positions=y_positions,
                field_width_m=field_width_m,
                sideline_band_distance_m=sideline_band_distance_m,
                collect_debug=collect_debug,
            )
            entries.append(entry)
            if cluster_event is not None:
                cluster_events.append(cluster_event)

        if show_plot and shirts_for_plot:
            visualize_shirt_clusters(shirts_for_plot)
        return entries, cluster_events

    def _detect_shirt_colors(self, frame_bgr, boxes, class_labels, show_plot):
        shirts_for_plot = []
        candidate_indexes = []
        candidate_shirts = []

        for index, class_name in enumerate(class_labels):
            if str(class_name) not in self.candidate_classes:
                continue
            shirt = extract_shirt_crop(frame_bgr, boxes[index])
            if shirt is None:
                continue
            candidate_indexes.append(index)
            candidate_shirts.append(shirt)
            if show_plot:
                shirts_for_plot.append(shirt)

        if not candidate_shirts:
            return shirts_for_plot, {}

        shirt_colors = self.shirt_detector.get_color_kmeans_batch(candidate_shirts)
        return shirts_for_plot, dict(zip(candidate_indexes, shirt_colors))

    def _build_detection_entry(
        self,
        det_index,
        class_name,
        bbox_xyxy,
        confidence,
        field_position,
        shirt_color,
        x_positions,
        y_positions,
        field_width_m,
        sideline_band_distance_m,
        collect_debug,
    ):
        field_position = field_position_to_tuple(field_position)
        bbox_size = bbox_area(bbox_xyxy)
        serialized_color = serialize_color(shirt_color)
        trace = (
            {
                "det_index": int(det_index),
                "class_name_yolo": class_name,
                "field_position_m": list(field_position) if field_position is not None else None,
                "shirt_crop_available": shirt_color is not None,
                "shirt_color_available": shirt_color is not None,
                "bbox_size": bbox_size,
            }
            if collect_debug
            else None
        )
        if class_name not in self.candidate_classes:
            if trace is not None:
                trace["relabel"] = {
                    "reason": "non_candidate_class",
                    "class_name_input": class_name,
                    "class_td": class_name,
                    "team": None,
                    "distances": None,
                }
            return {
                "class_td": class_name,
                "team": None,
                "shirt_color": None,
                "distances": None,
                "bbox_size": bbox_size,
                "class_name_yolo": class_name,
                "shirt_crop_available": False,
                "shirt_color_available": False,
                "trace": trace,
            }, None

        can_be_ref, is_middle_ref = referee_position_gate(
            x_positions,
            y_positions,
            field_position,
            field_width_m,
            sideline_band_distance_m,
        )
        can_be_goalkeeper = goalkeeper_position_gate(
            x_positions,
            field_position,
            field_width_m,
            sideline_band_distance_m,
        )

        sample_bucket = class_name
        effective_class = class_name
        sample_reason = "initial_class"
        cluster_event = None
        if field_position is not None and shirt_color is not None and class_name in self.color_model.updated:
            if self.color_model.updated[class_name]:
                bucket, effective, sample_reason = self.color_model.decide_sample_class(
                    shirt_color,
                    can_be_goalkeeper=can_be_goalkeeper,
                    can_be_middle_ref=is_middle_ref,
                )
                sample_bucket = bucket or class_name
                effective_class = effective or class_name
            sample_trace, cluster_event = self.color_model.maybe_add_sample(
                sample_bucket,
                shirt_color,
                confidence,
                self.n_frame,
                collect_debug=collect_debug,
            )
            if trace is not None and sample_trace is not None:
                trace["sample_candidate"] = sample_trace

        if trace is not None:
            trace["sample_decision"] = {
                "sample_bucket": sample_bucket,
                "effective_class": effective_class,
                "reason": sample_reason,
            }

        resolved_class, team, distances, relabel_trace = self._reassign_class(
            shirt_color,
            effective_class,
            field_position,
            can_be_ref,
            is_middle_ref,
            can_be_goalkeeper,
        )
        final_class = class_name if field_position is None else resolved_class
        serialized_distances = serialize_distances(distances)
        if trace is not None:
            trace["relabel"] = {
                **relabel_trace,
                "class_name_input": effective_class,
                "class_td": final_class,
                "team": team,
                "distances": serialized_distances,
            }
        return {
            "class_td": final_class,
            "team": team,
            "shirt_color": serialized_color,
            "distances": serialized_distances,
            "bbox_size": bbox_size,
            "class_name_yolo": class_name,
            "shirt_crop_available": shirt_color is not None,
            "shirt_color_available": shirt_color is not None,
            "trace": trace,
        }, cluster_event

    def _reassign_class(
        self,
        shirt_color,
        class_name,
        field_position,
        can_be_ref,
        is_middle_ref,
        can_be_goalkeeper,
    ):
        team, distances = self.color_model.assign_team(shirt_color)
        referee_cluster_match = self.color_model.matches_referee_cluster(distances)

        if team == "referee":
            if not referee_cluster_match:
                return class_name, self.color_model.nearest_outfield_team(distances), distances, self._relabel_trace(
                    "referee_cluster_outlier_fallback",
                )
            if class_name == "referee":
                return "referee", None, distances, self._relabel_trace("referee_confirmed")
            if field_position is None:
                return class_name, self.color_model.nearest_outfield_team(distances), distances, self._relabel_trace(
                    "referee_color_without_field_position",
                )
            if can_be_ref:
                return "referee", None, distances, self._relabel_trace(
                    "relabel_to_referee_by_color_and_position",
                )
            if can_be_goalkeeper:
                return "goalkeeper", None, distances, self._relabel_trace(
                    "relabel_to_goalkeeper_after_referee_color_check",
                )
            return class_name, self.color_model.nearest_outfield_team(distances), distances, self._relabel_trace(
                "fallback_to_nearest_outfield_team",
            )

        if team in self.color_model.team_colors and class_name == "goalkeeper" and (field_position is None or can_be_goalkeeper):
            return "goalkeeper", None, distances, self._relabel_trace("goalkeeper_confirmed")

        if team in self.color_model.team_colors and (
            self.color_model.updated["player"] or len(self.color_model.outfield_team_distance_stats) >= 2
        ):
            if class_name == "referee" and not self.color_model.updated["referee"]:
                return class_name, team, distances, self._relabel_trace(
                    "outfield_cluster_decision_without_referee_cluster",
                )
            if self.color_model.matches_outfield_cluster(distances):
                return "player", team, distances, self._relabel_trace(
                    "relabel_to_player_by_outfield_cluster",
                )
            if field_position is None or can_be_goalkeeper:
                return "goalkeeper", None, distances, self._relabel_trace(
                    "relabel_to_goalkeeper_by_outlier_and_position",
                )

        return class_name, team, distances, self._relabel_trace("keep_current_class")

    @staticmethod
    def _relabel_trace(reason):
        return {"reason": reason}

    def _summary(self, entries):
        return {
            "total_detections": len(entries),
            "candidate_detections": sum(
                1 for entry in entries if entry["class_name_yolo"] in self.candidate_classes
            ),
            "detections_with_shirt_crop": sum(
                1 for entry in entries if entry["shirt_crop_available"]
            ),
            "detections_with_shirt_color": sum(
                1 for entry in entries if entry["shirt_color_available"]
            ),
            "relabelled_detections": sum(
                1
                for entry in entries
                if entry["class_td"] != entry["class_name_yolo"]
            ),
        }
