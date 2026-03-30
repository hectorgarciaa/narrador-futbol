import numpy as np
import supervision as sv

from football_ai.core import get_config
from football_ai.detection import Detector
from football_ai.identification import TeamDetector
from football_ai.reference_points import PnLCalibFieldProjector
from football_ai.tracking import ByteTrack

config = get_config()
tracking_cfg = config.tracking
TRACKER_CONF = {
    "track_thresh": tracking_cfg['track_thresh'],
    "track_buffer": tracking_cfg['track_buffer'],
    "match_thresh": tracking_cfg['match_thresh'],
    "frame_rate": tracking_cfg['frame_rate'],
    "minimum_consecutive_frames": tracking_cfg['minimum_consecutive_frames'],
    "max_total_tracks": tracking_cfg["max_total_tracks"],
    "enforce_internal_class_limits": tracking_cfg["enforce_internal_class_limits"],
    "team_mismatch_penalty": tracking_cfg["team_mismatch_penalty"],
    "second_match_threshold": tracking_cfg["second_match_threshold"],
    "unconfirmed_match_threshold": tracking_cfg["unconfirmed_match_threshold"],
    
    "reassign_motion_factor": tracking_cfg["reassign_motion_factor"],
    "reassign_min_distance": tracking_cfg["reassign_min_distance"],
    "reassign_min_samples": tracking_cfg["reassign_min_samples"],
    "reassign_motion_growth_cap_frames": tracking_cfg["reassign_motion_growth_cap_frames"],
    
    "use_field_position_as_primary_cost": tracking_cfg["use_field_position_as_primary_cost"],
    "use_bbox_center_for_matching": tracking_cfg["use_bbox_center_for_matching"],
    "bbox_center_distance_weight": tracking_cfg["bbox_center_distance_weight"],
    "bbox_center_distance_gate_px": tracking_cfg["bbox_center_distance_gate_px"],
    "lost_time_penalty_weight": tracking_cfg["lost_time_penalty_weight"],
    "lost_time_penalty_max_frames": tracking_cfg["lost_time_penalty_max_frames"],
    "strict_person_class_separation": tracking_cfg["strict_person_class_separation"],
    "reserve_penalty_spot_seed_players": tracking_cfg["reserve_penalty_spot_seed_players"],
    "reserve_penalty_spot_seed_match_distance_m": tracking_cfg["reserve_penalty_spot_seed_match_distance_m"],
    "require_field_position_for_reassign": tracking_cfg["require_field_position_for_reassign"],
    
    "max_reassign_lost_frames": tracking_cfg["max_reassign_lost_frames"],
    "max_reassign_lost_frames_by_class": tracking_cfg["max_reassign_lost_frames_by_class"],
    
    "motion_std_gate_enabled": tracking_cfg["motion_std_gate_enabled"],
    "motion_std_factor": tracking_cfg["motion_std_factor"],
    "motion_std_min_samples": tracking_cfg["motion_std_min_samples"],
    "motion_std_floor": tracking_cfg["motion_std_floor"],
    
    "ball_expected_position_gate_px": tracking_cfg["ball_expected_position_gate_px"],
    "ball_expected_position_gate_growth_per_frame": tracking_cfg["ball_expected_position_gate_growth_per_frame"],
    "ball_expected_position_confidence_relax": tracking_cfg["ball_expected_position_confidence_relax"],
    "ball_size_ratio_per_frame": tracking_cfg["ball_size_ratio_per_frame"],
    "ball_size_min_samples": tracking_cfg["ball_size_min_samples"],
    "ball_size_std_factor": tracking_cfg["ball_size_std_factor"],
    "ball_size_std_floor": tracking_cfg["ball_size_std_floor"],
    "ball_max_reassign_lost_frames": tracking_cfg["ball_max_reassign_lost_frames"],
    "ball_high_conf_override": tracking_cfg["ball_high_conf_override"],
    
    "referee_recovery_max_lost_frames": tracking_cfg["referee_recovery_max_lost_frames"],
    "referee_recovery_max_distance": tracking_cfg["referee_recovery_max_distance"],
}



plncalib_cfg = config.pnlcalib
PLNCALIB_CONF = {
    "field_length_m": plncalib_cfg["field_length_m"],
    "field_width_m": plncalib_cfg["field_width_m"],
    "max_width": plncalib_cfg["max_width"],
    "bottom_offset_ratio": plncalib_cfg["bottom_offset_ratio"],
    "keypoint_threshold": plncalib_cfg["keypoint_threshold"],
    "line_threshold": plncalib_cfg["line_threshold"],
    "pnl_refine": plncalib_cfg["pnl_refine"],
    "temporal_blend": plncalib_cfg["temporal_blend"],
    "pixels_per_meter": plncalib_cfg["pixels_per_meter"],
    "device": plncalib_cfg["device"],
}

FIELD_TRACKING_CONF = {
    "enabled": tracking_cfg.get("use_field_positions", True),
    "method": tracking_cfg.get("field_position_method", "pnlcalib"),
    "classes": tracking_cfg.get("field_position_classes", ["player", "goalkeeper"]),
    
    "field_length_m": tracking_cfg.get("field_length_m", 106.0),
    "field_width_m": tracking_cfg.get("field_width_m", 68.0),
    "max_width": tracking_cfg.get("field_position_max_width", 1280),
    "bottom_offset_ratio": tracking_cfg.get("field_position_bottom_offset_ratio", 0.04),
    "keypoint_threshold": tracking_cfg.get("field_position_keypoint_threshold", 0.3434),
    "line_threshold": tracking_cfg.get("field_position_line_threshold", 0.7867),
    "pnl_refine": tracking_cfg.get("field_position_pnl_refine", True),
    "temporal_blend": tracking_cfg.get("field_position_temporal_blend", 0.20),
    "pixels_per_meter": tracking_cfg.get("field_position_pixels_per_meter", 8),
    "device": tracking_cfg.get("field_position_device"),
    
    "match_distance_gate_m": tracking_cfg.get("field_position_match_distance_gate_m", 8.0),
    "match_distance_max_lost_frames": tracking_cfg.get("field_position_match_distance_max_lost_frames"),
    "match_distance_cap_m": tracking_cfg.get("field_position_match_distance_cap_m", 6.0),
    "match_distance_growth_mode": tracking_cfg.get("field_position_match_distance_growth_mode", "power"),
    "match_distance_lost_exponent": tracking_cfg.get("field_position_match_distance_lost_exponent", 1.0),
    "match_distance_decay_per_frame": tracking_cfg.get("field_position_match_distance_decay_per_frame", 0.0),
    "match_distance_weight": tracking_cfg.get("field_position_match_distance_weight", 0.25),
    "reassign_min_field_distance_m": tracking_cfg.get("reassign_min_field_distance_m", 4.0),
}

class Tracker:
    def __init__(self, model_path, conf, team_colors, tracker_conf,
                 project_root=None, max_tracks_per_class=None,
                 field_tracking_conf=None, plncalib_conf=None):
        self.model = Detector(model_path, conf)
        self.team_detector = TeamDetector(True, team_colors) #, min_samples=60, min_size_cluster=4, candidate_classes=["player"])
        self.field_projector = PnLCalibFieldProjector(
                project_root=project_root,
                field_length_m=float(plncalib_conf["field_length_m"]),
                field_width_m=float(plncalib_conf["field_width_m"]),
                max_width=int(plncalib_conf["max_width"]),
                bottom_offset_ratio=float(plncalib_conf["bottom_offset_ratio"]),
                keypoint_threshold=float(plncalib_conf["keypoint_threshold"]),
                line_threshold=float(plncalib_conf["line_threshold"]),
                pnl_refine=bool(plncalib_conf["pnl_refine"]),
                temporal_blend=float(plncalib_conf["temporal_blend"]),
                pixels_per_meter=int(plncalib_conf["pixels_per_meter"]),
                device=plncalib_conf["device"],
            )
        
        self.enforce_internal_class_limits = bool(
            tracker_conf.get("enforce_internal_class_limits", False)
        )
        
        internal_class_limits = (
            max_tracks_per_class if self.enforce_internal_class_limits else None
        )

        self.tracker = ByteTrack(
            tracker_conf["track_thresh"],
            tracker_conf["track_buffer"],
            tracker_conf["match_thresh"],
            tracker_conf["frame_rate"],
            tracker_conf["minimum_consecutive_frames"],
            max_tracks_per_class=internal_class_limits,
            team_mismatch_penalty=tracker_conf.get("team_mismatch_penalty", 1000.0),
            second_match_threshold=tracker_conf.get("second_match_threshold", 0.7),
            unconfirmed_match_threshold=tracker_conf.get("unconfirmed_match_threshold", 0.8),
            use_field_positions=field_tracking_conf.get("enabled", False),
            field_position_classes=field_tracking_conf.get("classes", ["player", "goalkeeper"]),
            field_distance_gate_m=field_tracking_conf.get("match_distance_gate_m", 8.0),
            field_distance_weight=field_tracking_conf.get("match_distance_weight", 0.25),
            field_distance_gate_max_lost_frames=field_tracking_conf.get("match_distance_max_lost_frames"),
            field_distance_gate_cap_m=field_tracking_conf.get("match_distance_cap_m"),
            field_distance_growth_mode=field_tracking_conf.get("match_distance_growth_mode", "power"),
            field_distance_lost_exponent=field_tracking_conf.get("match_distance_lost_exponent", 1.0),
            field_distance_decay_per_frame=field_tracking_conf.get("match_distance_decay_per_frame", 0.0),
            lost_time_penalty_weight=tracker_conf.get("lost_time_penalty_weight", 0.0),
            lost_time_penalty_max_frames=tracker_conf.get("lost_time_penalty_max_frames", 10),
            use_field_position_as_primary_cost=tracker_conf.get("use_field_position_as_primary_cost", False),
            use_bbox_center_for_matching=tracker_conf.get("use_bbox_center_for_matching", True),
            bbox_center_distance_weight=tracker_conf.get("bbox_center_distance_weight", 0.7),
            bbox_center_distance_gate_px=tracker_conf.get("bbox_center_distance_gate_px", 120.0),
        )

    def get_tracks(self, video, show_kmeans=False):
        model_detections = self.model.detect(video)
        tracks = {"player": [], "goalkeeper": [], "referee": [], "ball": [] }

        for n_frame, detections in enumerate(model_detections):
            detections_sv = sv.Detections.from_ultralytics(detections)
            teams_of_detected_objects = self.team_detector.detect_teams(detections, show_plot=show_kmeans)
            teams_labels = [o["team"] for o in teams_of_detected_objects]
            class_labels = [o["class"] for o in teams_of_detected_objects]
            field_projection = self.field_projector.project_detections(detections.orig_img, detections_sv.xyxy, class_names=class_labels)
            field_positions = field_projection.field_positions_m
            ground_points_projected = field_projection.ground_points_image_projected

            detections_sv.data["team"] = np.array([dicc["team"] for dicc in teams_of_detected_objects], dtype=object)
            detections_sv.data["distances"] = np.array([dicc["distances"] for dicc in teams_of_detected_objects], dtype=object)
            detections_sv.data["shirt_color"] = np.array([dicc["shirt_color"] for dicc in teams_of_detected_objects], dtype=object)
            detections_sv.data["field_position"] = np.asarray(field_positions, dtype=np.float32)
            detections_sv.data["ground_point_image"] = np.asarray(ground_points_projected, dtype=np.float32)

            tracks_detection = self.tracker.update_with_detections(
                detections_sv,
                teams_labels,
                class_labels,
            )

            for key in tracks.keys():
                tracks[key].append({})




