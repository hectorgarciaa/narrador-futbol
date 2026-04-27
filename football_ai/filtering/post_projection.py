from __future__ import annotations

import numpy as np

from football_ai.core import (
    PHASE_FILTERING,
    REJECT_CODE_HOMOGRAPHY_NOT_USABLE,
    REJECT_CODE_INVALID_FIELD_POSITION,
    REJECT_CODE_KEPT,
    REJECT_CODE_LABELS,
    REJECT_CODE_OUTSIDE_FIELD,
    REJECT_CODE_RESCUED_BY_TRACK_OVERLAP,
    make_phase_packet,
)
from football_ai.reference_points.common import PitchGeometry
from football_ai.reference_points.geometry import project_image_points, points_inside_field_mask


def _tlbr_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    box_a = np.asarray(box_a, dtype=np.float32).reshape(-1)
    box_b = np.asarray(box_b, dtype=np.float32).reshape(-1)
    if box_a.size < 4 or box_b.size < 4:
        return 0.0
    if not np.all(np.isfinite(box_a[:4])) or not np.all(np.isfinite(box_b[:4])):
        return 0.0
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def filter_reference_points(
    reference_packet,
    *,
    active_track_boxes_xyxy=None,
    sideline_margin_m=0.75,
    geometry=None,
    field_length_m=None,
    field_width_m=None,
):
    reference_clean = reference_packet["clean"]
    num_detections = reference_clean["num_detections"]
    det_id = reference_clean["det_id"]
    bbox_xyxy = reference_clean["bbox_xyxy"]
    confidence = reference_clean["confidence"]
    class_name = reference_clean["class_name"]
    field_positions_m = reference_clean["field_positions_m"]
    homography = np.asarray(reference_clean["homography_image_to_field_3x3"], dtype=np.float64)
    ground_points_image_original = np.asarray(
        reference_clean["ground_points_image_original"],
        dtype=np.float32,
    ).reshape(-1, 2)
    recomputed_positions = project_image_points(ground_points_image_original, homography).astype(np.float32)
    finite_mask = np.all(np.isfinite(recomputed_positions), axis=1)
    inside_mask = np.zeros(num_detections, dtype=bool)
    active_track_boxes = []
    for box in active_track_boxes_xyxy or []:
        box = np.asarray(box, dtype=np.float32).reshape(-1)
        if box.size >= 4:
            active_track_boxes.append(box[:4])

    homography_valid = reference_clean["homography_valid"]
    field_positions_usable = reference_clean["field_positions_usable_for_tracking"]
    if geometry is None:
        geometry = PitchGeometry(
            field_length_m=106.0 if field_length_m is None else float(field_length_m),
            field_width_m=68.0 if field_width_m is None else float(field_width_m),
        )
    current_field_length = float(field_length_m if field_length_m is not None else geometry.field_length_m)
    current_field_width = float(field_width_m if field_width_m is not None else geometry.field_width_m)

    if homography_valid and field_positions_usable and np.any(finite_mask):
        candidate_points = recomputed_positions[finite_mask]
        base_inside_mask = points_inside_field_mask(
            candidate_points,
            geometry=geometry,
            margin_m=0.0,
        )
        x_coords = candidate_points[:, 0]
        y_coords = candidate_points[:, 1]
        sideline_inside_mask = (
            (x_coords >= 0.0)
            & (x_coords <= current_field_length)
            & (y_coords >= -float(sideline_margin_m))
            & (y_coords <= current_field_width + float(sideline_margin_m))
        )
        inside_mask[finite_mask] = np.logical_or(base_inside_mask, sideline_inside_mask)

    keep_mask = []
    reject_code = []
    trace_detections = []
    rescued_count = 0

    for index in range(num_detections):
        max_iou = 0.0
        det_box = bbox_xyxy[index]
        for track_box in active_track_boxes:
            max_iou = max(max_iou, _tlbr_iou(det_box, track_box))

        is_finite = bool(finite_mask[index])
        inside_field = bool(inside_mask[index])
        rescued_by_track_overlap = False

        if not homography_valid or not field_positions_usable:
            keep = True
            code = REJECT_CODE_HOMOGRAPHY_NOT_USABLE
        elif not is_finite:
            keep = False
            code = REJECT_CODE_INVALID_FIELD_POSITION
        elif inside_field:
            keep = True
            code = REJECT_CODE_KEPT
        elif max_iou > 0.0:
            keep = True
            code = REJECT_CODE_RESCUED_BY_TRACK_OVERLAP
            rescued_by_track_overlap = True
            rescued_count += 1
        else:
            keep = False
            code = REJECT_CODE_OUTSIDE_FIELD

        keep_mask.append(bool(keep))
        reject_code.append(int(code))
        trace_detections.append(
            {
                "det_id": det_id[index],
                "keep": keep,
                "reject_code": code,
                "reject_label": REJECT_CODE_LABELS[code],
                "is_finite_field_position": is_finite,
                "inside_field": inside_field,
                "rescued_by_track_overlap": rescued_by_track_overlap,
                "max_iou_with_active_tracks": float(max_iou),
            }
        )

    clean = {
        "num_detections": num_detections,
        "det_id": det_id,
        "bbox_xyxy": bbox_xyxy,
        "confidence": confidence,
        "class_name": class_name,
        "field_positions_m": field_positions_m,
        "keep_mask": keep_mask,
        "reject_code": reject_code,
        "kept_count": sum(1 for item in keep_mask if item),
        "rejected_count": sum(1 for item in keep_mask if not item),
    }
    trace = {
        "detections": trace_detections,
        "summary": {
            "total_before_filter": num_detections,
            "total_kept": clean["kept_count"],
            "total_rejected": clean["rejected_count"],
            "total_rescued_by_iou": rescued_count,
        },
    }
    return make_phase_packet(
        phase_name=PHASE_FILTERING,
        frame_index=reference_packet["frame_index"],
        frame_time_ms=reference_packet["frame_time_ms"],
        image_width=reference_packet["image_width"],
        image_height=reference_packet["image_height"],
        clean=clean,
        trace=trace,
    )


__all__ = ["filter_reference_points"]
