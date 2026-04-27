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
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def filter_reference_points(
    reference_packet,
    *,
    active_track_boxes_xyxy=None,
    sideline_margin_m=0.75,
    geometry=None,
    field_length_m=None,
    field_width_m=None,
):
    clean_in = reference_packet["clean"]
    num_detections = int(clean_in["num_detections"])
    homography_valid = bool(clean_in["homography_valid"])
    field_positions_usable = bool(clean_in["field_positions_usable_for_tracking"])

    if geometry is None:
        geometry = PitchGeometry(
            field_length_m=106.0 if field_length_m is None else float(field_length_m),
            field_width_m=68.0 if field_width_m is None else float(field_width_m),
        )
    field_length_m = float(geometry.field_length_m if field_length_m is None else field_length_m)
    field_width_m = float(geometry.field_width_m if field_width_m is None else field_width_m)

    ground_points = np.asarray(
        clean_in["ground_points_image_original"],
        dtype=np.float32,
    ).reshape(-1, 2)
    homography = np.asarray(
        clean_in["homography_image_to_field_3x3"],
        dtype=np.float64,
    )
    projected_positions = project_image_points(ground_points, homography).astype(np.float32)
    finite_mask = np.all(np.isfinite(projected_positions), axis=1)
    inside_mask = np.zeros(num_detections, dtype=bool)

    if homography_valid and field_positions_usable and np.any(finite_mask):
        candidates = projected_positions[finite_mask]
        inside_pitch = points_inside_field_mask(candidates, geometry=geometry, margin_m=0.0)
        inside_sideline_band = (
            (candidates[:, 0] >= 0.0)
            & (candidates[:, 0] <= field_length_m)
            & (candidates[:, 1] >= -float(sideline_margin_m))
            & (candidates[:, 1] <= field_width_m + float(sideline_margin_m))
        )
        inside_mask[finite_mask] = np.logical_or(inside_pitch, inside_sideline_band)

    active_track_boxes = []
    for box in active_track_boxes_xyxy or []:
        box = np.asarray(box, dtype=np.float32).reshape(-1)
        if box.size >= 4:
            active_track_boxes.append(box[:4])

    kept_indices = []
    accepted_trace = []
    rejected_trace = []
    rescued_count = 0

    for index in range(num_detections):
        max_iou = max(
            (_tlbr_iou(clean_in["bbox_xyxy"][index], track_box) for track_box in active_track_boxes),
            default=0.0,
        )
        is_finite = bool(finite_mask[index])
        inside_field = bool(inside_mask[index])
        rescued_by_iou = False

        if not homography_valid or not field_positions_usable:
            keep = True
            reject_code = REJECT_CODE_HOMOGRAPHY_NOT_USABLE
        elif not is_finite:
            keep = False
            reject_code = REJECT_CODE_INVALID_FIELD_POSITION
        elif inside_field:
            keep = True
            reject_code = REJECT_CODE_KEPT
        elif max_iou > 0.0:
            keep = True
            reject_code = REJECT_CODE_RESCUED_BY_TRACK_OVERLAP
            rescued_by_iou = True
            rescued_count += 1
        else:
            keep = False
            reject_code = REJECT_CODE_OUTSIDE_FIELD

        trace_item = {
            "det_id": clean_in["det_id"][index],
            "bbox_xyxy": list(clean_in["bbox_xyxy"][index]),
            "confidence": float(clean_in["confidence"][index]),
            "class_name": str(clean_in["class_name"][index]),
            "field_position_m": list(clean_in["field_positions_m"][index]),
            "keep": bool(keep),
            "reject_code": int(reject_code),
            "reject_label": REJECT_CODE_LABELS[reject_code],
            "is_finite_field_position": is_finite,
            "inside_field": inside_field,
            "rescued_by_track_overlap": rescued_by_iou,
            "max_iou_with_active_tracks": float(max_iou),
        }
        if keep:
            kept_indices.append(index)
            accepted_trace.append(trace_item)
        else:
            rejected_trace.append(trace_item)

    clean_out = {
        key: [value[index] for index in kept_indices]
        if isinstance(value, list) and len(value) == num_detections
        else value
        for key, value in clean_in.items()
    }
    clean_out["num_detections"] = len(kept_indices)

    return make_phase_packet(
        phase_name=PHASE_FILTERING,
        frame_index=reference_packet["frame_index"],
        frame_time_ms=reference_packet["frame_time_ms"],
        image_width=reference_packet["image_width"],
        image_height=reference_packet["image_height"],
        clean=clean_out,
        trace={
            "accepted_detections": accepted_trace,
            "rejected_detections": rejected_trace,
            "summary": {
                "total_before_filter": num_detections,
                "total_kept": len(kept_indices),
                "total_rejected": len(rejected_trace),
                "total_rescued_by_iou": rescued_count,
                "homography_valid": homography_valid,
                "field_positions_usable_for_tracking": field_positions_usable,
            },
        },
    )


__all__ = ["filter_reference_points"]
