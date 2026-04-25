import numpy as np

from football_ai.reference_points.pnlcalib_runtime import points_inside_field_mask


def _invalidate_unusable_field_positions(
    field_positions,
    field_positions_usable_for_tracking=True,
):
    positions = np.asarray(field_positions, dtype=np.float32).reshape(-1, 2).copy()
    if not bool(field_positions_usable_for_tracking) and positions.size > 0:
        positions[:] = np.nan
    return positions


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


def filter_post_homography(
    field_positions,
    *,
    has_homography,
    field_positions_usable_for_tracking=True,
    geometry=None,
    field_length_m=None,
    field_width_m=None,
    sideline_margin_m=0.75,
    detection_boxes=None,
    active_track_boxes=None,
):
    positions = _invalidate_unusable_field_positions(
        field_positions,
        field_positions_usable_for_tracking=field_positions_usable_for_tracking,
    )
    if len(positions) == 0:
        return np.zeros(0, dtype=bool), positions
    if not bool(has_homography):
        return np.ones(len(positions), dtype=bool), positions

    finite_mask = np.all(np.isfinite(positions), axis=1)
    inside_mask = np.zeros(len(positions), dtype=bool)
    if np.any(finite_mask):
        candidate_points = positions[finite_mask]
        base_inside_mask = points_inside_field_mask(
            candidate_points,
            geometry=geometry,
            margin_m=0.0,
        )
        current_field_length = (
            float(field_length_m)
            if field_length_m is not None
            else float(getattr(geometry, "field_length_m", 106.0))
        )
        current_field_width = (
            float(field_width_m)
            if field_width_m is not None
            else float(getattr(geometry, "field_width_m", 68.0))
        )
        x_coords = candidate_points[:, 0]
        y_coords = candidate_points[:, 1]
        sideline_margin_mask = (
            (x_coords >= 0.0)
            & (x_coords <= current_field_length)
            & (y_coords >= -float(sideline_margin_m))
            & (y_coords <= current_field_width + float(sideline_margin_m))
        )
        inside_mask[finite_mask] = np.logical_or(base_inside_mask, sideline_margin_mask)

    keep_mask = np.logical_or(~finite_mask, inside_mask)
    if np.all(keep_mask) or detection_boxes is None or not active_track_boxes:
        return keep_mask, positions

    detection_boxes = np.asarray(detection_boxes, dtype=np.float32).reshape(-1, 4)
    overlap_keep_mask = np.zeros(len(detection_boxes), dtype=bool)
    for det_idx, det_box in enumerate(detection_boxes):
        if keep_mask[det_idx]:
            continue
        for track_box in active_track_boxes:
            if _tlbr_iou(det_box, track_box) > 0.0:
                overlap_keep_mask[det_idx] = True
                break
    return np.logical_or(keep_mask, overlap_keep_mask), positions
