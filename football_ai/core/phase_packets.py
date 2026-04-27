PHASE_DETECTOR = "DETECTOR"
PHASE_REFERENCE_POINTS = "REFERENCE_POINTS"
PHASE_FILTERING = "FILTERING"

REJECT_CODE_KEPT = 0
REJECT_CODE_OUTSIDE_FIELD = 1
REJECT_CODE_HOMOGRAPHY_NOT_USABLE = 2
REJECT_CODE_INVALID_FIELD_POSITION = 3
REJECT_CODE_RESCUED_BY_TRACK_OVERLAP = 4

REJECT_CODE_LABELS = {
    REJECT_CODE_KEPT: "kept",
    REJECT_CODE_OUTSIDE_FIELD: "outside_field",
    REJECT_CODE_HOMOGRAPHY_NOT_USABLE: "homography_not_usable",
    REJECT_CODE_INVALID_FIELD_POSITION: "invalid_field_position",
    REJECT_CODE_RESCUED_BY_TRACK_OVERLAP: "rescued_by_track_overlap",
}

DETECTOR_RENDER_COLORS_BGR = {
    "player": [0, 255, 0],
    "referee": [255, 0, 0],
    "ball": [0, 165, 255],
    "goalkeeper": [0, 255, 255],
}

IDENTITY_HOMOGRAPHY_3X3 = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

def default_render_color_bgr(class_name):
    return list(DETECTOR_RENDER_COLORS_BGR.get(str(class_name), [255, 255, 255]))

def make_phase_packet(*, phase_name, frame_index, frame_time_ms, image_width, image_height, clean, trace):
    return {
        "phase_name": phase_name,
        "frame_index": int(frame_index),
        "frame_time_ms": float(frame_time_ms),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "clean": clean,
        "trace": trace,
    }

__all__ = [
    "PHASE_DETECTOR",
    "PHASE_REFERENCE_POINTS",
    "PHASE_FILTERING",
    "REJECT_CODE_KEPT",
    "REJECT_CODE_OUTSIDE_FIELD",
    "REJECT_CODE_HOMOGRAPHY_NOT_USABLE",
    "REJECT_CODE_INVALID_FIELD_POSITION",
    "REJECT_CODE_RESCUED_BY_TRACK_OVERLAP",
    "REJECT_CODE_LABELS",
    "IDENTITY_HOMOGRAPHY_3X3",
    "default_render_color_bgr",
    "make_phase_packet",
]
