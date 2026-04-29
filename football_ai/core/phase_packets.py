PHASE_DETECTOR = "DETECTOR"
PHASE_REFERENCE_POINTS = "REFERENCE_POINTS"
PHASE_PROJECTION = PHASE_REFERENCE_POINTS
PHASE_FILTERING = "FILTERING"
PHASE_IDENTIFICATION = "IDENTIFICATION"
PHASE_BYTETRACK = "BYTETRACK"
PHASE_CANONICALTRACK = "CANONICALTRACK"
PHASE_POSESSION = "POSESSION"
PHASE_POSITION_INFERING = "POSITION_INFERING"
PHASE_ACTIONS_DETECTOR = "ACTIONS_DETECTOR"

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

def make_phase_packet(
    *,
    phase_name,
    frame_index,
    frame_time_ms,
    image_width,
    image_height,
    clean,
    trace,
    elapsed_ms=None,
):
    return {
        "phase_name": phase_name,
        "frame_index": int(frame_index),
        "frame_time_ms": float(frame_time_ms),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "clean": clean,
        "trace": trace,
        "elapsed_ms": None if elapsed_ms is None else float(elapsed_ms),
    }

__all__ = [
    "PHASE_DETECTOR",
    "PHASE_REFERENCE_POINTS",
    "PHASE_PROJECTION",
    "PHASE_FILTERING",
    "PHASE_IDENTIFICATION",
    "PHASE_BYTETRACK",
    "PHASE_CANONICALTRACK",
    "PHASE_POSESSION",
    "PHASE_POSITION_INFERING",
    "PHASE_ACTIONS_DETECTOR",
    "IDENTITY_HOMOGRAPHY_3X3",
    "default_render_color_bgr",
    "make_phase_packet",
]
