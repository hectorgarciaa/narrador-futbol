from __future__ import annotations

from ultralytics import YOLO

from .utils import normalize_detection_class_name
from football_ai.core import (
    PHASE_DETECTOR,
    default_render_color_bgr,
    make_phase_packet,
)


class Detector:
    def __init__(self, model_path, conf=0.01, verbose=False):
        self.model = YOLO(model_path)
        self.conf = conf
        self.verbose = verbose

    def predict_frame(
        self,
        frame_bgr,
        frame_index=0,
        frame_time_ms=0.0,
        *,
        execution_mode="runtime",
    ):
        collect_debug = str(execution_mode).strip().lower() == "debug"
        result = self.model.predict(
            frame_bgr,
            stream=False,
            conf=self.conf,
            verbose=self.verbose,
        )[0]

        names = result.names
        boxes = result.boxes

        bbox_xyxy = []
        confidence = []
        class_name = []
        trace_detections = []
        total_raw = 0

        if boxes is not None and len(boxes) > 0:
            raw_xyxy = boxes.xyxy.cpu().numpy()
            raw_conf = boxes.conf.cpu().numpy()
            raw_cls = boxes.cls.cpu().numpy().astype(int)
            total_raw = len(raw_cls)
            for bbox, score, cid in zip(raw_xyxy, raw_conf, raw_cls):
                raw_name = str(names.get(int(cid), cid))
                normalized_name = normalize_detection_class_name(raw_name)
                if not normalized_name:
                    continue
                det_id = len(bbox_xyxy)
                color = default_render_color_bgr(normalized_name)
                serialized_bbox = [float(value) for value in bbox.tolist()[:4]]

                bbox_xyxy.append(serialized_bbox)
                confidence.append(float(score))
                class_name.append(normalized_name)
                if collect_debug:
                    trace_detections.append(
                        {
                            "det_id": int(det_id),
                            "bbox_xyxy": serialized_bbox,
                            "confidence": float(score),
                            "class_id": int(cid),
                            "class_name": normalized_name,
                            "class_name_raw": raw_name,
                            "render_color_bgr": color[:3],
                        }
                    )

        clean = {
            "num_detections": len(bbox_xyxy),
            "det_id": list(range(len(bbox_xyxy))),
            "bbox_xyxy": bbox_xyxy,
            "confidence": confidence,
            "class_name": class_name,
        }
        trace = (
            {
                "detections": trace_detections,
                "summary": {
                    "total_raw": total_raw,
                    "total_supported": len(bbox_xyxy),
                    "total_discarded": total_raw - len(bbox_xyxy),
                },
            }
            if collect_debug
            else {}
        )
        return make_phase_packet(
            phase_name=PHASE_DETECTOR,
            frame_index=frame_index,
            frame_time_ms=frame_time_ms,
            image_width=int(frame_bgr.shape[1]),
            image_height=int(frame_bgr.shape[0]),
            clean=clean,
            trace=trace,
        )


__all__ = ["Detector"]
