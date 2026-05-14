# detection

Fase de deteccion de objetos para el pipeline de tracking, basada en modelos YOLO de Ultralytics. Esta carpeta contiene el wrapper minimo usado por la fase y un adaptador de fase que invoca ese wrapper.

## Clases soportadas

Se normalizan clases a estas cuatro etiquetas canonicas y todo lo demas se descarta:

| Clase | Descripcion |
|---|---|
| `player` | Jugador de campo |
| `goalkeeper` | Portero |
| `referee` | Arbitro |
| `ball` | Balon |

La normalizacion acepta alias comunes (`person`, `people`, `gk`, `ref`, `sports ball`, etc.) y los mapea a las etiquetas anteriores.

---

## `detector.py` — `Detector`

Wrapper minimo sobre `ultralytics.YOLO`.

**Constructor**
- `Detector(model_path, conf=0.01, verbose=False)`

**Metodo principal**
- `predict_frame(frame_bgr, frame_index=0, frame_time_ms=0.0, *, execution_mode="runtime")`

`frame_bgr` debe ser un `np.ndarray` con forma `(H, W, 3)` en BGR (formato OpenCV). En `execution_mode="debug"` se incluye `trace` con detalle por deteccion.

**Salida (`PhaseFramePacket`)**
- `clean`:
  - `num_detections`: `int`
  - `det_id`: lista `int` secuencial (0..N-1)
  - `bbox_xyxy`: lista de listas `[x1, y1, x2, y2]` en `float`
  - `confidence`: lista `float`
  - `class_name`: lista `str` (solo clases soportadas)
- `trace` (solo en debug):
  - `detections`: lista de detecciones con `det_id`, `bbox_xyxy`, `confidence`, `class_id`, `class_name`, `class_name_raw`, `render_color_bgr`
  - `summary`: `total_raw`, `total_supported`, `total_discarded`

**Ejemplo minimo**

```python
from football_ai.tracking.phases.detection import Detector
import cv2

detector = Detector(
    model_path="models/yolo/v11/yolov11m.pt",
    conf=0.1,
)

cap = cv2.VideoCapture("partido.mp4")
ok, frame_bgr = cap.read()
if ok:
    packet = detector.predict_frame(
        frame_bgr,
        frame_index=0,
        frame_time_ms=0.0,
        execution_mode="debug",
    )
    clean = packet["clean"]
    trace = packet["trace"]
    print(clean["num_detections"])
    print(trace.get("summary", {}))
cap.release()
```

---

## `phase.py` — `DetectionPhase`

Adaptador de fase que instancia `Detector` y delega en `predict_frame`.

**Uso**

```python
from football_ai.tracking.phases.detection import DetectionPhase

phase = DetectionPhase(
    model_path="models/yolo/v11/yolov11m.pt",
    detector_conf={"conf": 0.1, "verbose": False},
)
packet = phase.execute(frame_bgr, frame_index=0, frame_time_ms=0.0)
```
