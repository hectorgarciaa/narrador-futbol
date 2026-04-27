# detection

Módulo de detección de objetos en fotogramas de vídeo de fútbol usando modelos YOLO de Ultralytics. Expone un wrapper genérico y una cabeza de detección personalizada para el balón.

## Clases detectadas

El sistema detecta cuatro clases, definidas en el dataset de fine-tuning de Roboflow:

| Clase | Descripción |
|---|---|
| `player` | Jugador de campo |
| `goalkeeper` | Portero |
| `referee` | Árbitro |
| `ball` | Balón |

---

## `detector.py` — `Detector`

**Objetivo:** Proporcionar una interfaz mínima sobre `ultralytics.YOLO` para que el resto del sistema no dependa directamente de la API de Ultralytics.

**Implementación:**
- Instancia un modelo YOLO en `__init__` con la ruta y el umbral de confianza.
- `predict_frame(frame_bgr, frame_index, frame_time_ms)` procesa un único frame y devuelve un `PhaseFramePacket` de fase `DETECTOR`.
- El paquete filtra desde el primer momento a las clases soportadas del proyecto (`player`, `goalkeeper`, `referee`, `ball`) y solo esas entran en `clean` y `trace`.
- La salida se divide en:
  - `clean`: arrays paralelos estables para encadenar pipeline.
  - `trace`: objetos serializables pensados para JSON y render/debug.

```python
from football_ai.detection import Detector
import cv2

detector = Detector(
    model_path="models/yolo/v11/yolov11m.pt",
    conf=0.1
)

cap = cv2.VideoCapture("partido.mp4")
ok, frame_bgr = cap.read()
if ok:
    packet = detector.predict_frame(frame_bgr, frame_index=0, frame_time_ms=0.0)
    clean = packet["clean"]
    trace = packet["trace"]
    print(clean["num_detections"])
    print(trace["summary"])
cap.release()
```

**¿Por qué un wrapper?** Desacopla el resto del código de Ultralytics: si se cambiase la librería de detección, solo habría que modificar esta clase.

El script [scripts/detect.py](/home/hegarc04/narrador-futbol/scripts/detect.py:1) usa directamente este wrapper y resuelve `video_path` y `model_path` tanto por shortcut de `config.yaml` como por ruta desde la raíz del repo.

---

## `ball_detector.py` — `DetectR8`

**Objetivo:** Mejorar la detección del balón, que es el objeto más pequeño y difícil de detectar en un partido, mediante una cabeza de detección con menor `reg_max`.

**Contexto técnico:** En YOLO, el módulo `Detect` usa Distribution Focal Loss (DFL) donde `reg_max` controla el número de bins para predecir la distribución de cada coordenada del bounding box. Por defecto es 16. Reducirlo a 8 produce un modelo más compacto y potencialmente más estable para objetos pequeños con poco contexto espacial.

**Implementación:**
- Hereda de `ultralytics.nn.modules.head.Detect`.
- En `__init__`, sobreescribe `reg_max = 8` y `no = nc + 8*4`.
- Reconstruye las cabezas `cv2` (regresión de bbox) y `dfl` con los nuevos parámetros.

```python
from football_ai.detection import DetectR8
from ultralytics import YOLO

# Cargar modelo fine-tuned de balón
model = YOLO("models/finetuning-balon/v11/yolov11m/weights/best.pt")

# Sustituir la cabeza de detección estándar por DetectR8
model.model[-1] = DetectR8(
    nc=model.model[-1].nc,
    ch=model.model[-1].ch
)

# Ahora el modelo detecta con reg_max=8
results = model("partido.mp4", stream=True)
```

> **Importante:** Este reemplazo debe hacerse **después** de cargar los pesos del modelo fine-tuned (que fue entrenado con `reg_max=8`). Si se aplica antes de cargar pesos, las dimensiones no coincidirán. Ver `scripts/detect_ball.py` para el uso correcto.

---

## Relación entre las dos clases

`Detector` y `DetectR8` son **independientes**. `Detector` es el wrapper genérico usado por `Tracker` para detección de jugadores. `DetectR8` es un módulo de bajo nivel que se aplica directamente al modelo YOLO en `scripts/detect_ball.py` para el caso especial de detección de balón.
