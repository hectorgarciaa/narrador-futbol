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
- `detect(video, stream=True)` llama a `model.predict()` y devuelve un **generador** de objetos `Results` de Ultralytics, uno por frame. El modo `stream=True` evita cargar todos los frames en memoria a la vez, lo que es crítico para vídeos de partido completos.
- Antes de devolver cada `Results`, normaliza `result.names` a las cuatro clases canónicas del proyecto: `player`, `goalkeeper`, `referee`, `ball`. Ahí se colapsan aliases como `person`, `gk`, `goalie`, `ref`, `sports ball` o variantes en plural.

```python
from football_ai.detection import Detector

detector = Detector(
    model_path="models/yolo/v11/yolov11m.pt",
    conf=0.1
)

for frame_result in detector.detect("partido.mp4", stream=True):
    # frame_result es un objeto Results de Ultralytics
    boxes = frame_result.boxes          # coordenadas, confianza, clase
    image = frame_result.orig_img       # frame BGR (NumPy)
    names = frame_result.names          # {0: 'player', 1: 'goalkeeper', ...}
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
