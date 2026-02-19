# tracking

Pipeline completo de tracking multi-objeto para un partido de fútbol. Combina detección YOLO, identificación de equipo por color de camiseta y el algoritmo ByteTrack en un único flujo frame a frame.

---

## `tracker.py` — `Tracker`

### Objetivo
Orquestar las tres etapas del pipeline de tracking y devolver un diccionario de tracks con información completa por cada objeto detectado en cada frame.

### Inicialización

```python
from football_ai.tracking import Tracker
import numpy as np

tracker = Tracker(
    model_path="models/finetuning/yolov11m.pt",
    conf=0.1,
    tracker_conf={
        "track_thresh": 0.5,
        "track_buffer": 90,
        "match_thresh": 0.945,
        "frame_rate": 25,
        "minimum_consecutive_frames": 5
    },
    team_colors={
        "Real Madrid": np.array([255, 127, 127]),
        "Wolfsburgo": np.array([224, 77, 196])
    },
    ball_min_conf=0.01
)
```

### Pipeline interno de `get_tracks(video, show_kmeans)`

Por cada frame del vídeo:

1. **Detección YOLO** (`Detector.detect`): genera las detecciones brutas del frame.
2. **Identificación de equipo** (`TeamDetector.detect_teams`): por cada detección extrae el color de camiseta (KMeans en LAB) y asigna un equipo.
3. **ByteTrack** (`ByteTrack.update_with_detections`): asocia las detecciones a tracks con IDs persistentes entre frames. Usa la etiqueta de equipo como penalización adicional en el coste de asociación.
4. **Fallback de balón**: si ByteTrack no activó ningún track para el balón en ese frame (porque su confianza es demasiado baja para el umbral de activación), se añaden las detecciones YOLO crudas con IDs `"fallback_N"`. Esto garantiza que siempre haya información del balón aunque no sea trazable.

### Formato de salida

```python
tracks = {
    "player":     [frame_0_dict, frame_1_dict, ...],  # lista de len = n_frames
    "goalkeeper": [...],
    "referee":    [...],
    "ball":       [...]
}
```

Cada `frame_N_dict` es `{track_id: datos_objeto}` donde `track_id` es un entero (o `"fallback_N"` para balón sin tracking) y `datos_objeto` es:

```python
{
    "bbox":        [x1, y1, x2, y2],   # coordenadas en píxeles
    "confidence":  float,              # confianza de la detección YOLO
    "team":        str | None,         # nombre del equipo asignado
    "distances":   {"Equipo A": float, "Equipo B": float} | None,
    "shirt_color": [L, A, B] | None,   # color en espacio LAB
    "bbox_size":   float               # área del bounding box en píxeles²
}
```

---

## `byte_tracker.py` — `ByteTrack`

### Objetivo
Implementar el algoritmo **ByteTrack** con una extensión propia para incorporar la información de equipo como restricción adicional en la asociación de detecciones a tracks.

### Base: ByteTrack original

ByteTrack es un algoritmo de tracking multi-objeto que mejora otros métodos al usar **todas** las detecciones (no solo las de alta confianza) en una segunda ronda de asociación:

1. **Primera asociación** (alta confianza): detecciones con `score > track_thresh` se asocian a tracks activos usando distancia IoU + filtro de Kalman.
2. **Segunda asociación** (baja confianza): detecciones con `0.1 < score < track_thresh` se asocian a tracks perdidos en el paso anterior.
3. **Nuevos tracks**: detecciones sin asociar inicializan nuevos tracks tentatives.
4. **Confirmación**: un track pasa a activo después de `minimum_consecutive_frames` frames consecutivos.
5. **Eliminación**: un track perdido se elimina tras `lost_track_buffer` frames sin detección.

### Extensión: penalización por equipo

Se ha añadido un atributo `team` a cada `STrack`. Si en la primera asociación se intenta asociar una detección de un equipo distinto al del track, se añade una **penalización** (configurable vía `team_penalty` en config.yaml, por defecto 1000) a la matriz de costes IoU, haciendo esa asociación prácticamente imposible.

Además, se implementa un mecanismo de **tolerancia a cambios temporales de equipo**:
- Si el equipo asignado cambia en un frame, no se actualiza inmediatamente.
- Se contabilizan los frames de discordancia en `team_switch_frames`.
- Solo después de `team_switch_threshold` frames consecutivos (configurable en config.yaml, por defecto 5) con el nuevo equipo se confirma el cambio.
- Esto evita que falsos positivos del `TeamDetector` contaminen la asignación de equipo.

### Parámetros principales

| Parámetro | Efecto |
|---|---|
| `track_thresh` | Confianza mínima para primera asociación. Subir = menos falsos tracks |
| `track_buffer` | Frames que un track sobrevive sin ser visto. Subir = mejor manejo de oclusiones |
| `match_thresh` | IoU mínimo para considerar una asociación válida. Subir = menos mezcla de jugadores |
| `minimum_consecutive_frames` | Frames para confirmar un track. Subir = elimina tracks de ruido |
| `frame_rate` | Afecta a `max_time_lost = frame_rate / 30 * track_buffer` |

### Estados de un track

```
Detected → Tentative → [confirmed] Active → Lost → Removed
                           ^________________|
                           (reasociado en frames siguientes)
```
