# tracking

Pipeline completo de tracking multi-objeto para un partido de fútbol. Combina detección YOLO, identificación de equipo por color de camiseta, proyección automática al campo 2D y el algoritmo ByteTrack en un único flujo frame a frame.

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
        "track_thresh": 0.15,
        "track_buffer": 90,
        "match_thresh": 0.945,
        "frame_rate": 25,
        "minimum_consecutive_frames": 5
    },
    team_colors={
        "Real Madrid": np.array([255, 127, 127]),
        "Wolfsburgo": np.array([224, 77, 196])
    },
    ball_min_conf=0.01,
    field_tracking_conf={
        "enabled": True,
        "method": "pnlcalib",
        "classes": ["player", "goalkeeper"],
        "field_length_m": 106.0,
        "field_width_m": 68.0,
    }
)
```

Si no pasas `field_tracking_conf`, el tracker puede funcionar solo con `bbox` en imagen. En `scripts/track.py`, por defecto se lee esta configuración desde `config.yaml` y se activa la proyección 2D del campo.
`PnLCalib` no se guarda dentro de este repositorio como código versionado: el propio tracker lo clona en `models/reference_points/pnlcalib_repo/` y descarga sus pesos en la primera ejecución. Por tanto, otra persona que ya tenga este repo solo necesita `git pull`; no tiene que clonar `PnLCalib` manualmente.

### Pipeline interno de `get_tracks(video, show_kmeans)`

Por cada frame del vídeo:

1. **Detección YOLO** (`Detector.detect`): genera las detecciones brutas del frame.
2. **Identificación de equipo** (`TeamDetector.detect_teams`): por cada detección de `player/goalkeeper` extrae color de camiseta (KMeans en LAB) y asigna equipo. Puede operar en modo `reference` o `auto-bootstrap`.
3. **PnLCalibFieldProjector**: calibra el campo en ese frame y proyecta `player` y `goalkeeper` a coordenadas métricas `[x_m, y_m]` sobre el césped.
4. **ByteTrack** (`ByteTrack.update_with_detections`): asocia las detecciones a tracks con IDs persistentes entre frames. Usa la etiqueta de equipo como penalización adicional y, para `player`/`goalkeeper`, incorpora distancia en el campo 2D al coste de asociación.
5. **Fallback de balón**: si ByteTrack no activó ningún track para el balón en ese frame (porque su confianza es demasiado baja para el umbral de activación), se añaden las detecciones YOLO crudas con IDs `"fallback_N"`. Esto garantiza que siempre haya información del balón aunque no sea trazable.

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
    "field_position_m": [x, y] | None, # coordenadas reales sobre el campo en metros
    "ground_point_image": [x, y] | None,# punto imagen usado para proyectar al campo
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
Implementar el algoritmo **ByteTrack** con extensiones propias para incorporar la información de equipo y la posición 2D sobre el campo como restricciones adicionales en la asociación de detecciones a tracks.

### Base: ByteTrack original

ByteTrack es un algoritmo de tracking multi-objeto que mejora otros métodos al usar **todas** las detecciones (no solo las de alta confianza) en una segunda ronda de asociación:

1. **Primera asociación** (alta confianza): detecciones con `score > track_thresh` se asocian a tracks activos usando distancia IoU + filtro de Kalman.
2. **Segunda asociación** (baja confianza): detecciones con `0.1 < score < track_thresh` se asocian a tracks perdidos en el paso anterior.
3. **Nuevos tracks**: detecciones sin asociar inicializan nuevos tracks tentatives.
4. **Confirmación**: un track pasa a activo después de `minimum_consecutive_frames` frames consecutivos.
5. **Eliminación**: un track perdido se elimina tras `lost_track_buffer` frames sin detección.

### Extensión: penalización por equipo

Se ha añadido un atributo `team` a cada `STrack`. Si en la primera asociación se intenta asociar una detección de un equipo distinto al del track, se añade una **penalización** (configurable vía `team_penalty` en config.yaml, por defecto 1000) a la matriz de costes IoU, haciendo esa asociación prácticamente imposible.

### Modos de asignación de equipo (TeamDetector)

Parámetros en `config.yaml`:
- `team_assignment_mode`: `reference` o `auto-bootstrap`.
- `team_bootstrap_frames`: frames iniciales usados para construir clusters de color en `auto-bootstrap`.
- `team_bootstrap_min_samples`: muestras mínimas de color para cerrar bootstrap.
- `team_bootstrap_num_teams`: número de equipos/clusters a separar (normalmente 2).
- `team_bootstrap_min_cluster_samples`: mínimo de muestras por cluster al cerrar bootstrap (por defecto 4, es decir, >3 nodos por cluster).
- `team_auto_name_prefix`: prefijo de nombres automáticos (`Equipo 1`, `Equipo 2`, ...).
- `team_candidate_classes`: clases que aportan muestras de color (por defecto `player`, `goalkeeper`).

En `auto-bootstrap`, una vez cerrada la fase inicial, cada equipo queda fijado con la **mediana de su cluster** para evitar intercambio de etiquetas entre frames. Si aparece un cluster pequeño (<=3), se descarta como ruido y se re-clusteriza sobre el cluster mayor.

### Extensión: coste espacial en campo 2D

Para `player` y `goalkeeper`, si existe homografía válida en el frame:
- cada detección se proyecta a una coordenada real del campo `[x_m, y_m]`;
- cada track mantiene su última posición proyectada;
- el coste de matching añade una penalización proporcional a la distancia recorrida sobre el campo;
- además se bloquean asociaciones físicamente imposibles si la distancia en metros supera el umbral configurado.

Esto reduce cambios de ID provocados solo por movimiento de cámara o paneos fuertes.

### Extensión: gate estadístico anti-ID-switch

En `Tracker` se aplica además un filtro de movimiento por track canónico:
- mantiene media y desviación típica de la distancia recorrida por frame;
- calcula un límite dinámico `media + k * desviacion_tipica`;
- bloquea reasignaciones cuya distancia supere ese límite (escalado por frames perdidos), con un suelo mínimo de movimiento permitido.

Parámetros en `config.yaml`:
- `use_field_position_as_primary_cost` (si `true`, para `player/goalkeeper` el coste base del matching es distancia en campo)
- `use_bbox_center_for_matching` (mezcla distancia entre centros de bbox en el coste de matching)
- `bbox_center_distance_weight` (0..1, cuánto pesa bbox frente a IoU)
- `bbox_center_distance_gate_px` (normalización en píxeles; escala con frames perdidos)
- `lost_time_penalty_weight` (penaliza candidatos con más frames perdidos)
- `lost_time_penalty_max_frames` (normalización del penalizador temporal)
- `field_position_match_distance_gate_m` (base del gate espacial en metros)
- `field_position_match_distance_cap_m` (tope absoluto del gate espacial acumulado)
- `field_position_match_distance_max_lost_frames` (tope de crecimiento temporal del gate)
- `field_position_match_distance_growth_mode` (`power` o `linear_decay`)
- `field_position_match_distance_lost_exponent` (si < 1, crecimiento sublineal)
- `field_position_match_distance_decay_per_frame` (en `linear_decay`, cuánto decrece cada paso por frame perdido)
- `motion_std_gate_enabled`
- `motion_std_factor`
- `motion_std_min_samples`
- `motion_std_floor`
- `strict_person_class_separation` (si `true`, no mezcla `player` y `goalkeeper`)
- `require_field_position_for_reassign` (si `true`, `player/goalkeeper` no hacen fallback a píxeles)
- `max_reassign_lost_frames` / `max_reassign_lost_frames_by_class` (opcionales; `null` o `<=0` desactiva el corte temporal y permite reapariciones tardías)

Para reducir ID switches en clips largos, conviene combinar este gate con límites de reasignación más estrictos:
- `reassign_min_distance` (imagen, píxeles; útil en clases sin campo)
- `reassign_min_field_distance_m` (campo 2D, metros)

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
