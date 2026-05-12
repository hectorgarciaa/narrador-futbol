# tracking

Pipeline completo de tracking canónico para un partido de fútbol. Consume fases desacopladas y las transforma en tracks finales estables por ID canónico, junto con roles online.

Nota de desacoplo: la lógica de canonización (matching canónico, seeds, balón y gates de continuidad) vive en `football_ai/canonicaltrack`, y la posesión vive en `football_ai/posession`. `tracking` se mantiene como orquestador del pipeline.

---

## `tracker.py` — `Tracker`

### Objetivo
Orquestar el pipeline de tracking desacoplado (`DETECTOR -> REFERENCE_POINTS -> FILTERING -> IDENTIFICATION -> BYTETRACK -> CANONICALTRACK -> POSESSION + POSITION_INFERING`) y devolver un diccionario de tracks con información completa por cada objeto detectado en cada frame.

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
    ball_min_conf=0.0035,
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
`PnLCalib` no se guarda dentro de este repositorio como código versionado: el propio tracker lo clona en `external/pnlcalib/` y descarga sus pesos en `models/pnlcalib/` durante la primera ejecución. Por tanto, otra persona que ya tenga este repo solo necesita `git pull`; no tiene que clonar `PnLCalib` manualmente.

### Pipeline interno de `get_tracks(video, show_kmeans, frame_hook=None, collect_visual_debug=False, profile_phases=False)`

Por cada frame del vídeo:

1. **Detector** (`Detector.predict_frame`): emite un `PhaseFramePacket` `DETECTOR` con dos vistas fijas del frame:
   - `clean`: arrays paralelos (`det_id`, `bbox_xyxy`, `confidence`, `class_name`, ...).
   - `trace`: objetos serializables para JSON/debug/render.
2. **Reference Points** (`PnLCalibFieldProjector.project_frame`): consume `frame_bgr + detector_packet` y emite `REFERENCE_POINTS`. Ejecuta un único forward de `PnLCalib` por frame, reutiliza esos heatmaps en los intentos adaptativos y publica la homografía final en coordenadas de imagen original.
   - `clean` contiene solo las señales que consumen fases posteriores del pipeline: `num_detections`, `det_id`, `bbox_xyxy`, `confidence`, `class_name`, `class_name_raw`, `field_positions_m`, `ground_points_image_original`, `field_positions_usable_for_tracking`, `homography_valid` y `homography_image_to_field_3x3`.
   - `trace` agrupa el resto de metadatos de calibración y depuración: `ground_points_image_projected`, métricas de calidad (`quality_status`, `quality_score`, `reprojection_error_px`, recuentos visibles, thresholds usados, `attempt_count`, `estimation_mode`) y el detalle extendido (`attempts`, `keypoints`, `lines`, rechazos).
3. **Filtering** (`filter_reference_points`): consume `REFERENCE_POINTS` y emite `FILTERING`.
   - `clean` conserva el mismo esquema que `REFERENCE_POINTS`, pero ya filtrado: solo quedan las detecciones aceptadas por la validación geométrica.
   - `trace` separa `accepted_detections` y `rejected_detections`, con `reject_code`, `reject_label`, flags geométricos y resumen agregado.
4. **Identificación de equipo** (`TeamDetector.identify_packet`): consume `FILTERING.clean`, trabaja ya con `frame_bgr + arrays alineados por detección` y emite `IDENTIFICATION`, sin depender de objetos `Results` de YOLO. Puede operar en modo `reference` o `auto-bootstrap`. Si el tracking se lanza desde la interfaz con un `lineup_spec.json`, por defecto usa los colores definidos por el usuario como referencias directas de equipo, igual que el tracking normal. Si se fuerza `auto-bootstrap`, el detector arranca sin referencias y aprende equipos neutrales. A partir de este punto, `goalkeeper` ya no se colapsa con `player`: tanto ByteTrack como la capa canónica distinguen ambas clases.
   - `clean` mantiene `class_name` como clase YOLO original y añade `class_name_td`, `team`, `shirt_color`, `distances`, `bbox_size`, `referee_reassign_gate` y `goalkeeper_reassign_gate`.
   - `trace` incluye detalle por detección (`sample_decision`, motivo de relabel, gates de referee/goalkeeper) y el estado/eventos de clustering (`clusters.events`, colores activos, distancias robustas por equipo).
   - Si `TeamDetector` ha propuesto relabelar una detección `player/goalkeeper` a `referee` por color, esa reasignación solo se acepta si la detección cumple al menos una de estas condiciones: estar dentro de la banda `+- tracking.referee_sideline_band_distance_m` respecto a las líneas laterales, o caer entre la cuarta `x` más a la izquierda y la cuarta más a la derecha de los jugadores visibles en ese frame. Si no cumple ninguna de las dos, la detección vuelve a su clase original de YOLO y recupera el equipo de campo más cercano por color.
   - Además, una detección actualmente relabelada como `player` o `referee` puede promocionarse a `goalkeeper` si su color de camiseta es un outlier robusto respecto a los dos equipos de campo y su `x` proyectada queda fuera del corredor delimitado por la tercera persona más a la izquierda y la tercera más a la derecha visibles en ese frame. Para evitar confundir linieres con porteros, esta promoción solo se permite si la detección queda a más de 3 metros de las bandas laterales.
5. **BYTETRACK** (`football_ai.bytetrack.ByteTrackPhase.track_packet`): consume `IDENTIFICATION.clean` como fase desacoplada y emite `BYTETRACK`.
   - `clean` conserva todas las señales de `IDENTIFICATION` y añade `tracker_id`, `class_tracker`, `tracked_mask`, `tracked_count` y `tracked_detections`.
   - `trace` publica `detection_debug` alineado por `det_id`, `matching_debug` por subfase (`high_iou`, `low_iou`, `unconfirmed_iou`, `high_bbox`, `low_bbox`, `unconfirmed_bbox`) y un resumen de entradas trackeadas/no trackeadas.
   - Esta fase ya no vive dentro de `tracking`: el tracker canónico la consume como entrada intermedia del pipeline.
6. **CANONICALTRACK** (`football_ai.canonicaltrack.CanonicalTrackPhase.canonicalize_packet`): consume exclusivamente `BYTETRACK` y emite `CANONICALTRACK`.
   - `clean` publica `tracks_frame` por clase (`player`, `goalkeeper`, `referee`, `ball`), `summary` y `canonical_ids_in_frame`.
   - `trace` publica depuración de asignaciones y descartes (`pending_assignments_debug`, `discard_reason_by_raw_idx`, `forced_absorption_debug`, `canonical_state_debug_snapshot`, `ball_selection_debug`).
   - `Tracker` reconstruye los acumulados históricos de salida a partir de `CANONICALTRACK.clean.tracks_frame`, sin ejecutar inline la capa canónica.
7. **Seeds canónicos opcionales en punto de penalti**: si `reserve_penalty_spot_seed_players=true`, el tracker crea dos tracks semilla sintéticos de clase `goalkeeper` en los puntos de penalti. No pasan por `TeamDetector`, así que no contaminan el clustering de colores ni tienen equipo asignado. Sí participan en la reasignación canónica por posición de campo, reservando los IDs 1 y 2 para porteros no visibles al inicio. Mientras no absorban una detección real, también se escriben en el JSON final con `synthetic_seed=true`.
8. **Lógica especial para los IDs reservados y roles online**: esos dos IDs quedan reservados a porteros. Solo detecciones cuya clase resuelta sea `goalkeeper` pueden ocupar los IDs 1 y 2, tanto en la capa canónica como al absorber sobre los seeds especiales. Su equipo se sigue asignando por el modelo de roles posicionales y el defensa más cercano. Como ya se consideran porteros conocidos, no entran al Set Transformer y se etiquetan manualmente como `POR`. La memoria táctica se sigue acumulando por segmentos semánticos y puede reiniciarse si hay un salto espacial fuerte, una reasignación/reabsorción canónica (`canonical_relinked`) o una deriva sostenida del track hacia otro slot/jugador del lineup.
9. **Reasignación canónica coherente con ByteTrack**: para `player/goalkeeper` con `field_position_m`, la segunda capa de IDs canónicos usa exactamente el mismo gate geométrico que ByteTrack (`field_position_match_distance_*`). No añade un suelo extra ni expansión por velocidad en esa capa, así que no puede reusar un ID final con un salto de campo mayor que el permitido por la capa base. Si falta `field_position_m` porque la homografía fue rechazada o no se pudo calcular, la canonización cae a bbox/centro de bbox y mantiene la continuidad con señales visuales en vez de bloquear la reasignación.
   - Los IDs canónicos de personas se limitan al rango `1..N_personas` (el balón no consume ese rango y siempre usa `id=0`).
  - Los `goalkeeper` siguen reservando los IDs `1-2`, pero los `player` se reparten en dos bloques fijos por equipo: `3-12` para el primer equipo de campo y `13-22` para el segundo. La capa canónica nunca crea más de 10 jugadores por equipo ni permite que un `player` cruce de bloque en reassign, relink o forced absorption.
  - Si el `TeamDetector` emite etiquetas explícitas `Equipo 1`/`Equipo 2`, esos nombres se alinean directamente con los bloques `3-12` y `13-22`; con nombres reales de club, el canónico fija la correspondencia al primer mapeo estable que vea y la conserva durante el resto del vídeo.
  - Los IDs canónicos `tracking.referee_canonical_ids` (por defecto `23,24,25`) quedan reservados a árbitros.
  - Esos slots de árbitro ya no son intercambiables: el primero (`23`) se reserva al árbitro central, el segundo (`24`) al linier de la banda superior (`sideline_top`) y el tercero (`25`) al linier de la banda inferior (`sideline_bottom`).
  - La reabsorción de árbitros usa una lógica específica por zona de campo: `sideline_top`, `sideline_bottom` y `central`, calculadas desde `field_position_m`. Si un árbitro reaparece en la misma zona, reabsorbe ese ID reservado aunque haya habido un pequeño gap temporal.
  - En la capa canónica, un ID `referee` solo acepta detecciones de entrada cuya clase resuelta siga siendo `referee`; ya no puede reabsorber detecciones `player`.
  - El árbitro central solo puede crear/reabsorber sobre el slot `23` si la detección cae en zona `central`, es decir, fuera de la franja `tracking.referee_sideline_band_distance_m` respecto a las bandas laterales, y además su `x` en campo cae entre la segunda `x` más a la izquierda y la segunda más a la derecha de los `player/goalkeeper` visibles en ese frame; así se evita absorber tracks pegados a las porterías.
  - Cada linier solo puede crear/reabsorber su slot de banda correspondiente: `24` para `sideline_top` y `25` para `sideline_bottom`. Si ese slot está libre se crea ahí; si ya existe, solo se intenta absorber sobre ese mismo slot.
   - Además existe una **absorción forzada conservadora** solo para `player`: si un canónico de jugador lleva perdido al menos `tracking.forced_absorption_player_min_lost_frames` frames y ByteTrack mantiene durante `tracking.forced_absorption_player_min_consistent_frames` frames seguidos un `raw_tracker_id` huérfano con la misma clase `player` y el mismo equipo, la capa canónica puede reusar ese ID perdido aunque el gate geométrico normal no lo aceptase. Esta vía no aplica gate de posición como veto, pero sí mantiene el bloque de IDs del equipo (`3-12` o `13-22`). Cuando hay varios canónicos perdidos y/o varios huérfanos compatibles del mismo equipo, construye todas las parejas posibles y resuelve un matching greedy por menor distancia usando, para cada canónico, la muestra del huérfano más cercana en tiempo al frame en que se perdió ese canónico.
10. **Selección robusta del balón**: las candidatas de balón, tanto las devueltas por ByteTrack como las detecciones YOLO crudas, pasan por un gate específico de continuidad. A diferencia de `player/goalkeeper`, aquí no se aplica además el gate genérico de reasignación: se usa solo la lógica propia del balón para no perder cobertura. Se valida que el balón:
   - no salte a una posición incompatible con su trayectoria reciente;
   - no cambie de tamaño de forma abrupta entre frames;
   - y, si hay varias candidatas plausibles, se prioriza la más coherente con la posición esperada y la confianza.
   Si ninguna candidata es físicamente plausible, ese frame queda sin balón en vez de aceptar un teletransporte. Cuando la trayectoria prevista saca el balón fuera de la imagen, la búsqueda queda anclada al borde por el que salió; no se aceptan reapariciones “hacia atrás” dentro de la pantalla. Solo tras `tracking.ball.max_reassign_lost_frames` frames perdidos se permite una redetección libre por máxima confianza.
11. **POSESSION** (`football_ai.posession.PosessionPhase.process`): consume `CANONICALTRACK`, estima posesión con heurísticas temporales (distancia balón-pie, contacto estricto/flexible y señales de movimiento), y emite un packet con `clean`+`trace`.
   - `clean` conserva toda la salida previa de `CANONICALTRACK`, añade `possession` y enriquece `tracks_frame` con metadatos de posesión por track.
   - `trace` conserva la traza de `CANONICALTRACK` y añade el bloque `possession`.
12. **POSITION_INFERING** (`football_ai.positions.PositionInferingPhase.process`): consume el mismo `CANONICALTRACK` en paralelo, anota `tracks_frame` con roles online y emite un packet propio con `clean`+`trace`.
   - `clean` conserva la salida canónica, reescribe `tracks_frame` con metadatos de rol y añade un resumen del frame en `position_infering`.
   - `trace` conserva la traza de `CANONICALTRACK` y añade `position_infering.frame_summary` más estadísticas acumuladas.
13. **Merge final del tracker**: `Tracker` fusiona `POSESSION` y `POSITION_INFERING` sobre el mismo `tracks_frame` canónico antes de reconstruir el `tracks` final y antes de ejecutar hooks per-frame opcionales como PathCRF live.

Todas las fases heredan de `football_ai.core.Phase` y exponen `process()`, que mide automáticamente el tiempo de ejecución e inyecta `elapsed_ms` en el packet. Las fases iniciales viven en sus módulos dueños (`football_ai.detection.DetectionPhase`, `football_ai.reference_points.ProjectionPhase`, `football_ai.filtering.FilteringPhase`, `football_ai.identification.IdentificationPhase`) y `tracking` solo orquesta. Si `profile_phases=True`, el tracker imprime por frame solo las 8 fases funcionales del pipeline (`Detection`, `Projection`, `Filtering`, `Identification`, `ByteTrack`, `CanonicalTrack`, `Posession`, `PositionInfering`) y su total. Las tareas auxiliares fuera de fase, como construir debug visual o ejecutar `frame_hook`, siguen ocurriendo pero ya no entran en ese profiling.

### Formato de salida

```python
tracks = {
    "player":     [frame_0_dict, frame_1_dict, ...],  # lista de len = n_frames
    "goalkeeper": [...],
    "referee":    [...],
    "ball":       [...],
    "possession": [frame_0_possession, frame_1_possession, ...]
}
```

Cada `frame_N_dict` es `{track_id: datos_objeto}` donde `track_id` es un entero y `datos_objeto` es:

```python
{
    "bbox":        [x1, y1, x2, y2],   # coordenadas en píxeles
    "field_position_m": [x, y] | None, # coordenadas reales sobre el campo en metros
    "ground_point_image": [x, y] | None,# punto imagen usado para proyectar al campo
    "confidence":  float,              # confianza de la detección YOLO
    "team":        str | None,         # nombre del equipo asignado
    "distances":   {"Equipo A": float, "Equipo B": float} | None,
    "shirt_color": [L, A, B] | None,   # color en espacio LAB
    "class_name_td": str | None,       # clase relabelada por TeamDetector
    "bbox_size":   float,              # área del bounding box en píxeles²
    "source_raw_tracker_id": int,      # raw tracker de ByteTrack que alimentó este canónico
    "canonical_assignment_mode": str,  # raw_continuity / canonical_relinked_to_new_raw_tracker / forced_absorption / ...
    "canonical_relinked": bool,        # true si el canónico reapareció enlazado a otro raw tracker
    "goalkeeper_reassign_gate": dict | None,
    "forced_absorption": bool,         # true si el ID entró por reabsorción forzada sin gate de posición
    "forced_absorption_raw_tracker_streak_frames": int | None,
    "forced_absorption_canonical_lost_frames": int | None,
    "forced_absorption_source_raw_tracker_id": int | None,
    "forced_absorption_mode": str | None,
    "forced_absorption_reference_frame": int | None,
    "forced_absorption_distance_sq": float | None,
    "is_possession_player": bool,      # true en el jugador/portero poseedor del frame
    "ball_owning_team_id": str | None, # equipo con posesión en ese frame
    "ball_owning_player_id": int | None, # id canónico del jugador poseedor
    "player_id": int | None,           # alias del poseedor para consumidores downstream
    "possession_reason": str | None,   # razón heurística (ej. start_touch, last_touch_hold)
    "identity_segment_id": int | None, # segmento semántico activo de ese canonical_id
    "identity_segment_start_frame": int | None,
    "identity_segment_observations": int | None,
    "identity_reset": bool | None,
    "identity_reset_reason": str | None,
    "identity_reset_position_jump_m": float | None,
    "segment_majority_role": str | None,
    "segment_recent_majority_role": str | None,
    "segment_majority_expected_role_slot": str | None
}
```

Y para `tracks["possession"][frame_id]`:

```python
{
    "team_id": str | None,
    "player_id": int | None,
    "reason": str | None,
    "ball_detected": bool,
    "nearest_track_id": int | None,
    "nearest_team_id": str | None
}
```

---

## ByteTrack desacoplado

La implementación base de ByteTrack ya no vive en este módulo, sino en [`football_ai/bytetrack`](../bytetrack/README.md). `tracking` consume su salida `BYTETRACK` y se centra en orquestar fases desacopladas + artefactos finales.

## `football_ai/bytetrack/byte_tracker.py` — `ByteTrack`

### Objetivo
Implementar el algoritmo **ByteTrack** con extensiones propias para incorporar la información de equipo y la posición 2D sobre el campo como restricciones adicionales en la asociación de detecciones a tracks.

### Base: ByteTrack original

ByteTrack es un algoritmo de tracking multi-objeto que mejora otros métodos al usar **todas** las detecciones (no solo las de alta confianza) en una segunda ronda de asociación:

1. **Primera asociación** (alta confianza): detecciones con `score > track_thresh` se asocian a tracks activos usando distancia IoU + filtro de Kalman.
2. **Segunda asociación** (baja confianza): detecciones con `0.1 < score < track_thresh` se asocian a tracks perdidos en el paso anterior.
3. **Nuevos tracks**: detecciones sin asociar pueden inicializar tracks tentativos, pero antes pasan un filtro anti-solape:
   - si solapan por encima de `new_track_active_overlap_iou` con cualquier track activo (`Tracked` + `is_activated=true`), se descartan;
   - si solapan por encima de `new_track_unconfirmed_overlap_iou` con un `unconfirmed` previo, se descartan;
   - entre candidatos nuevos del mismo frame, si solapan por encima de `new_track_candidate_overlap_iou`, se conserva solo el de mayor confianza.
4. **Confirmación**: un track pasa a activo después de `minimum_consecutive_frames` frames consecutivos.
5. **Eliminación**: un track perdido se elimina tras `lost_track_buffer` frames sin detección.

### Extensión: gates duros + consenso por track

Cada detección de persona llega al tracking con dos clases:
- `class_yolo`: clase original de YOLO.
- `class_name_td`: clase reetiquetada por `TeamDetector` (`class` se mantiene como alias interno de compatibilidad).

El matching raw ya no usa penalizaciones aditivas grandes ni `fuse_score`. En su lugar, cada subfase construye un `base_cost` y un `feasible_mask` común:
- `class` es siempre gate duro;
- `team` es gate duro solo para `player`, y se omite si falta el equipo en track o detección;
- `bbox_size` es gate duro;
- `field_position` solo actúa como gate cuando existen ambas posiciones válidas;
- las tres primeras pasadas exigen `IoU > 0`;
- las tres segundas pasadas exigen `IoU = 0` y rescatan por `bbox_center_distance`.

Además, cada `STrack` sigue acumulando evidencia temporal (`class_vote_weight_relabel`, `class_vote_weight_yolo`) y solo cambia su clase interna cuando la nueva hipótesis supera un margen de consenso (`class_consensus_switch_margin`).

### Extensión: color de camiseta como metadata (sin coste/gate)

El color LAB de camiseta se mantiene en la metadata de detecciones/tracks para análisis y visualización, pero actualmente no añade penalización al coste de ByteTrack ni gate de veto en la canonización.

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

### Extensión: gate espacial en campo 2D

Para `player` y `goalkeeper`, si existe homografía válida en el frame:
- cada detección se proyecta a una coordenada real del campo `[x_m, y_m]`;
- cada track mantiene su última posición proyectada;
- el matching raw no usa esa distancia como coste principal;
- sí bloquea asociaciones físicamente imposibles si la distancia en metros supera el umbral configurado.

Esto reduce cambios de ID provocados solo por movimiento de cámara o paneos fuertes.

### Extensión: gate estadístico anti-ID-switch

En `Tracker` se aplica además un filtro de movimiento por track canónico:
- mantiene media y desviación típica de la distancia recorrida por frame;
- calcula un límite dinámico `media + k * desviacion_tipica`;
- bloquea reasignaciones cuya distancia supere ese límite (escalado por frames perdidos), con un suelo mínimo de movimiento permitido.

Parámetros en `config.yaml`:
- `track_activation_threshold` y `low_conf_threshold` (separan detecciones `high_conf` y `low_conf`)
- `bbox_center_distance_gate_px` (gate para rescates con `IoU = 0`; escala con frames perdidos)
- `lost_time_penalty_weight` (penaliza candidatos con más frames perdidos)
- `lost_time_penalty_max_frames` (normalización del penalizador temporal)
- `class_vote_weight_relabel` / `class_vote_weight_yolo` (peso de cada señal de clase en el consenso temporal)
- `class_consensus_switch_margin` (ventaja mínima acumulada para permitir cambio de clase en el track interno)
- `bbox_height_ratio_threshold` / `bbox_width_ratio_threshold` (gates duros de cambio de tamaño)
- `bbox_height_ratio_threshold` (desviación relativa máxima permitida en altura antes de penalizar)
- `bbox_width_ratio_threshold` (desviación relativa máxima permitida en ancho antes de penalizar)
- `field_position_match_distance_gate_m` (base del gate espacial en metros)
- `field_position_match_distance_cap_m` (tope absoluto del gate espacial acumulado)
- `field_position_match_distance_max_lost_frames` (tope de crecimiento temporal del gate)
- `field_position_match_distance_growth_mode` (`power` o `linear_decay`)
- `field_position_match_distance_lost_exponent` (si < 1, crecimiento sublineal)
- `field_position_match_distance_decay_per_frame` (en `linear_decay`, cuánto decrece cada paso por frame perdido)
- `adaptive_thresholds_enabled` (si `true`, PnLCalib reintenta el frame con thresholds más bajos antes de decidir la homografía final)
- `adaptive_max_attempts` (máximo de intentos por frame, contando el threshold base)
- `adaptive_keypoint_threshold_step` / `adaptive_line_threshold_step` (rebaja progresiva aplicada en cada intento)
- `adaptive_keypoint_threshold_min` / `adaptive_line_threshold_min` (suelo de thresholds para el rescate)
- Dentro de `projector.constructor.projection_quality_analyzer_conf`:
- `adaptive_min_visible_keypoints` / `adaptive_min_visible_lines` (mínimos duros para no aceptar una homografía con soporte insuficiente)
- `validation_score_threshold` (score mínimo global para clasificar la homografía como `good`)
- `validation_geometry_fit_min` / `validation_support_quality_min` / `validation_coverage_quality_min` (suelos por submétrica; si alguno cae por debajo, la homografía se rechaza aunque el score total sea alto)
- `validation_reprojection_error_threshold_px` (escala de referencia para convertir el `reprojection_error` a score)
- `validation_homography_condition_number_threshold` (tope numérico de condición para rechazar homografías mal condicionadas)
- `validation_keypoint_world_error_threshold_m` (escala de referencia para puntuar la distancia entre keypoints proyectados y sus posiciones canónicas en campo)
- `validation_line_world_error_threshold_m` (escala de referencia para puntuar la distancia entre líneas planas proyectadas y sus segmentos canónicos)
- `validation_target_visible_keypoints` / `validation_target_visible_lines` (objetivos de soporte a partir de los cuales el score de soporte satura)
- `validation_target_image_point_hull_area_ratio` (objetivo de cobertura en imagen medido con el área del convex hull de keypoints y extremos de línea)
- `validation_target_image_x_span_ratio` / `validation_target_image_y_span_ratio` (objetivos de apertura horizontal/vertical en imagen)
- `validation_target_field_coverage_ratio` (objetivo de cobertura semántica en campo usando las posiciones canónicas visibles de keypoints y líneas planas)
- En `quality_diagnostics`, si `PnLCalib` no devuelve homografía (`no_homography`), el frame queda forzado a `rejected` con `quality_score=0.0` y `rejection_type=no_solution`
- `rejection_type` resume el motivo dominante del rechazo: `no_solution`, `bad_geometry`, `low_support` o `low_score`
- `new_track_active_overlap_iou` (IoU máxima permitida entre un track nuevo y un track activo ya consolidado)
- `new_track_unconfirmed_overlap_iou` (IoU máxima permitida entre un track nuevo y un `unconfirmed` previo)
- `new_track_candidate_overlap_iou` (IoU máxima permitida entre dos candidatos nuevos del mismo frame)
- `max_tracks_per_class` (lo usa hoy la capa canónica/final del tracker; reparto recomendado: `goalkeeper=2`, `player=20`, `referee=3`, `ball=1`. Esos `20 player` se reparten internamente como `10 + 10` entre los bloques canónicos `3-12` y `13-22`)
- `motion_std_gate_enabled`
- `motion_std_factor`
- `motion_std_min_samples`
- `motion_std_floor`
- `referee_canonical_ids` (IDs reservados exclusivamente para árbitros; por defecto `[23, 24, 25]`)
- `referee_sideline_band_distance_m` (franja en metros desde cada banda para clasificar linieres vs árbitro central)
- `ball.expected_position_gate_px`
- `ball.expected_position_gate_growth_per_frame`
- `ball.expected_position_confidence_relax`
- `ball.size_ratio_per_frame`
- `ball.size_min_samples`
- `ball.size_std_factor`
- `ball.size_std_floor`
- `ball.max_reassign_lost_frames`
- `ball.high_conf_override`
- `strict_person_class_separation` (si `true`, no mezcla `player` y `goalkeeper`)
- `reserve_penalty_spot_seed_players` (si `true`, reserva dos IDs sintéticos en los puntos de penalti)
- `reserve_penalty_spot_seed_match_distance_m` (radio máximo en metros para absorber una detección real sobre cada seed)
- `special_seed_role_team_assignment_enabled` (si `true`, ejecuta el modelo de roles para asignar equipo a los IDs reservados)
- `special_seed_role_model_path` (checkpoint del Set Transformer usado por el tracking principal)
- `special_seed_canonical_ids` (IDs canónicos tratados con esa lógica especial; por defecto `[1, 2]`)
- `special_seed_defender_roles` (roles que cuentan como defensas al buscar el más cercano)
- `expected_roles_by_team` (once esperado por equipo para restringir la inferencia posicional online con Hungarian)
- `role_stabilization_expected_roles_assignment` (estrategia para resolver `expected_roles_by_team` al congelar roles estables: `hungarian`, `greedy` o `ratio_priority`; por defecto `hungarian`)
- `role_stabilization_expected_roles_min_ratio` (solo en `ratio_priority`: ratio acumulado mínimo hasta la primera plaza libre en la foto global; por defecto `0.40`)
- `role_stabilization_expected_roles_min_final_ratio` (solo en `ratio_priority`: ratio mínimo de la plaza final elegida; por defecto hereda `role_stabilization_expected_roles_min_ratio`)
- `role_stabilization_expected_roles_min_count` (solo en `ratio_priority`: mínimo de observaciones del jugador para entrar en la asignación táctica de la foto global; por defecto `300`)
- `role_stabilization_window_frames` (número máximo de observaciones visibles por ID antes de congelar su role estable; en `ratio_priority` también actúa como frame de corte global de la foto; por defecto `600`)
- `role_stabilization_min_observations` (mínimo de observaciones antes de permitir congelado; si coincide con la ventana, la congelación ocurre al agotar esa ventana, por defecto `600`)
- `role_stabilization_vote_ratio` (porcentaje mínimo de dominio de una clase para congelar el role antes de agotar la ventana)
- `role_context_interpolation_max_gap_frames` (frames máximos en los que un canónico ausente puede reaparecer solo como contexto interpolado para el Set Transformer)
- `role_recent_position_window` (ventana corta de posiciones recientes del segmento, en metros, usada para la mediana espacial de swap)
- `role_recent_role_window` (ventana corta de roles crudos recientes del segmento)
- `role_swap_min_recent_samples` (mínimo de muestras visibles antes de permitir corte por swap)
- `role_swap_position_jump_m` (salto espacial mínimo para sospechar swap sin señal estructural de relink)
- `role_swap_position_jump_relinked_m` (salto espacial mínimo si la capa canónica ya marcó `canonical_relinked`)
- `role_swap_role_change_min_ratio` (dominancia mínima del rol reciente para usarlo como evidencia de swap)
- `role_swap_role_change_min_confidence` (confianza mínima del rol actual para activar la señal semántica de swap)
- `reassign_motion_growth_cap_frames` (tope de frames perdidos que se usan para extrapolar el salto permitido solo en clases sin homografía)
- `forced_absorption_enabled` (activa la reabsorción forzada conservadora para `player`)
- `forced_absorption_player_min_lost_frames` (frames mínimos perdidos del canónico `player` antes de ceder el ID; por defecto `20`)
- `forced_absorption_player_min_consistent_frames` (frames consecutivos mínimos del `raw_tracker_id` huérfano con misma clase/equipo; por defecto `10`)
- `output/tracker/<video>_role_artifacts/<video>_frame_role_predictions.csv` (predicción cruda frame a frame antes del congelado estable)
- `output/tracker/<video>_role_artifacts/<video>_player_role_summary.csv` (resumen estable por track al terminar el vídeo)
- `output/tracker/<video>_role_artifacts/<video>_greedy_role_diagnostics.csv` (traza paso a paso de las métricas usadas por el greedy al congelar slots estables)
- `output/tracker/<video>_role_artifacts/<video>_role_assignment_vs_detected_pre<frame>.png` (comparativa por jugador entre distribución detectada hasta el frame de corte y posición final)
- `output/tracker/<video>_role_artifacts/<video>_<equipo>_ratio_priority_step_by_step.png` (solo en `ratio_priority`: simulación paso a paso de la asignación snapshot por equipo)
- `require_field_position_for_reassign` (legacy: si `true`, `player/goalkeeper` priorizan el espacio de campo cuando existe, pero si la homografía falta o se rechaza el canónico cae a píxeles)
- `max_reassign_lost_frames` / `max_reassign_lost_frames_by_class` (opcionales; `null` o `<=0` desactiva el corte temporal y permite reapariciones tardías)

Para reducir ID switches en clips largos, conviene combinar este gate con límites de reasignación más estrictos:
- `reassign_min_distance` (imagen, píxeles; útil en clases sin campo)
- `field_position_match_distance_*` (si el problema está en `player/goalkeeper` con homografía)

Además, se implementa un mecanismo de **consenso temporal de equipo**:
- Cada track acumula votos de equipo por frame (`team_vote_weight`).
- El equipo canónico del track solo cambia cuando el nuevo equipo supera al actual por un margen (`team_consensus_switch_margin`).
- Esto evita que falsos positivos puntuales del `TeamDetector` contaminen la identidad de equipo.

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
