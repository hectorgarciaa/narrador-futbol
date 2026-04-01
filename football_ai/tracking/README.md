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
`PnLCalib` no se guarda dentro de este repositorio como código versionado: el propio tracker lo clona en `models/reference_points/pnlcalib_repo/` y descarga sus pesos en la primera ejecución. Por tanto, otra persona que ya tenga este repo solo necesita `git pull`; no tiene que clonar `PnLCalib` manualmente.

### Pipeline interno de `get_tracks(video, show_kmeans, frame_hook=None, collect_visual_debug=False, profile_phases=False)`

Por cada frame del vídeo:

1. **Detección YOLO** (`Detector.detect`): genera las detecciones brutas del frame.
2. **Identificación de equipo** (`TeamDetector.detect_teams`): por cada detección de `player/goalkeeper` extrae color de camiseta (KMeans en LAB) y asigna equipo. Puede operar en modo `reference` o `auto-bootstrap`. Si el tracking se lanza desde la interfaz con un `lineup_spec.json`, por defecto usa los colores definidos por el usuario como referencias directas de equipo, igual que el tracking normal. Si se fuerza `auto-bootstrap`, el detector arranca sin referencias y aprende equipos neutrales.
3. **PnLCalibFieldProjector + gate de relabel a árbitro**: calibra el campo en ese frame y proyecta `player` y `goalkeeper` a coordenadas métricas `[x_m, y_m]` sobre el césped. Después, si `TeamDetector` ha propuesto relabelar una detección `player/goalkeeper` a `referee` por color, esa reasignación solo se acepta si la detección cumple al menos una de estas condiciones: estar dentro de la banda `+- tracking.referee_sideline_band_distance_m` respecto a las líneas laterales, o caer entre la cuarta `x` más a la izquierda y la cuarta más a la derecha de los jugadores visibles en ese frame. Si no cumple ninguna de las dos, la detección vuelve a su clase original de YOLO y recupera el equipo de campo más cercano por color.
4. **ByteTrack** (`ByteTrack.update_with_detections`): asocia las detecciones a tracks con IDs persistentes entre frames. Usa la etiqueta de equipo como penalización adicional, combina doble señal de clase por detección (clase YOLO original + clase reetiquetada por `TeamDetector`) para permitir remapeos controlados cuando discrepan, añade consenso temporal de clase por track y, para `player`/`goalkeeper`, incorpora distancia en el campo 2D al coste de asociación. También puede penalizar cambios bruscos de tamaño de bbox: si la detección nueva difiere más de `bbox_height_ratio_threshold` en alto o más de `bbox_width_ratio_threshold` en ancho respecto al track previo, añade `bbox_size_mismatch_penalty` al coste por cada dimensión fuera de rango. La política anti-solape se aplica solo al **nacimiento** de tracks nuevos: un candidato se descarta si solapa por encima de `bytetracker.new_track_active_overlap_iou` con un track activo ya consolidado, si solapa por encima de `bytetracker.new_track_unconfirmed_overlap_iou` con un `unconfirmed` previo o si solapa por encima de `bytetracker.new_track_candidate_overlap_iou` con otro candidato nuevo del mismo frame; en ese último caso se conserva solo el de mayor confianza. La rama de `unconfirmed` ya no aplica filtros extra de deduplicación por solape: una vez nacido un tentativo, solo sigue el matching normal de ByteTrack. En la capa canónica, si ByteTrack mantiene el mismo `raw_tracker_id`, ese vínculo se conserva directamente siempre que pase un gate de continuidad; solo si falla esa continuidad la detección vuelve a competir por otro ID canónico.
5. **Seeds canónicos opcionales en punto de penalti**: si `reserve_penalty_spot_seed_players=true`, el tracker crea dos tracks semilla sintéticos de clase `player` en los puntos de penalti. No pasan por `TeamDetector`, así que no contaminan el clustering de colores ni tienen equipo asignado. Sí participan en la reasignación canónica por posición de campo, reservando dos IDs para jugadores no visibles al inicio. Mientras no absorban una detección real, también se escriben en el JSON final con `synthetic_seed=true`.
6. **Lógica especial para los IDs reservados**: esos dos IDs no exigen coincidencia `player/goalkeeper` para recuperar una detección real y no fijan equipo por color durante el tracking. Su equipo se asigna frame a frame durante el propio tracking con el modelo de roles posicionales y el defensa más cercano. Como ya se consideran porteros conocidos, no entran al Set Transformer y se etiquetan manualmente como `POR`. Si defines `tracking.expected_roles_by_team`, o pasas un `lineup_spec.json` desde la interfaz, ese once esperado se usa solo al congelar el `role` estable, no para imponer una plaza táctica distinta en cada frame. El congelado se decide con la evidencia acumulada del jugador durante la ventana de estabilización y luego se asigna una plaza esperada por equipo con la estrategia configurada en `tracking.role_stabilization_expected_roles_assignment` (`hungarian`, `greedy` o `ratio_priority`). El modo `ratio_priority` hace una foto global al llegar a `role_stabilization_window_frames`: con esa evidencia fija intenta asignar plazas esperadas por equipo, y si un rol dominante no es válido para la alineación esperada o una plaza válida ya quedó ocupada, transfiere esa masa a la siguiente plaza válida libre del ranking del jugador. Después exige un ratio acumulado mínimo para esa plaza válida y, además, un ratio mínimo en la propia plaza final. Si aún quedan plazas libres tras ese filtro, resuelve los descartes restantes con una asignación óptima entre jugadores pendientes y slots disponibles y, si todavía queda algún slot del once sin cubrir, completa esas plazas con los tracks ya congelados que seguían sin `expected_role_slot` usando la mejor combinación restante. Cuando un once esperado contiene dos `MC` o dos `DC`, tras congelar ambos slots el sistema los desdobla a `MC_IZQ/MC_DCHO` o `DC_IZQ/DC_DCHO` usando la media acumulada de distancia a las bandas hasta ese instante, en coordenadas ya orientadas por dirección de ataque. Si existe un `lineup_spec.json`, en ese mismo instante también intenta resolver `player_name` combinando `team + display_role_slot`. Los jugadores con menos de `role_stabilization_expected_roles_min_count` observaciones no compiten por plaza táctica en la fase fuerte y primero se congelan con su rol dominante de esa foto. El histórico previo ya no se backfillea: los frames anteriores mantienen la predicción original que tuvieron. Si una detección reaparece con un `raw_tracker_id` ya arrastrando otro canónico, los IDs especiales solo pueden reclamarla si ese canónico no estaba realmente activo y además la geometría favorece al ID especial.
7. **Reasignación canónica coherente con ByteTrack**: para `player/goalkeeper` con `field_position_m`, la segunda capa de IDs canónicos usa exactamente el mismo gate geométrico que ByteTrack (`field_position_match_distance_*`). No añade un suelo extra ni expansión por velocidad en esa capa, así que no puede reusar un ID final con un salto de campo mayor que el permitido por la capa base.
   - Los IDs canónicos de personas se limitan al rango `1..N_personas` (el balón no consume ese rango y siempre usa `id=0`).
   - Los IDs canónicos `tracking.referee_canonical_ids` (por defecto `23,24,25`) quedan reservados a árbitros.
   - La reabsorción de árbitros usa una lógica específica por zona de campo: `sideline_top`, `sideline_bottom` y `central`, calculadas desde `field_position_m`. Si un árbitro reaparece en la misma zona, reabsorbe ese ID reservado aunque haya habido un pequeño gap temporal.
   - En la capa canónica, un ID `referee` solo acepta detecciones de entrada cuya clase resuelta siga siendo `referee`; ya no puede reabsorber detecciones `player`.
   - El árbitro central solo puede reabsorber detecciones cuya `x` en campo caiga entre la segunda `x` más a la izquierda y la segunda más a la derecha de los `player/goalkeeper` visibles en ese frame; así se evita absorber tracks pegados a las porterías.
8. **Selección robusta del balón**: las candidatas de balón, tanto las devueltas por ByteTrack como las detecciones YOLO crudas, pasan por un gate específico de continuidad. A diferencia de `player/goalkeeper`, aquí no se aplica además el gate genérico de reasignación: se usa solo la lógica propia del balón para no perder cobertura. Se valida que el balón:
   - no salte a una posición incompatible con su trayectoria reciente;
   - no cambie de tamaño de forma abrupta entre frames;
   - y, si hay varias candidatas plausibles, se prioriza la más coherente con la posición esperada y la confianza.
   Si ninguna candidata es físicamente plausible, ese frame queda sin balón en vez de aceptar un teletransporte. Cuando la trayectoria prevista saca el balón fuera de la imagen, la búsqueda queda anclada al borde por el que salió; no se aceptan reapariciones “hacia atrás” dentro de la pantalla. Solo tras `tracking.ball.max_reassign_lost_frames` frames perdidos se permite una redetección libre por máxima confianza.
9. **Estimación de posesión online**: tras cerrar el frame, se ejecuta una heurística temporal de posesión (`TeamPossessionEstimator`) que decide `equipo + jugador` en control del balón usando distancia balón-pie, contacto estricto/flexible y señales de movimiento del balón (cambio de dirección, caída de velocidad, continuidad del portador y cambios de equipo). El resultado se inyecta en el propio `tracks` de ese frame para consumo posterior (PathCRF, visualización, análisis).

Si `profile_phases=True`, el tracker imprime por frame los tiempos de cada fase y el total, sin modificar la lógica ni el resultado del pipeline.

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
    "bbox_size":   float,              # área del bounding box en píxeles²
    "is_possession_player": bool,      # true en el jugador/portero poseedor del frame
    "ball_owning_team_id": str | None, # equipo con posesión en ese frame
    "ball_owning_player_id": int | None, # id canónico del jugador poseedor
    "player_id": int | None,           # alias del poseedor para consumidores downstream
    "possession_reason": str | None    # razón heurística (ej. start_touch, last_touch_hold)
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

## `byte_tracker.py` — `ByteTrack`

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

### Extensión: penalización por equipo

Se ha añadido un atributo `team` a cada `STrack`. Si en la primera asociación se intenta asociar una detección de un equipo distinto al del track, se añade una **penalización** (configurable vía `team_penalty` en config.yaml, por defecto 1000) a la matriz de costes IoU, haciendo esa asociación prácticamente imposible.

### Extensión: doble señal de clase y consenso por track

Cada detección de persona llega al tracking con dos clases:
- `class_yolo`: clase original de YOLO.
- `class`: clase reetiquetada por `TeamDetector`.

El matching interno aplica la penalización de clase con este patrón: si `class_team == track.class` no penaliza; si `class_team != track.class` pero `class_yolo == track.class` aplica la relajada (`class_mismatch_relaxed_penalty`); y si ninguna señal apoya al track aplica la fuerte (`class_mismatch_penalty`). Además, cada `STrack` acumula evidencia temporal (`class_vote_weight_relabel`, `class_vote_weight_yolo`) y solo cambia su clase interna cuando la nueva hipótesis supera un margen de consenso (`class_consensus_switch_margin`).

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
- `class_mismatch_penalty` (penalización fuerte por mismatch de clase en matching interno)
- `class_mismatch_relaxed_penalty` (penalización reducida cuando clase YOLO y clase reetiquetada discrepan)
- `allow_class_remap_when_signals_disagree` (habilita la vía de remapeo por clase cuando hay desacuerdo de señales)
- `class_vote_weight_relabel` / `class_vote_weight_yolo` (peso de cada señal de clase en el consenso temporal)
- `class_consensus_switch_margin` (ventaja mínima acumulada para permitir cambio de clase en el track interno)
- `shirt_color_distance_weight` (actualmente sin efecto en matching interno)
- `shirt_color_distance_gate` (parámetro legado; sin efecto mientras la penalización LAB esté desactivada)
- `bbox_size_mismatch_penalty` (penalización por cambio brusco de tamaño entre bbox previa y bbox candidata)
- `bbox_height_ratio_threshold` (desviación relativa máxima permitida en altura antes de penalizar)
- `bbox_width_ratio_threshold` (desviación relativa máxima permitida en ancho antes de penalizar)
- `field_position_match_distance_gate_m` (base del gate espacial en metros)
- `field_position_match_distance_cap_m` (tope absoluto del gate espacial acumulado)
- `field_position_match_distance_max_lost_frames` (tope de crecimiento temporal del gate)
- `field_position_match_distance_growth_mode` (`power` o `linear_decay`)
- `field_position_match_distance_lost_exponent` (si < 1, crecimiento sublineal)
- `field_position_match_distance_decay_per_frame` (en `linear_decay`, cuánto decrece cada paso por frame perdido)
- `new_track_active_overlap_iou` (IoU máxima permitida entre un track nuevo y un track activo ya consolidado)
- `new_track_unconfirmed_overlap_iou` (IoU máxima permitida entre un track nuevo y un `unconfirmed` previo)
- `new_track_candidate_overlap_iou` (IoU máxima permitida entre dos candidatos nuevos del mismo frame)
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
- `use_shirt_color_for_reassign` (actualmente sin efecto: la canonización no veta por color LAB)
- `shirt_color_reassign_distance_gate` (parámetro legado de color LAB en canonización)
- `shirt_color_reassign_distance_growth_per_frame` (parámetro legado de color LAB en canonización)
- `shirt_color_reassign_distance_cap` (parámetro legado de color LAB en canonización)
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
- `reassign_motion_growth_cap_frames` (tope de frames perdidos que se usan para extrapolar el salto permitido solo en clases sin homografía)
- `output/tracker/<video>_role_artifacts/<video>_frame_role_predictions.csv` (predicción cruda frame a frame antes del congelado estable)
- `output/tracker/<video>_role_artifacts/<video>_player_role_summary.csv` (resumen estable por track al terminar el vídeo)
- `output/tracker/<video>_role_artifacts/<video>_greedy_role_diagnostics.csv` (traza paso a paso de las métricas usadas por el greedy al congelar slots estables)
- `output/tracker/<video>_role_artifacts/<video>_role_assignment_vs_detected_pre<frame>.png` (comparativa por jugador entre distribución detectada hasta el frame de corte y posición final)
- `output/tracker/<video>_role_artifacts/<video>_<equipo>_ratio_priority_step_by_step.png` (solo en `ratio_priority`: simulación paso a paso de la asignación snapshot por equipo)
- `require_field_position_for_reassign` (si `true`, `player/goalkeeper` no hacen fallback a píxeles)
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
