# `football_ai.bytetrack`

Fase desacoplada de asociación multi-objeto basada en ByteTrack.

## Responsabilidad

Este módulo vive entre `IDENTIFICATION` y la capa de tracking canónico:

`IDENTIFICATION.clean` → `ByteTrackPhase.track_packet(...)` → packet `BYTETRACK` → tracking canónico

- `clean`: publica solo las señales que consume `CANONICALTRACK`: `det_id`, `bbox_xyxy`,
  `confidence`, `class_name`, `field_positions_m`, `ground_points_image_original`
  y `tracked_detections`.
- `trace`: en runtime normal expone solo un resumen agregado y la alineación básica
  track/detección; `detection_debug` y `matching_debug` solo se construyen cuando
  `tracking.execution_mode=debug`.

## Piezas principales

- `byte_tracker.py`: fachada pública y estado principal del tracker.
- `phase.py`: adaptador de fase que consume/produce `phase_packet`.
- `association_costs.py`: builder común de `base_cost`, `feasible_mask` y métricas por pareja.
- `detection_metadata.py`: construcción de detecciones temporales y consenso de clase/equipo.
- `pipeline_steps.py`: solver común y los 6 subpasos del frame (`high_iou`, `low_iou`, `unconfirmed_iou`, `high_bbox`, `low_bbox`, `unconfirmed_bbox`).
- `new_track_filter.py`: filtro anti-solape para nacimiento de tracks nuevos.
- `debug_tools.py`: trazas homogéneas de matching para las 6 fases.
- `utils.py`: helpers geométricos y operaciones sobre listas de tracks.

## Baseline actual

Tras la auditoría comparativa larga sobre `ucl_30s` (`750` frames, ejecución secuencial), el baseline recomendado del módulo queda con:

- `track_activation_threshold = 0.10`
- `low_conf_threshold = 0.01`
- `bbox_center_distance_gate_px = 90.0`
- `bbox_center_distance_gate_max_lost_frames = 4`
- `bbox_center_distance_gate_cap_px = 360.0`

Ese ajuste (`tight_bbox_gate`) empató en cobertura canónica con `higher_activation_tight_bbox`, pero ganó por un margen pequeño en switches, relinks y fragmentación raw.

## Runtime vs auditoría

El módulo distingue dos caminos:

- **runtime normal**: calcula solo `base_cost`, `feasible_mask` y el coste activo de la subfase.
  No serializa `detection_debug` ni `matching_debug`, evita construir `pair_metrics`
  forenses completos y solo calcula métricas de `bbox_center` cuando la subfase activa
  realmente las necesita.
- **audit/debug**: activa la traza homogénea completa por detección y por subfase para análisis
  forense del matching.

Además, el solver de asignación se resuelve de forma interna: usa `lap` cuando está disponible
y cae a `scipy.optimize.linear_sum_assignment` como fallback. No se expone como hiperparámetro.

## Contrato de entrada

`ByteTrackPhase` asume el contrato publicado por `IDENTIFICATION.clean` y no añade una capa
extra de validación defensiva en runtime. La fase trabaja dando por hechas, alineadas por frame,
estas señales:

- `det_id`
- `bbox_xyxy`
- `confidence`
- `class_name`
- `class_td`
- `team`
- `distances`
- `shirt_color`
- `bbox_size`
- `field_positions_m`
- `ground_points_image_original`

Si el contrato upstream se rompe, el fallo aparecerá de forma natural en el punto donde se use la
señal inconsistente; el módulo prioriza el camino normal del pipeline sobre validaciones extras.

## Estado del tracker

La consolidación final del frame conserva ahora el histórico acumulado de `removed_tracks` y
elimina de `lost_tracks` los tracks que acaban de pasar a `Removed` en ese mismo frame. Esto
evita perder histórico y corrige un caso donde un track eliminado podía seguir figurando en
`lost_tracks` hasta el frame siguiente.
