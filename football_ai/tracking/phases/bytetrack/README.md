# `football_ai.bytetrack`

Fase desacoplada de asociación multi-objeto basada en ByteTrack.

## Responsabilidad

Este módulo vive entre `IDENTIFICATION` y la capa de tracking canónico:

`IDENTIFICATION.clean` → `ByteTrackPhase.track_packet(...)` → packet `BYTETRACK` → tracking canónico

- `clean`: conserva todas las señales de entrada y añade `tracker_id`, `class_tracker`, `tracked_mask`, `tracked_count` y `tracked_detections`.
- `trace`: expone `detection_debug`, `matching_debug` y un resumen agregado.

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

Ese ajuste (`tight_bbox_gate`) empató en cobertura canónica con `higher_activation_tight_bbox`, pero ganó por un margen pequeño en switches, relinks y fragmentación raw.
