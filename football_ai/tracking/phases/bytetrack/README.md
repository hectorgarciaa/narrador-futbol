# `football_ai.bytetrack`

Fase desacoplada de asociación multi-objeto basada en ByteTrack.

## Responsabilidad

Este módulo vive entre `IDENTIFICATION` y la capa de tracking canónico:

`IDENTIFICATION.clean` → `ByteTrackPhase.track_packet(...)` → packet `BYTETRACK` → tracking canónico

- `clean`: conserva todas las señales de entrada y añade `tracker_id`, `class_tracker`, `tracked_mask`, `tracked_count` y `tracked_detections`.
- `trace`: expone `detection_debug`, `unconfirmed_association_debug` y un resumen agregado.

## Piezas principales

- `byte_tracker.py`: fachada pública y estado principal del tracker.
- `phase.py`: adaptador de fase que consume/produce `phase_packet`.
- `association_costs.py`: costes de asociación, gates y penalizaciones.
- `detection_metadata.py`: construcción de detecciones temporales y consenso de clase/equipo.
- `pipeline_steps.py`: pasos del algoritmo por frame (`high`, `low`, `unconfirmed`, activación y cierre de estado).
- `new_track_filter.py`: filtro anti-solape para nacimiento de tracks nuevos.
- `debug_tools.py`: trazas y diagnóstico del matching de `unconfirmed`.
- `utils.py`: helpers geométricos y operaciones sobre listas de tracks.
