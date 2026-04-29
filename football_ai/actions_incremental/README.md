# actions_incremental

Módulo alternativo en paralelo a `football_ai.actions` para reconstruir el flujo offline de PathCRF de forma incremental e independiente.

## Objetivo actual

- consumir paquetes frame a frame;
- mantener una representación materializada por slot (`home_1..11`, `away_1..11`, `referee_1..3`, `ball`) para reutilizarla entre frames;
- ejecutar PathCRF online sobre esa ventana incremental;
- aplicar un postprocesado causal y emitir `raw_edge` + `confirmed_action`;
- seguir en paralelo a `football_ai.actions`, sin sustituirlo todavía.

## Componentes

- `detector.py`: clase `ActionsDetector`, stateful, incremental y autosuficiente, con la lógica de adaptación integrada y caché materializada por slot.
- `runtime.py`: clase `ActionsRuntime`, ejecuta PathCRF y aplica el postprocesado causal sobre la ventana actual.
- `pathcrf_semantics.py`, `pathcrf_setpieces.py`, `pathcrf_shot.py`: reglas semánticas y heurísticas locales, sin depender de `football_ai.actions`.
- `phase.py`: clase `ActionsDetectorPhase`, wrapper `Phase` no intrusivo para el pipeline.

## Estado actual

- El detector incremental ya no depende del adaptador offline para construir el dataframe de PathCRF.
- Las reglas semánticas de set pieces y shot también viven dentro del módulo, así que `actions_incremental` es autocontenido.
- El runtime mantiene una vista materializada por slot y reutiliza esa estructura entre frames.
- La actualización es causal: `0..n-1` quedan fijados y cada `update(frame_n)` solo añade la fila nueva y expulsa la más antigua si la ventana está llena.
- La salida del phase se publica en `clean.actions_incremental` y `trace.actions_incremental`.
- Las asignaciones `track_id -> slot` son persistentes dentro de la ventana y solo se liberan cuando el track desaparece del estado activo.
