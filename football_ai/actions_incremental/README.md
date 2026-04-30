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
- El detector usa un flujo online único (sin rutas offline): mapeo determinista `canonical_id -> slot` y actualización causal frame a frame.
- El mapeo actual es fijo: `1 -> home_1`, `2 -> away_1`, `3..12 -> home_2..11`, `13..22 -> away_2..11`, `23..25 -> referee_1..3`.
- `runtime.py`: clase `ActionsRuntime`, ejecuta PathCRF y aplica el postprocesado causal sobre la ventana actual.
- `pathcrf_semantics.py`, `pathcrf_setpieces.py`, `pathcrf_shot.py`: reglas semánticas y heurísticas locales, sin depender de `football_ai.actions`.
- `phase.py`: clase `ActionsDetectorPhase`, wrapper `Phase` no intrusivo para el pipeline.

## Estado actual

- El detector incremental ya no depende del adaptador offline para construir el dataframe de PathCRF.
- Las reglas semánticas de set pieces y shot también viven dentro del módulo, así que `actions_incremental` es autocontenido.
- El runtime mantiene una vista materializada por slot y reutiliza esa estructura entre frames.
- La actualización es causal: `0..n-1` quedan fijados y cada `update(frame_n)` solo añade la fila nueva y expulsa la más antigua si la ventana está llena.
- El warmup de slots ya no arrastra la primera observación real contra la seed/template: cuando aparece por primera vez un slot real, ese frame se ancla directamente a la observación y las derivadas se reinician.
- El balón incremental replica la semántica del legacy y se exporta vacío (`NaN`) para no introducir una trayectoria sintética distinta a la del adaptador offline.
- La salida del phase se publica en `clean.actions_incremental` y `trace.actions_incremental`.
- Las asignaciones `track_id -> slot` ya no dependen de heurísticas espaciales ni de inferencia de lado: salen directamente del ID canónico activo dentro de la ventana.

## Debug recomendado

Para comparar el comportamiento del runtime incremental contra el pipeline legacy y aislar si la deriva nace en las entradas por slot o en la inferencia/postproceso, usa:

```bash
python scripts/actions/compare_pathcrf_modes.py output/tracks_json/tracker/partido_corto_tracks.json
```

La salida deja tres vistas separadas:
- `offline/`: adaptador histórico + PathCRF legacy;
- `incremental/`: detector/runtime incremental actual;
- `checkpoints/`: replay del legacy sobre snapshots acumulados en los frames donde el incremental dispara inferencia.

Además exporta diffs por frame/slot y error frente a observaciones reales en `compare/`, lo que ayuda a detectar problemas típicos del warmup causal: slots pegados demasiado tiempo a la seed/template, suavizado excesivo o aristas divergentes aunque el tracking ya parezca continuo visualmente.
