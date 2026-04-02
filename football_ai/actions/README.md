# actions

Utilidades para preparar datos de acciones y tender un puente entre el tracking del proyecto y repositorios externos de detección de acciones, como `PathCRF`.

## `pathcrf_adapter.py`

Convierte un `tracks.json` del tracker actual a un parquet ancho compatible con el formato que consume `PathCRF`.

### Qué hace

- fusiona `player` y `goalkeeper` en 22 slots fijos (`home_1..11`, `away_1..11`);
- conserva 3 árbitros en slots `referee_1..3`;
- genera `ball_x/ball_y` y columnas de estado (`frame_id`, `period_id`, `timestamp`, `phase_id`, `episode_id`, `ball_state`, `ball_owning_team_id`, `player_id`);
- interpola huecos internos por coordenadas de campo con interpolación lineal;
- rellena extremos con arrastre (`ffill/bfill`);
- si un slot nunca aparece en el clip, sintetiza una trayectoria razonable a partir de una plantilla simple de formación y la desplaza según el centro del equipo visible en ese frame;
- para el balón, usa una heurística sencilla de portador: jugador/portero visible más cercano al centro del bbox del balón; si no hay candidato plausible, mantiene el último portador válido.

### Limitaciones explícitas

- El `tracks.json` ya puede incluir `field_position_m` del balón cuando existe proyección válida; aun así, `ball_x/ball_y` puede seguir recurriendo al portador inferido o a fallback temporal cuando esa proyección falte o sea poco fiable.
- Si faltan jugadores durante todo el clip, el adaptador crea slots sintéticos; eso sirve para estructurar PathCRF, pero no equivale a tracking real.
- `phase_id`, `episode_id` y `ball_state` salen en esta primera fase como una única secuencia viva (`1`, `1`, `"alive"`). Más adelante se puede endurecer con segmentación real.

### CLI

```bash
python scripts/actions/convert_tracks_to_pathcrf.py output/tracks_json/tracker/partido_corto_tracks.json
```

Salida por defecto:

```text
football_ai/actions/pathcrf/data/narrador/tracking_processed/<video>.parquet
football_ai/actions/pathcrf/data/narrador/tracking_processed/<video>.summary.json
```

## `pathcrf_wrapper.py`

Wrapper de inferencia para reutilizar el repo clonado de `PathCRF` desde este proyecto sin tener que invocarlo a mano.

### Qué hace

- carga el checkpoint desde `football_ai/actions/repo/pathcrf/saved/<trial>/model/`;
- lee el parquet ancho ya convertido o convierte primero un `tracks.json`;
- ejecuta `PathCRF` sobre ese tracking y exporta:
  - secuencia de aristas activa por frame (`*_edge_sequence.parquet`);
  - eventos detectados (`*_events.parquet`);
  - salidas macro (`*_macro_prev.parquet`, `*_macro_next.parquet`) cuando existen;
  - resumen de ejecución (`*_summary.json`);
- opcionalmente llama al drawer 2D específico de PathCRF para generar un MP4 del campo con la arista activa.

### Nota de compatibilidad

Los checkpoints `set_*` incluidos en el repo clonado (`trial=120` por defecto) se pueden cargar aunque en la `venv` no esté instalado `torch_geometric`: el wrapper inyecta un stub mínimo porque ese paquete solo se importa de forma global en el repo upstream. Si se usa un checkpoint `agent_model=gat`, entonces sí hace falta `torch_geometric` real.

### Script recomendado

```bash
python scripts/actions/run_pathcrf.py video_prueba_corto
```

También acepta un `tracks.json` directo:

```bash
python scripts/actions/run_pathcrf.py output/tracks_json/tracker/partido_corto_tracks.json
```

Salida típica:

```text
output/actions/pathcrf/<video>/<video>_tracking.parquet
output/actions/pathcrf/<video>/<video>_edge_sequence.parquet
output/actions/pathcrf/<video>/<video>_events.parquet
output/actions/pathcrf/<video>/<video>_summary.json
output/actions/pathcrf/<video>/<video>_pitch_pathcrf.mp4
```
