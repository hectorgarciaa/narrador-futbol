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

- El `tracks.json` actual no proyecta el balón al campo, así que `ball_x/ball_y` es una aproximación basada en el portador inferido, no una triangulación física exacta.
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
