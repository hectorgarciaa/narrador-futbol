# actions

Utilidades para preparar datos de acciones y tender un puente entre el tracking del proyecto y repositorios externos de detección de acciones, como `PathCRF`.

## `pathcrf_adapter.py`

Convierte un `tracks.json` del tracker actual a un parquet ancho compatible con el formato que consume `PathCRF`.

### Qué hace

- fusiona `player` y `goalkeeper` en 22 slots fijos (`home_1..11`, `away_1..11`);
- conserva 3 árbitros en slots `referee_1..3`;
- genera columnas de estado (`frame_id`, `period_id`, `timestamp`, `phase_id`, `episode_id`, `ball_state`, `ball_owning_team_id`, `player_id`) y deja `ball_x/ball_y` vacío de forma intencionada;
- interpola huecos internos por coordenadas de campo con interpolación lineal;
- asigna los slots de jugadores por ajuste espacial a una plantilla base de equipo para mantener una semántica más estable de `home_1..11` y `away_1..11`;
- suaviza temporalmente las trayectorias exportadas con mediana móvil, Savitzky-Golay y limitación de jitter antes de recalcular velocidades;
- rellena extremos con arrastre (`ffill/bfill`);
- si un slot nunca aparece en el clip, sintetiza una trayectoria razonable a partir de una plantilla simple de formación y la desplaza según el centro del equipo visible en ese frame;
- no exporta señal de balón usable en `ball_x/ball_y`, para evitar contaminar PathCRF con proyecciones pobres del balón;
- por defecto **no** exporta `player_id` ni `ball_owning_team_id` como señal de posesión hacia PathCRF, para no contaminar la inferencia con una heurística ruidosa;
- ignora seeds sintéticos y observaciones fantasma (`bbox=None`, `confidence<=0`) al construir slots para PathCRF.

### Limitaciones explícitas

- `ball_x/ball_y` queda vacío siempre en este adaptador; por tanto, cualquier visualización o postproceso dependiente del balón debe tratar esa ausencia explícitamente.
- Si faltan jugadores durante todo el clip, el adaptador crea slots sintéticos; eso sirve para estructurar PathCRF, pero no equivale a tracking real.
- `phase_id`, `episode_id` y `ball_state` salen en esta primera fase como una única secuencia viva (`1`, `1`, `"alive"`). Más adelante se puede endurecer con segmentación real.
- Si el clip ya es un tramo continuo corto de juego vivo, mantener un único `episode_id` es intencional; no se corta artificialmente sin una heurística fiable de reinicio.

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
  - eventos base detectados (`*_events.parquet`);
  - eventos semánticos refinados (`*_events_semantic.parquet`) con:
    - relabel de inicios de episodio a `corner`, `throw_in` y `goalkick`;
    - heurística adaptada de `shot` sobre eventos `kick` en zona de remate;
    - columnas de trazabilidad `event_type_raw`, `event_type_semantic`, `semantic_source` y puntuaciones de tiro;
  - JSON enriquecido para comentarios (`*_commentary_events.json`) cuando también existe `tracks.json`;
  - salidas macro (`*_macro_prev.parquet`, `*_macro_next.parquet`) cuando existen;
  - resumen de ejecución (`*_summary.json`);
- opcionalmente llama al drawer específico de PathCRF para generar un MP4:
  - sobre el broadcast real con `bbox` reales + mini-mapa 2D si dispone de vídeo y `tracks.json`;
  - o en modo 2D puro si solo hay parquet.

### Postproceso semántico añadido en este repo

- `pathcrf_setpieces.py`: reclasifica el primer evento de cada `episode_id` a `corner`, `throw_in` o `goalkick` con reglas geométricas sobre el campo;
- `pathcrf_shot.py`: adapta la heurística upstream de tiro al flujo local basado en `kick/control/out`, manteniendo puntuaciones y flags auxiliares;
- `pathcrf_commentary.py`: invierte `pathcrf_id -> track_id` con `person_slot_assignments`, recupera nombre/equipo/posición desde `tracks.json` y genera un JSON rico listo para pasar después a Gemma.

El JSON de comentarios conserva tanto el evento semántico enriquecido como un subpayload `commentary_event` listo para la fase LLM. Si el actor de PathCRF cae en un slot sintético o no se puede revertir a un `track_id` real, el evento se marca como no listo para comentario y se documenta el `skip_reason`.

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
output/actions/pathcrf/<video>/<video>_events_semantic.parquet
output/actions/pathcrf/<video>/<video>_commentary_events.json
output/actions/pathcrf/<video>/<video>_summary.json
output/actions/pathcrf/<video>/<video>_pitch_pathcrf.mp4
```
