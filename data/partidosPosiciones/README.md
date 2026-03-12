# data/partidosPosiciones

Clips de vídeo usados para el flujo experimental de dataset de roles posicionales (`experiments/positions`).

## Uso

1. Genera tracks para cada clip con `scripts/track.py` (con `tracking.use_field_positions=true` en `config.yaml`), por ejemplo:
   - `python scripts/track.py video_test_1`
   - `python scripts/track.py video_test_29`
   - o en lote: `python scripts/track_partidos_posiciones.py` (usa plan fijo de colores/equipos por `video_test_*` y salta los vídeos que ya tengan `*_tracks.json`)
   - para forzar reproceso de todos: `python scripts/track_partidos_posiciones.py --force`
2. Guarda el JSON de tracks por vídeo en:
   - `output/tracks_json/tracker/<video_sanitizado>_tracks.json`
   - resumen por vídeo: `output/tracks_json/tracker/<video_sanitizado>_summary.json`
   - dataset acumulado de métricas: `output/datasets/positions/common/tracking_metrics.csv`
3. Ejecuta el notebook:
   - `jupyter lab experiments/positions/position_role_dataset.ipynb`

## Convención de nombres del JSON de tracks

El notebook sanitiza el nombre del vídeo para encontrar su tracks JSON:
- `test (1).mp4` -> `test_1_tracks.json`
- `test (11).mp4` -> `test_11_tracks.json`

Si no encuentra el archivo nombrado, usa `output/tracks_json/tracker/tracks.json` como fallback legacy.
