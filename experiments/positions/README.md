# experiments/positions

Utilidades y notebook para construir un dataset supervisado de rol nominal de jugador a partir de tracking proyectado al campo.

## Archivos

- `position_dataset.py`: funciones para preparar observaciones, validar etiquetas de rol, inferir orientación de ataque y construir samples con features tabulares + tensor de compañeros.
- `position_role_dataset.ipynb`: flujo end-to-end de creación del dataset.

## Pipeline del notebook

1. Carga vídeos de `data/partidosPosiciones/`.
2. Carga tracks por vídeo (`<video_sanitizado>_tracks.json` o `tracks.json` legacy). `scripts/track.py` ya genera ambos formatos.
3. Construye observaciones por jugador/frame (`x, y, x_m, y_m, team_id, bbox, confidence`).
4. Genera plantilla JSON de etiquetado temporal cada 6 segundos (`0, 6, 12, ...`) en `output/datasets/positions/labels/`.
5. Rellena `role_label` por (`frame_id`, `team_id`, `player_id`) y carga ese JSON.
6. Reorienta coordenadas para un sistema común de ataque.
7. Genera muestras de jugador objetivo + compañeros (tensor + máscara) usando solo filas etiquetadas.
8. Exporta CSV/NPZ/JSON por ejecución a `output/datasets/positions/<match_id>_<timestamp>/`.
9. Añade automáticamente el resultado al dataset común acumulado en `output/datasets/positions/common/`.

## Labels v1

`POR, LI, DFC_IZQ, DFC_DER, LD, MC, MI, MD, EI, ED, DC`
