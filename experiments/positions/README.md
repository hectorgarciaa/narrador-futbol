# experiments/positions

Utilidades y notebook para construir un dataset supervisado de rol nominal de jugador a partir de tracking proyectado al campo.

## Archivos

- `position_dataset.py`: funciones para preparar observaciones, validar etiquetas de rol, inferir orientación de ataque y construir samples con features tabulares + tensor de compañeros.
- `position_role_dataset.ipynb`: flujo end-to-end de creación del dataset.

## Pipeline del notebook

1. Carga vídeos de `data/partidosPosiciones/`.
2. Carga tracks por vídeo (`<video_sanitizado>_tracks.json` o `tracks.json` legacy). `scripts/track.py` ya genera ambos formatos.
3. Construye observaciones por jugador/frame (`x, y, x_m, y_m, team_id, bbox, confidence`).
4. Selecciona un frame para etiquetar IDs.
5. Aplica `ROLE_MAP` manual por equipo/jugador.
6. Reorienta coordenadas para un sistema común de ataque.
7. Genera muestras de jugador objetivo + compañeros (tensor + máscara).
8. Exporta CSV/NPZ/JSON a `output/datasets/positions/<match_id>_<timestamp>/`.

## Labels v1

`POR, LI, DFC_IZQ, DFC_DER, LD, MC, MI, MD, EI, ED, DC`
