# experiments/positions

Utilidades y notebook para construir un dataset supervisado de rol nominal de jugador a partir de tracking proyectado al campo.

## Archivos

- `position_dataset.py`: funciones para preparar observaciones, validar etiquetas de rol, inferir orientación de ataque y construir samples con features tabulares + tensor de compañeros.
- `position_role_dataset.ipynb`: flujo end-to-end de creación del dataset.
- `position_role_workflow.py`: implementación equivalente en script (`.py`) del flujo completo (preparación, etiquetado interactivo en Jupyter y exportación).
- `set_transformer_pipeline.py`: entrenamiento e inferencia de un clasificador Set Transformer usando `base_table.csv` reconstruido por partido.

## Set Transformer para inferencia de roles

Si ya existe un dataset común etiquetado en `data/posiciones_etiquetadas/common/base_table.csv`, puedes entrenar un modelo de roles y aplicarlo al vídeo `partido_ajustado` sin pasar por el etiquetado manual otra vez.

La generación de features canoniza cada equipo en una misma vista táctica:
- si el equipo ataca hacia `+x`, se conserva `(x, y)`
- si ataca hacia `-x`, se rota 180° a `(1 - x, 1 - y)` para mantener consistente la semántica `IZQ/DER`

Entrenamiento:

```bash
cd /Users/carloscole/narrador-futbol
python -m experiments.positions.set_transformer_pipeline train \
  --project-root /Users/carloscole/narrador-futbol
```

Si cambias la canonización geométrica o cualquier feature derivada, añade:

```bash
  --rebuild-from-base-table
```

Inferencia:

```bash
cd /Users/carloscole/narrador-futbol
python -m experiments.positions.set_transformer_pipeline predict \
  --project-root /Users/carloscole/narrador-futbol \
  --model-path models/positions/set_transformer/<timestamp>/set_transformer_checkpoint.pt \
  --video-path data/partidoPrueba/partido_ajustado.mp4
```

Si usas el notebook `experiments/set_transformer.ipynb`, puedes pasar además `EXPECTED_ROLES_BY_TEAM` para imponer el once esperado de cada equipo con Hungarian sobre las probabilidades agregadas por jugador.
Ese mismo formato de listas ya puede definirse también en `config.yaml` bajo `tracking.expected_roles_by_team`.
El comando `python -m experiments.positions.set_transformer_pipeline predict` lo toma por defecto desde `config.yaml` cuando no inyectas un mapping explícito desde Python/notebook.
En el tracking principal, ese ajuste no cambia la etiqueta de cada frame: se usa solo cuando un track alcanza la ventana de estabilización y entonces congela una plaza estable por equipo a partir de la evidencia acumulada del jugador.
Para depurar esa diferencia, `track.py` exporta también `output/tracks_json/tracker/<video>_frame_role_predictions.csv` con la predicción cruda frame a frame antes del congelado estable, `output/tracks_json/tracker/<video>_player_role_summary.csv` con el resumen estable final y `output/tracks_json/tracker/<video>_greedy_role_diagnostics.csv` con las métricas usadas en cada paso del greedy.
El notebook permite elegir entre reutilizar un checkpoint ya entrenado o reentrenar el modelo antes de inferir.
La celda inicial recarga `set_transformer_pipeline.py`, así que cambios recientes del pipeline no requieren reiniciar el kernel para que se apliquen.
Además, tras la inferencia, puede renderizar automáticamente el MP4 anotado usando el `tracks_with_predicted_roles.json`.

Salidas:
- checkpoint y métricas en `models/positions/set_transformer/<timestamp>/`
- predicciones por frame en `output/predictions/positions/partido_ajustado_<timestamp>/frame_role_predictions.csv`
- resumen estable por jugador en `output/predictions/positions/partido_ajustado_<timestamp>/player_role_summary.csv`
- tracks enriquecidos con `predicted_role` en `output/predictions/positions/partido_ajustado_<timestamp>/tracks_with_predicted_roles.json`

Cuando activas esa restricción offline, el `player_role_summary.csv` conserva tanto la predicción libre (`predicted_role_unconstrained`) como la restringida (`predicted_role`), además del label del modelo que sustentó esa plaza (`matched_model_role`) y del slot esperado asignado (`expected_role_slot`).
Si sobran jugadores detectados respecto a las plazas esperadas, esos jugadores no fallan: reciben igualmente la mejor posición permitida dentro del once esperado y quedan marcados con `assignment_method = expected_roles_fallback_best_allowed`.
En ese modo, el vídeo anotado enseña el `predicted_role` estable restringido y no el `predicted_role_frame` libre.

Render de vídeo anotado:

```bash
cd /Users/carloscole/narrador-futbol
python -m experiments.positions.set_transformer_pipeline render-video \
  --project-root /Users/carloscole/narrador-futbol \
  --video-path data/partidoPrueba/partido_ajustado.mp4 \
  --tracks-path output/predictions/positions/partido_ajustado_<timestamp>/tracks_with_predicted_roles.json
```

Salida adicional:
- vídeo anotado en `output/predictions/positions/partido_ajustado_<timestamp>/partido_ajustado_roles_annotated.mp4`

Nota:
- El dataset común actual no contiene ejemplos etiquetados de `goalkeeper`, así que la inferencia asigna `POR` por heurística cuando `class_name == goalkeeper`.

## Grid search experimental del Set Transformer

El grid search está aislado en `experiments/positions/set_transformer_grid_search.py` y no modifica el pipeline productivo de `football_ai/positions`.
Prueba combinaciones de:

- número de capas ocultas del MLP del jugador objetivo: `1`, `2`, `3`
- expansión del feed-forward interno de los bloques de atención: `x2`, `x4`
- número de bloques de Set Transformer: `2`, `3`, `4`
- dropout: `0.10`, `0.15`, `0.20`

Ejecución:

```bash
cd /home/cmantill/narrador-futbol
python -m experiments.positions.set_transformer_grid_search \
  --project-root /home/cmantill/narrador-futbol
```

Salidas:

- carpeta por búsqueda en `experiments/positions/grid_search_outputs/<timestamp>/`
- subcarpeta por configuración en `runs/`
- `training_history.csv`, `metrics.json`, `config.json`, matrices de confusión y curvas de entrenamiento por configuración
- `grid_search_summary.csv` y `grid_search_summary.json` con todas las pruebas ordenables
- `best_hyperparameters.json` y `best_model.pt` con el mejor modelo según `val_macro_f1`

Para hacer una prueba rápida sin entrenar sobre todo el dataset:

```bash
python -m experiments.positions.set_transformer_grid_search \
  --project-root /Users/carloscole/narrador-futbol \
  --epochs 3 \
  --max-train-samples 2048 \
  --max-val-samples 512 \
  --max-test-samples 512
```

Tras el primer barrido, puede lanzarse el preset ampliado:

```bash
cd /home/cmantill/narrador-futbol
./.venv/bin/python -m experiments.positions.set_transformer_grid_search \
  --project-root /home/cmantill/narrador-futbol \
  --grid-preset full
```

Este preset explora 480 configuraciones combinando perfiles de anchura del modelo, regularización más fuerte, 3--5 bloques de atención, FF `x2/x4`, 1--2 capas en el MLP del jugador objetivo y dos tasas de aprendizaje.

Si se quiere evaluar la fusión de carrileros con laterales, puede activarse:

```bash
./.venv/bin/python -m experiments.positions.set_transformer_grid_search \
  --project-root /home/cmantill/narrador-futbol \
  --grid-preset best-run \
  --merge-wingbacks
```

Con `--merge-wingbacks`, las etiquetas `CI` y `CD` se remapean antes del split y del entrenamiento:

- `CI -> LI`
- `CD -> LD`

El preset `best-run` reentrena únicamente la mejor configuración encontrada en el grid ampliado anterior. Es útil para comparar rápidamente las métricas con y sin carrileros como clases independientes.

Para afinar los tres mejores modelos del grid corregido, puede usarse:

```bash
./.venv/bin/python -m experiments.positions.set_transformer_grid_search \
  --project-root /home/cmantill/narrador-futbol \
  --grid-preset refine-top3 \
  --merge-wingbacks \
  --rebuild-from-base-table
```

Este preset ejecuta tres subgrids secuenciales, uno por cada familia de modelo. En cada familia se conserva la arquitectura y se prueban valores nuevos de `learning_rate`, `weight_decay`, `dropout` y `label_smoothing`.

Resumen del barrido:

- `refine_run120_top_val`: 36 pruebas alrededor del mejor modelo por validación.
- `refine_run342_second_val`: 36 pruebas alrededor del segundo mejor modelo por validación.
- `refine_run290_third_val_best_balance`: 36 pruebas alrededor del modelo con mejor equilibrio validación/test.

En total se ejecutan 108 entrenamientos y se guardan en la misma estructura de `grid_search_outputs`.

## Uso recomendado en `.py` (sin depender del notebook)

### 1) Preparar schedule + JSON de etiquetas

```bash
cd /Users/carloscole/narrador-futbol
python -m experiments.positions.position_role_workflow prepare \
  --project-root /Users/carloscole/narrador-futbol \
  --video-filename "A1606b0e6_0 (50).mp4" \
  --label-every-seconds 6 \
  --label-window-frames 20
```

### 2) Etiquetar con UI estable desde Jupyter (solo 1 imagen: frame actual)

```python
from pathlib import Path
from experiments.positions.position_role_workflow import WorkflowConfig, create_labeler_ui

config = WorkflowConfig(
    project_root=Path("/Users/carloscole/narrador-futbol"),
    video_filename="A1606b0e6_0 (50).mp4",
    label_every_seconds=6.0,
    label_window_frames=20,
)

context, labeler, summary = create_labeler_ui(config)
print(summary)
labeler.display()
```

Notas:
- El preview se pinta en un único `widgets.Image` (no usa `matplotlib` inline), para evitar acumulación de imágenes.
- Si re-ejecutas la celda de creación de UI, la instancia anterior se cierra automáticamente.

### 3) Exportar dataset desde labels guardadas

```bash
cd /Users/carloscole/narrador-futbol
python -m experiments.positions.position_role_workflow export \
  --project-root /Users/carloscole/narrador-futbol \
  --video-filename "A1606b0e6_0 (50).mp4"
```

## Notebook mínimo

`position_role_dataset.ipynb` se ha simplificado a 2 celdas:
1. Setup de `sys.path` + imports del workflow.
2. Configuración de vídeo + creación de UI (`create_labeler_ui(config)`).

Todo el pipeline (schedule, labels JSON, UI, propagación, export y upsert a `common`) lo ejecuta `position_role_workflow.py`.

## Labels v1

`POR, CI, LI, DFC_IZQ, DFC_CENT, DFC_DER, LD, CD, MC, MI, MD, EI, ED, DC`
