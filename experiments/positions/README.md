# experiments/positions

Experimentos para construir el dataset de roles y validar el modelo posicional.

## Que incluye

- `position_role_dataset.ipynb`: flujo guiado de etiquetado y exportacion.
- `position_role_workflow.py`: version en script del mismo flujo.
- `set_transformer_grid_search.py`: barridos de hiperparametros del modelo.

## Objetivo

Tomar tracks con `field_position_m`, etiquetar roles reales y generar datos para entrenar el clasificador usado por `football_ai.positions`.

