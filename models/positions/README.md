# models/positions

Modelos entrenados para tareas posicionales.

- `20260427_002133/`: modelo operativo actual del clasificador de roles basado en Set Transformer. Incluye el checkpoint seleccionado, resumen del dataset, definición del grid search y comparativas agregadas, pero no incluye las carpetas `runs/` de cada entrenamiento.
- `grid_search_outputs/`: salida por defecto de nuevos barridos lanzados con `python -m football_ai.positions.set_transformer_grid_search`.
- `set_transformer/`: runs históricos del clasificador de roles.
