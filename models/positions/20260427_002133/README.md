# Modelo posicional 20260427_002133

Checkpoint operativo del Set Transformer de roles futbolísticos.

Contenido:

- `best_model.pt`: checkpoint cargado por el pipeline principal.
- `best_hyperparameters.json`: hiperparámetros seleccionados.
- `dataset_summary.json`: resumen del dataset usado.
- `grid_definition.json`: definición del barrido.
- `grid_search_summary.csv` y `grid_search_summary.json`: métricas agregadas de la optimización.

No se incluyen las carpetas `runs/` de cada entrenamiento para evitar duplicar artefactos pesados.

Este modelo fusiona carrileros con laterales antes del entrenamiento:

- `CI -> LI`
- `CD -> LD`

La taxonomía final es:

```text
LI, DFC_IZQ, DFC_CENT, DFC_DER, LD, MC, MI, MD, EI, ED, DC
```
