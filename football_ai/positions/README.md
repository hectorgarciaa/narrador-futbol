# positions

Modulo de roles posicionales y alineaciones.

## Que hace

- construye y mantiene el dataset etiquetado de posiciones;
- entrena el modelo de roles basado en Set Transformer;
- infiere roles online durante el tracking;
- valida `lineup_spec.json` y resuelve nombres de jugadores desde la interfaz.

## Subcarpetas

- `data/`: observaciones, templates y dataset comun.
- `model/`: entrenamiento, grid search, inferencia y render.
- `pipeline/`: logica online usada por el tracking principal.

## Punto de entrada

La pieza productiva del modulo es `pipeline/phase.py`, que expone `PositionInferingPhase`.

