# positions

Submódulo que agrupa toda la lógica de posiciones/roles que antes vivía mezclada en `scripts/track.py`.

Incluye:

- inferencia online de roles posicionales durante el tracking
- estabilización táctica por equipo (`hungarian`, `greedy`, `ratio_priority`)
- asignación especial de equipo para los IDs reservados de portero
- exportación de CSV de roles y PNG diagnósticos
- catálogo de formaciones para la interfaz web
- validación de `lineup_spec.json`
- matching `equipo + slot estable -> nombre de jugador`

El punto de entrada principal es `OnlineSpecialSeedRoleAssigner` en [tracking_roles.py](tracking_roles.py).

## Modelo operativo de roles

El pipeline principal usa el checkpoint seleccionado en el último barrido experimental:

```text
models/positions/20260427_002133/best_model.pt
```

Ese modelo se entrenó reconstruyendo las muestras desde `base_table.csv`, con carrileros fusionados antes del entrenamiento:

- `CI -> LI`
- `CD -> LD`

Por tanto, la taxonomía aprendida por el modelo queda en 11 clases:

```text
LI, DFC_IZQ, DFC_CENT, DFC_DER, LD, MC, MI, MD, EI, ED, DC
```

Las alineaciones y `expected_roles_by_team` deben usar directamente `LI` y `LD`; `CI` y `CD` solo se mantienen como etiquetas históricas de dataset para el grid search con `--merge-wingbacks`.

El código productivo carga checkpoints con la misma arquitectura que los experimentos: MLP configurable para el jugador objetivo, bloques Set Transformer configurables, expansión configurable en el feed-forward interno y `label_smoothing` en la función de pérdida cuando se entrena desde este módulo.

El entrenamiento desde CLI acepta los mismos parámetros principales:

```bash
./.venv/bin/python -m football_ai.positions.set_transformer_pipeline train \
  --project-root /home/cmantill/narrador-futbol \
  --rebuild-from-base-table \
  --objective-num-layers 1 \
  --ff-expansion 2 \
  --num-set-blocks 5 \
  --set-hidden-dim 256 \
  --objective-hidden-dim 256 \
  --teammate-embed-dim 128 \
  --fusion-hidden-dim 256 \
  --dropout 0.27 \
  --label-smoothing 0.05 \
  --learning-rate 0.0002 \
  --weight-decay 0.0015
```

El grid search usado para optimizar este modelo también está disponible como módulo del paquete:

```bash
./.venv/bin/python -m football_ai.positions.set_transformer_grid_search \
  --project-root /home/cmantill/narrador-futbol \
  --grid-preset refine-top3 \
  --merge-wingbacks \
  --rebuild-from-base-table
```

Por defecto, ese comando guarda nuevos barridos en `models/positions/grid_search_outputs/`.

Cuando ya se ha elegido una configuración, se puede entrenar el modelo final con
`train+val` y reservar `test` únicamente para la comprobación final:

```bash
./.venv/bin/python -m football_ai.positions.set_transformer_grid_search \
  --project-root /home/cmantill/narrador-futbol \
  --final-train-val \
  --rebuild-from-base-table
```

Este modo toma por defecto los hiperparámetros de
`models/positions/20260427_002133/best_hyperparameters.json`, usa su
`best_epoch` como número fijo de epochs y guarda los artefactos en
`models/positions/final_train_val/`. No utiliza el conjunto de test para early
stopping ni para seleccionar epochs.

## `lineup_spec.py`

Este módulo define las formaciones soportadas por la interfaz (`4-3-3`, `5-3-2`, `4-4-2`) y separa dos niveles:

- `ui_slots`: slots visibles para el usuario en la interfaz. Aquí se usan directamente `DC_IZQ/DC_DCHO` y `MC_IZQ/MC_DCHO` cuando la formación tiene duplicados.
- `tracking_slots`: slots que consume la lógica de estabilización online. Cuando hay duplicados, se traducen a la taxonomía base (`DC`, `DC` o `MC`, `MC`) y luego el tracker vuelve a desdoblarlos con la geometría lateral.

Además, `LineupSlotMatcher` resuelve el nombre del jugador una vez que el track ya tiene un slot estable:

- primero intenta `segment_majority_expected_role_slot`
- luego `segment_majority_role`
- primero intenta `display_role_slot`
- luego `expected_role_slot`
- y solo usa `predicted_role`/`predicted_role_frame` si ese slot es único en la formación

Eso evita asignaciones ambiguas de nombres mientras todavía no se han separado dos `MC` o dos `DC`.
