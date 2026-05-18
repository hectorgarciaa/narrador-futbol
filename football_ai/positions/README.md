# positions

Submódulo que agrupa toda la lógica de posiciones/roles que antes vivía mezclada en `scripts/track.py`.

Incluye:

- inferencia online de roles posicionales durante el tracking
- asignación dinámica por equipo con doble pasada configurable (`hungarian` o `ratio_priority`)
- asignación especial de equipo para los IDs reservados de portero
- exportación de CSV de roles y artefactos de tracking
- catálogo de formaciones para la interfaz web
- validación de `lineup_spec.json`
- matching `equipo + slot estable -> nombre de jugador`

## Estructura

- `data/`: dataset y helpers de observaciones.
- `data/core.py`: observaciones y acceso a datos base.
- `data/label_templates.py`: plantillas y aplicación de etiquetas periódicas.
- `data/common_dataset.py`: mantenimiento del dataset común.
- `model/`: arquitectura y entrenamiento.
- `model/data_utils.py`: carga, normalización y splits.
- `model/training.py`: entrenamiento offline.
- `model/session.py` / `model/inference.py`: inferencia online y batch.
- `model/render.py`: render anotado opcional.
- `pipeline/`: código productivo online usado por el tracking en vivo.
- `pipeline/config.py` y `pipeline/helpers.py`: configuración y utilidades del motor online.

El punto de entrada principal dentro del pipeline de tracking es `PositionInferingPhase` en [pipeline/phase.py](pipeline/phase.py). La lógica stateful online vive en `OnlineSpecialSeedRoleAssigner` en [pipeline/online.py](pipeline/online.py).

## Flujo Online Nuevo

La asignación online ya no congela roles ni usa snapshots de estabilización. El flujo actual es:

1. El modelo predice probabilidades para los tracks activos.
2. `Tier 1`: contra los slots esperados del equipo se puede aplicar `hungarian` o `ratio_priority` a nivel frame, generando un voto limpio por track en ese frame.
3. Cada track acumula esos votos dentro de su segmento activo. Si aparece una señal fuerte de relink o un salto estructural claro, se abre un segmento nuevo vacío antes de seguir votando. Los cambios tácticos temporales de rol no rompen por sí solos el segmento.
4. `Tier 2`: para pintar el rol definitivo del frame, se toma la mayoría acumulada de cada segmento activo y se resuelve otra vez contra los slots esperados del equipo usando `hungarian` o `ratio_priority`. La resolución final se reaplica por segmento real `(track_id, segment_id)` para que varios jugadores del mismo equipo no se pisen aunque compartan el mismo contador local de segmento.

## Configuración de asignación

Dentro de `positions` en `config.yaml`:

- `frame_expected_roles_assignment_method`: `hungarian` o `ratio_priority`
- `frame_ratio_priority_min_count`
- `frame_ratio_priority_min_cumulative_ratio`
- `frame_ratio_priority_min_final_ratio`
- `segment_expected_roles_assignment_method`: `hungarian` o `ratio_priority`
- `segment_ratio_priority_min_count`
- `segment_ratio_priority_min_cumulative_ratio`
- `segment_ratio_priority_min_final_ratio`

La implementación de `ratio_priority` reutiliza la lógica histórica de prioridad por ratios acumulados. En la pasada de segmento, además mantiene una resolución geométrica lateral para desambiguar duplicados `MC/DC` cuando el lineup usa slots tipo `*_IZQ` y `*_DCHO`.

## `pipeline/lineup_spec.py`

Este módulo define las formaciones soportadas por la interfaz (`4-3-3`, `5-3-2`, `4-4-2`) y separa dos niveles:

- `ui_slots`: slots visibles para el usuario en la interfaz. Aquí se usan directamente `DC_IZQ/DC_DCHO` y `MC_IZQ/MC_DCHO` cuando la formación tiene duplicados.
- `tracking_slots`: slots que consume la pasada de Húngaro a nivel frame. Cuando hay duplicados, se traducen a la taxonomía base (`DC`, `DC` o `MC`, `MC`).

Además, `LineupSlotMatcher` resuelve el nombre del jugador una vez que el track ya tiene un slot estable:

- primero intenta el slot final del segmento (`display_role_slot`)
- luego la mayoría del segmento (`segment_majority_expected_role_slot` / `segment_majority_role`)
- y solo cae a slots base si ese rol es único en la formación

Eso evita asignaciones ambiguas de nombres mientras todavía no se han separado dos `MC` o dos `DC`.
