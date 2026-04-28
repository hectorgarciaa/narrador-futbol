# positions

Submódulo que agrupa toda la lógica de posiciones/roles que antes vivía mezclada en `scripts/track.py`.

Incluye:

- inferencia online de roles posicionales durante el tracking
- asignación dinámica por equipo con doble pasada de Húngaro
- asignación especial de equipo para los IDs reservados de portero
- exportación de CSV de roles y PNG diagnósticos
- catálogo de formaciones para la interfaz web
- validación de `lineup_spec.json`
- matching `equipo + slot estable -> nombre de jugador`

## Estructura

- `data/`: dataset y helpers de observaciones.
- `model/`: arquitectura y entrenamiento.
- `pipeline/`: código productivo online usado por el tracking en vivo.

El punto de entrada principal dentro del pipeline de tracking es `PositionInferingPhase` en [pipeline/phase.py](pipeline/phase.py). La lógica stateful online vive en `OnlineSpecialSeedRoleAssigner` en [pipeline/assigner.py](pipeline/assigner.py).

## Flujo Online Nuevo

La asignación online ya no congela roles ni usa snapshots de estabilización. El flujo actual es:

1. El modelo predice probabilidades para los tracks activos.
2. `Tier 1`: se aplica Húngaro a nivel frame contra los slots esperados del equipo y eso genera un voto limpio por track en ese frame.
3. Cada track acumula esos votos dentro de su segmento activo. Si aparece una señal fuerte de relink o salto estructural, se abre un segmento nuevo vacío.
4. `Tier 2`: para pintar el rol definitivo del frame, se toma la mayoría acumulada de cada segmento activo y se vuelve a ejecutar Húngaro contra los slots esperados del equipo.

## `pipeline/lineup_spec.py`

Este módulo define las formaciones soportadas por la interfaz (`4-3-3`, `5-3-2`, `4-4-2`) y separa dos niveles:

- `ui_slots`: slots visibles para el usuario en la interfaz. Aquí se usan directamente `DC_IZQ/DC_DCHO` y `MC_IZQ/MC_DCHO` cuando la formación tiene duplicados.
- `tracking_slots`: slots que consume la pasada de Húngaro a nivel frame. Cuando hay duplicados, se traducen a la taxonomía base (`DC`, `DC` o `MC`, `MC`).

Además, `LineupSlotMatcher` resuelve el nombre del jugador una vez que el track ya tiene un slot estable:

- primero intenta el slot final del segmento (`display_role_slot`)
- luego la mayoría del segmento (`segment_majority_expected_role_slot` / `segment_majority_role`)
- y solo cae a slots base si ese rol es único en la formación

Eso evita asignaciones ambiguas de nombres mientras todavía no se han separado dos `MC` o dos `DC`.
