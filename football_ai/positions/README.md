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

El punto de entrada principal dentro del pipeline de tracking es `PositionInferingPhase` en [phase.py](phase.py). La lógica stateful que reutiliza esa fase sigue viviendo en `OnlineSpecialSeedRoleAssigner` en [tracking_roles.py](tracking_roles.py).

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
