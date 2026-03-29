# positions

Submódulo que agrupa toda la lógica de posiciones/roles que antes vivía mezclada en `scripts/track.py`.

Incluye:

- inferencia online de roles posicionales durante el tracking
- estabilización táctica por equipo (`hungarian`, `greedy`, `ratio_priority`)
- asignación especial de equipo para los IDs reservados de portero
- exportación de CSV de roles y PNG diagnósticos

El punto de entrada principal es `OnlineSpecialSeedRoleAssigner` en [tracking_roles.py](tracking_roles.py).
