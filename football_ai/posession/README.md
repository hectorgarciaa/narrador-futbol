# posession

Fase de estimacion heuristica de posesion.

## Que hace

- consume `tracks_frame` del tracking canonico;
- decide que equipo y jugador tienen la posesion;
- enriquece los tracks con metadatos de posesion;
- publica un packet reutilizable por roles y comentarios.

## Archivos clave

- `estimator.py`: heuristica principal.
- `phase.py`: integracion como fase del pipeline.

