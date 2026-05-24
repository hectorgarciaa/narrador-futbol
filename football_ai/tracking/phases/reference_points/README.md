# reference_points

Fase de calibracion y proyeccion al campo.

## Que hace

- estima una homografia imagen -> campo con PnLCalib;
- calcula el punto de apoyo de cada deteccion;
- proyecta cada objeto a coordenadas metricas;
- informa si la geometria es valida para el tracking.

## Archivos clave

- `projector.py`: proyector principal.
- `estimation.py`: inferencia de PnLCalib.
- `quality.py`: validacion de la homografia.
- `geometry.py`: utilidades geometricas.
- `runtime_loader.py`: carga del runtime externo.
- `phase.py`: integracion como fase `REFERENCE_POINTS`.

