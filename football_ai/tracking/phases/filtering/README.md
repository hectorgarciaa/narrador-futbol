# filtering

Fase de filtrado geometrico posterior a la proyeccion al campo.

## Que hace

- consume la salida de `REFERENCE_POINTS`;
- descarta detecciones fuera del campo cuando la homografia es fiable;
- conserva todas las detecciones si la geometria no es usable;
- deja un packet limpio para identificacion y tracking.

## Archivos clave

- `post_projection.py`: logica principal de filtrado.
- `phase.py`: integracion como fase `FILTERING`.

