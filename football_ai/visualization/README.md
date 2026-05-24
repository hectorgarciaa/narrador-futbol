# visualization

Modulo de renderizado de videos anotados.

## Que hace

- dibuja bounding boxes, IDs, equipos, roles y posesion;
- genera el mosaico 2x2 del tracking;
- renderiza salidas visuales para PathCRF y otras fases de analisis.

## Archivos clave

- `drawer.py`: renderer principal del tracking.
- `pathcrf_drawer.py`: visualizacion de eventos y slots de PathCRF.
- `simple_drawer.py`: utilidades de dibujo mas ligeras.

