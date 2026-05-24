# identification

Fase de identificacion de equipo y relabelado de clase.

## Que hace

- extrae color de camiseta;
- asigna equipo por clustering en espacio LAB;
- decide la clase final usada por tracking (`class_td`);
- enriquece cada deteccion con color, distancias y tamano.

## Archivos clave

- `shirt_detector.py`: extraccion de color por crop.
- `team_color_model.py`: modelo de colores de equipo.
- `team_detector.py`: decision final por deteccion.
- `phase.py`: integracion como fase `IDENTIFICATION`.

