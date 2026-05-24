# core

Infraestructura comun del proyecto.

## Responsabilidad

- cargar `config.yaml`;
- centralizar logging;
- definir la base de fases y packets del pipeline;
- convertir estructuras Python/NumPy a formatos serializables.

## Archivos clave

- `config.py`: acceso a configuracion y rutas.
- `logger.py`: inicializacion de logs.
- `phase.py`: clase base para fases.
- `phase_packets.py`: nombres y helpers de packets.
- `serialization.py`: conversion a JSON seguro.

