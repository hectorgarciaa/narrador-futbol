# football_ai

Paquete principal del proyecto. Reune la logica reutilizable de tracking, acciones, comentarios, posesion, roles y visualizacion.

## Modulos

- [core](core/README.md): configuracion, logging, packets y serializacion.
- [tracking](tracking/README.md): orquestacion del pipeline de tracking.
- [actions](actions/README.md): adaptador e inferencia de acciones con PathCRF.
- [commentaries](commentaries/README.md): generacion de comentarios y audio.
- [positions](positions/README.md): dataset, modelo e inferencia online de roles.
- [posession](posession/README.md): estimacion heuristica de posesion.
- [evaluation](evaluation/README.md): metricas y comparativas.
- [visualization](visualization/README.md): render de videos anotados.
- `pipeline/`: utilidades de rutas, persistencia y ejecucion del pipeline principal.

## Idea general

Los scripts de `scripts/` y la interfaz de `interfaz/` consumen este paquete. La carpeta `experiments/` se usa para pruebas y analisis, no como runtime principal.

