# interfaz

Interfaz web ligera para preparar una alineacion y lanzar el tracking.

## Que hace

- recoge nombres de equipos, colores y formacion;
- valida el `lineup_spec.json`;
- lanza `scripts/track.py`;
- muestra estado, logs y artefactos del run;
- puede integrarse con el modulo de comentarios.

## Ejecucion

Desde la raiz del repo:

```bash
python interfaz/app.py
```

Los artefactos de cada ejecucion se guardan en `output/interfaz/runs/`.

