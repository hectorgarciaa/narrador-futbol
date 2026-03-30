# interfaz

Interfaz web ligera para preparar alineaciones y lanzar el tracking del partido.

## Qué hace

- permite introducir dos equipos
- permite definir color de camiseta por equipo
- permite elegir formación (`4-3-3`, `5-3-2`, `4-4-2`)
- renderiza un campo más detallado con marcas reglamentarias y slots clicables
- guarda un `lineup_spec.json` por ejecución
- lanza `scripts/track.py --lineup-spec ...`
- muestra el estado del proceso y el log en vivo

## Cómo se ejecuta

Desde la raíz del proyecto:

```bash
python interfaz/app.py
```

Opciones:

```bash
python interfaz/app.py --host 127.0.0.1 --port 8765
```

Después abre `http://127.0.0.1:8765`.

## Qué genera

Cada ejecución crea:

- `output/interfaz/runs/<run_id>/lineup_spec.json`
- `output/interfaz/runs/<run_id>/status.json`
- `output/interfaz/runs/<run_id>/track.log`

El propio `track.py` copia además el spec dentro del directorio de artefactos del vídeo para dejar trazabilidad completa.
