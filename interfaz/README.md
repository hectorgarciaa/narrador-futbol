# interfaz

Interfaz web ligera para preparar alineaciones y lanzar el tracking del partido.

## Qué hace

- permite introducir dos equipos
- permite definir color de camiseta por equipo
- permite elegir formación (`4-3-3`, `5-3-2`, `4-4-2`)
- renderiza un campo más detallado con marcas reglamentarias y slots clicables
- guarda un `lineup_spec.json` por ejecución
- lanza `scripts/track.py --lineup-spec ...`
- arranca automáticamente el servidor de comentarios al abrir la interfaz
- permite elegir modo de comentarios `live` o `deferred` (por defecto `live`)
- precalienta un comentario de `intro` al arrancar la interfaz para que ya esté listo al guardar la alineación
- si el `intro` sale con una plantilla absurda o el servidor reutilizado es antiguo, la interfaz lo regenera localmente antes de guardarlo
- muestra el estado del proceso y el log en vivo

## Cómo se ejecuta

Desde la raíz del proyecto:

```bash
python interfaz/app.py
```

Opciones:

```bash
python interfaz/app.py --host 127.0.0.1 --port 8767
python interfaz/app.py --commentary-port 8788 --commentary-model gemma4:e2b
```

Después abre `http://127.0.0.1:8767`.

Nota:

- la interfaz expone también respuestas `HEAD` para mejorar compatibilidad con navegadores como Safari;
- si trabajas en una máquina remota, `127.0.0.1` debe estar reenviado a tu equipo local o abrirse con la IP/host remotos.

## Qué genera

Cada ejecución crea:

- `output/interfaz/runs/<run_id>/lineup_spec.json`
- `output/interfaz/runs/<run_id>/status.json`
- `output/interfaz/runs/<run_id>/track.log`
- `output/interfaz/runs/<run_id>/commentaries/events_manifest.jsonl`
- `output/interfaz/runs/<run_id>/commentaries/audio/000_intro.wav`
- `output/interfaz/runs/<run_id>/commentaries/commentary_track.wav` cuando el vídeo termina y se ensambla la pista diferida

El propio `track.py` copia además el spec dentro del directorio de artefactos del vídeo para dejar trazabilidad completa.

## Modo comentarios

- `live`: la interfaz intenta reproducir el `intro` ya precalentado nada más guardar y luego hace polling del manifiesto para sonar nuevos audios a medida que aparezcan.
- `deferred`: se siguen generando y guardando comentarios/audio, pero la interfaz no los reproduce al vuelo. Al terminar el tracking, la interfaz construye una pista continua desde el manifiesto y la incrusta en el MP4 final del tracking.

Para depurar el servidor de comentarios lanzado junto a la interfaz:

```bash
curl http://127.0.0.1:8767/api/commentary-service
```
