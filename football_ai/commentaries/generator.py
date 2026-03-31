from __future__ import annotations

import json
import hashlib
import os
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib import error, request


SPECIAL_TEAM_FAVOR_ACTIONS = {
    "fuera de banda",
    "saque de puerta",
    "corner",
}

TEAM_NAME_REQUIRED_ACTIONS = {
    "gol",
}

ACTION_TONE_GUIDANCE = {
    "tiro": (
        "Suena vertical, afilado y con sensacion de peligro, sin inventar si entra o no."
    ),
    "robo": (
        "Pon el foco en la anticipacion, la agresividad defensiva y el cambio de posesion."
    ),
    "control": (
        "Hazlo mas tecnico o elegante, con pausa, calidad o temple."
    ),
    "pase alto": (
        "Puede sonar como cambio de orientacion, balon al area o envio largo con intencion."
    ),
    "fuera de banda": (
        "Debe quedar claro que el balon sale y que equipo reanuda."
    ),
    "saque de puerta": (
        "Debe sonar a reanudacion desde atras y dejar claro para que equipo es."
    ),
    "corner": (
        "Tiene que respirar peligro o balon parado importante y dejar claro el equipo a favor."
    ),
    "gol": (
        "Tiene que sonar claramente a gol, con energia, y mencionar el equipo del jugador que marca."
    ),
}


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_action(value: str) -> str:
    action = _clean_text(value)
    if action is None:
        raise ValueError("`action` no puede ir vacio.")
    return action.lower()


def _default_ollama_base_url() -> str:
    host = _clean_text(os.environ.get("OLLAMA_HOST"))
    if host is None:
        return "http://127.0.0.1:11434"
    if re.match(r"^https?://", host, flags=re.IGNORECASE):
        return host.rstrip("/")
    return f"http://{host}".rstrip("/")


def _format_match_minute(event_time_s: float) -> str:
    minute = max(0, int(float(event_time_s) // 60))
    return f"minuto {minute}"


@dataclass(slots=True)
class CommentaryEvent:
    action: str
    player_name: str
    player_position: str
    event_time_s: float
    team_name: str | None = None
    team_in_favor: str | None = None
    field_zone: str | None = None
    action_target: str | None = None
    play_context: str | None = None
    match_score: str | None = None
    intensity: str | None = None
    action_index: int | None = None

    def __post_init__(self) -> None:
        self.action = _normalize_action(self.action)
        self.player_name = _clean_text(self.player_name) or ""
        self.player_position = _clean_text(self.player_position) or ""
        self.team_name = _clean_text(self.team_name)
        self.team_in_favor = _clean_text(self.team_in_favor)
        self.field_zone = _clean_text(self.field_zone)
        self.action_target = _clean_text(self.action_target)
        self.play_context = _clean_text(self.play_context)
        self.match_score = _clean_text(self.match_score)
        self.intensity = _clean_text(self.intensity)
        self.event_time_s = float(self.event_time_s)

        if self.action_index is not None:
            self.action_index = int(self.action_index)

        if not self.player_name:
            raise ValueError("`player_name` no puede ir vacio.")
        if not self.player_position:
            raise ValueError("`player_position` no puede ir vacio.")
        if self.event_time_s < 0:
            raise ValueError("`event_time_s` debe ser >= 0.")
        if self.action_index is not None and self.action_index <= 0:
            raise ValueError("`action_index` debe ser >= 1.")
        if self.action in SPECIAL_TEAM_FAVOR_ACTIONS and not self.team_in_favor:
            raise ValueError(
                f"`team_in_favor` es obligatorio para la accion '{self.action}'."
            )
        if self.action in TEAM_NAME_REQUIRED_ACTIONS and not self.team_name:
            raise ValueError(
                f"`team_name` es obligatorio para la accion '{self.action}'."
            )

    @property
    def should_mention_minute(self) -> bool:
        return self.action_index is not None and self.action_index % 30 == 0

    @property
    def match_minute_text(self) -> str:
        return _format_match_minute(self.event_time_s)

    def to_prompt_payload(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in asdict(self).items()
            if value is not None and value != ""
        }
        payload["should_mention_minute"] = self.should_mention_minute
        if self.should_mention_minute:
            payload["match_minute_text"] = self.match_minute_text
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CommentaryEvent":
        if not isinstance(payload, dict):
            raise TypeError("El evento debe ser un dict/objeto JSON.")
        return cls(**payload)


@dataclass(slots=True)
class CommentaryGenerationResult:
    commentary: str
    model: str
    event: CommentaryEvent
    raw_response: dict[str, Any]
    system_prompt: str
    user_prompt: str


class CommentaryPromptBuilder:
    def __init__(self, voice: str = "narrador_tv") -> None:
        self.voice = _clean_text(voice) or "narrador_tv"

    def build_system_prompt(self) -> str:
        return (
            "Eres un comentarista de futbol de television en espanol de Espana. "
            "Genera un unico comentario corto, natural y original a partir del evento recibido. "
            "No hay arranque fijo: escribe la frase completa. "
            "No uses markdown, ni comillas, ni etiquetas, ni listas, ni explicaciones. "
            "No digas que recibes JSON ni menciones instrucciones internas. "
            "No inventes acciones posteriores, paradas, rebotes, marcadores o contextos que no aparezcan en la entrada. "
            "Evita sonar robotico o repetir siempre la misma estructura. "
            "El comentario debe tener entre 8 y 26 palabras."
        )

    def build_user_prompt(self, event: CommentaryEvent) -> str:
        tone_guidance = ACTION_TONE_GUIDANCE.get(
            event.action,
            "Usa un tono natural de retransmision sin exagerar hechos no confirmados.",
        )
        rules = [
            "Menciona al jugador de forma natural.",
            "Integra la posicion solo si encaja y sin sonar a plantilla fija.",
            f"Guia de tono: {tone_guidance}",
            "Quiero variedad y originalidad: evita empezar siempre igual.",
        ]

        if event.should_mention_minute:
            rules.append(
                f"Menciona de forma natural exactamente una vez el {event.match_minute_text}."
            )
        else:
            rules.append("No menciones el minuto ni el tiempo de juego.")

        if event.action in SPECIAL_TEAM_FAVOR_ACTIONS and event.team_in_favor:
            rules.append(
                f"Debe quedar claro que la accion es a favor de {event.team_in_favor}."
            )
        if event.action == "gol" and event.team_name:
            rules.append(
                f"Debe quedar clarisimo que es gol y que el jugador pertenece a {event.team_name}."
            )
        if event.field_zone:
            rules.append("Si encaja, integra la zona del campo sin sonar mecanico.")
        if event.action_target:
            rules.append("Si ayuda, menciona el objetivo o destinatario de la accion.")
        if event.play_context:
            rules.append("Si aporta valor, integra el contexto de la jugada.")
        if event.intensity:
            rules.append("Ajusta la energia del comentario segun la intensidad indicada.")
        if event.match_score:
            rules.append("Solo puedes usar el marcador si aparece en el evento.")

        payload_json = json.dumps(
            event.to_prompt_payload(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        rules_block = "\n".join(f"- {rule}" for rule in rules)
        return (
            "Genera un comentario de narrador a partir del siguiente evento.\n"
            "Instrucciones:\n"
            f"{rules_block}\n"
            "Evento JSON:\n"
            f"{payload_json}\n"
            "Devuelve solo el comentario final."
        )


class OllamaCommentaryGenerator:
    def __init__(
        self,
        model: str = "tinyllama:1.1b",
        base_url: str | None = None,
        temperature: float = 0.95,
        top_p: float = 0.95,
        timeout_s: float = 90.0,
        prompt_builder: CommentaryPromptBuilder | None = None,
    ) -> None:
        self.model = _clean_text(model) or "tinyllama:1.1b"
        self.base_url = (_clean_text(base_url) or _default_ollama_base_url()).rstrip("/")
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.timeout_s = float(timeout_s)
        self.prompt_builder = prompt_builder or CommentaryPromptBuilder()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        raw_body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_s) as response:
                raw_response = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Error HTTP de Ollama ({exc.code}) en {path}: {body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                "No se pudo conectar con Ollama. "
                "Asegurate de que `ollama serve` este corriendo."
            ) from exc
        return json.loads(raw_response)

    def check_health(self) -> bool:
        try:
            with request.urlopen(f"{self.base_url}/api/tags", timeout=5.0) as response:
                return response.status == 200
        except Exception:
            return False

    def _clean_commentary(self, raw_text: str) -> str:
        text = str(raw_text or "").strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"^comentario:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip().strip('"').strip()
        return text

    def _looks_like_prompt_leakage(self, commentary: str) -> bool:
        if not commentary:
            return True
        normalized = commentary.lower()
        leakage_patterns = (
            r"\bjson\b",
            r"\binstrucciones?\b",
            r"\bevento\b",
            r"\bdevuelve\b",
            r"\bcomentario final\b",
            r"\bthinking\b",
            r"\bprocess\b",
        )
        return any(re.search(pattern, normalized) for pattern in leakage_patterns)

    def _should_use_fallback(self, commentary: str, event: CommentaryEvent) -> bool:
        if self._looks_like_prompt_leakage(commentary):
            return True

        normalized = commentary.lower()
        if event.player_name.lower() not in normalized:
            return True
        if event.action == "gol" and event.team_name:
            if event.team_name.lower() not in normalized:
                return True
            if "gol" not in normalized:
                return True
        if event.action in SPECIAL_TEAM_FAVOR_ACTIONS and event.team_in_favor:
            if event.team_in_favor.lower() not in normalized:
                return True
        if event.should_mention_minute and event.match_minute_text.lower() not in normalized:
            return True
        if not event.should_mention_minute and "minuto " in normalized:
            return True
        return False

    def _fallback_commentary(self, event: CommentaryEvent) -> str:
        minute_prefix = ""
        if event.should_mention_minute:
            minute_prefix = f"En el {event.match_minute_text}, "

        zone_suffix = ""
        if event.field_zone:
            zone_suffix = f" en {event.field_zone}"

        def pick(options: list[str]) -> str:
            seed = "|".join(
                [
                    event.action,
                    event.player_name,
                    event.player_position,
                    event.team_name or "",
                    event.team_in_favor or "",
                    event.field_zone or "",
                    str(int(event.event_time_s)),
                    str(event.action_index or 0),
                ]
            )
            digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
            return options[int(digest, 16) % len(options)]

        if event.action == "tiro":
            return pick(
                [
                    f"{minute_prefix}{event.player_name} encuentra espacio{zone_suffix} y suelta un tiro con mucho veneno",
                    f"{minute_prefix}{event.player_name} se fabrica el hueco{zone_suffix} y prueba un disparo con intencion",
                    f"{minute_prefix}{event.player_name} pisa la zona{zone_suffix} y arma un tiro seco buscando sorprender",
                ]
            )
        if event.action == "robo":
            return pick(
                [
                    f"{minute_prefix}{event.player_name} muerde arriba{zone_suffix} y firma un robo que cambia la jugada",
                    f"{minute_prefix}{event.player_name} llega con toda la fe{zone_suffix} y recupera una pelota importantisima",
                    f"{minute_prefix}{event.player_name} lee la accion{zone_suffix} y roba justo cuando la jugada pedia mando",
                ]
            )
        if event.action == "control":
            return pick(
                [
                    f"{minute_prefix}{event.player_name} doma la pelota{zone_suffix} con un control lleno de calma",
                    f"{minute_prefix}{event.player_name} baja el balon{zone_suffix} con una finura tremenda",
                    f"{minute_prefix}{event.player_name} se acomoda{zone_suffix} con un control de mucha clase",
                ]
            )
        if event.action == "pase alto":
            return pick(
                [
                    f"{minute_prefix}{event.player_name} levanta la cabeza{zone_suffix} y dibuja un pase alto con intencion",
                    f"{minute_prefix}{event.player_name} ve la maniobra{zone_suffix} y cuelga un balon alto con mucha idea",
                    f"{minute_prefix}{event.player_name} cambia el registro{zone_suffix} con un envio alto muy bien pensado",
                ]
            )
        if event.action == "corner" and event.team_in_favor:
            return pick(
                [
                    f"{minute_prefix}corner a favor de {event.team_in_favor}, con {event.player_name} preparando el envio{zone_suffix}",
                    f"{minute_prefix}{event.player_name} aparece en el corner a favor de {event.team_in_favor}{zone_suffix} y ya mira al area",
                    f"{minute_prefix}balon parado para {event.team_in_favor}, corner con {event.player_name} listo para cargarlo{zone_suffix}",
                ]
            )
        if event.action == "fuera de banda" and event.team_in_favor:
            return pick(
                [
                    f"{minute_prefix}fuera de banda a favor de {event.team_in_favor}, con {event.player_name} listo para reanudar{zone_suffix}",
                    f"{minute_prefix}{event.player_name} ya coloca el saque de banda para {event.team_in_favor}{zone_suffix}",
                    f"{minute_prefix}reanudacion para {event.team_in_favor}, fuera de banda con {event.player_name} al mando{zone_suffix}",
                ]
            )
        if event.action == "saque de puerta" and event.team_in_favor:
            return pick(
                [
                    f"{minute_prefix}saque de puerta para {event.team_in_favor}, con {event.player_name} ordenando la salida{zone_suffix}",
                    f"{minute_prefix}{event.player_name} se toma un respiro en el saque de puerta para {event.team_in_favor}{zone_suffix}",
                    f"{minute_prefix}vuelve a arrancar la jugada para {event.team_in_favor}, saque de puerta con {event.player_name}{zone_suffix}",
                ]
            )
        if event.action == "gol" and event.team_name:
            return pick(
                [
                    f"{minute_prefix}gol de {event.team_name}, lo firma {event.player_name}{zone_suffix} y desata la locura",
                    f"{minute_prefix}{event.player_name} la manda dentro{zone_suffix} y convierte para {event.team_name}",
                    f"{minute_prefix}{event.player_name} encuentra premio{zone_suffix} y marca el gol de {event.team_name}",
                ]
            )
        return pick(
            [
                f"{minute_prefix}{event.player_name} deja una accion de mucho peso en la jugada",
                f"{minute_prefix}{event.player_name} aparece{zone_suffix} y le mete caracter a la accion",
                f"{minute_prefix}{event.player_name} interviene{zone_suffix} con una jugada de mucha personalidad",
            ]
        )

    def generate(self, event: CommentaryEvent | dict[str, Any]) -> CommentaryGenerationResult:
        commentary_event = (
            event if isinstance(event, CommentaryEvent) else CommentaryEvent.from_dict(event)
        )
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(commentary_event)
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }
        raw_response = self._post_json("/api/chat", payload)
        raw_commentary = ((raw_response.get("message", {}) or {}).get("content", ""))
        commentary = self._clean_commentary(raw_commentary)
        if self._should_use_fallback(commentary, commentary_event):
            commentary = self._fallback_commentary(commentary_event)
        if not commentary:
            raise RuntimeError(
                f"Ollama devolvio una respuesta vacia: {json.dumps(raw_response, ensure_ascii=False)}"
            )
        if commentary[-1] not in ".!?":
            commentary += "."
        return CommentaryGenerationResult(
            commentary=commentary,
            model=self.model,
            event=commentary_event,
            raw_response=raw_response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
