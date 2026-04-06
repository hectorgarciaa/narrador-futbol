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

INTRO_ACTION = "intro"
PASS_ACTION_PREFIX = "pase"

TEAM_NAME_REQUIRED_ACTIONS = {
    "gol",
}

OPPONENT_TEAM_NAME_REQUIRED_ACTIONS = {
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
    "pase largo": (
        "Debe sonar a envio largo o cambio de juego. El jugador mencionado es quien da el pase. "
        "Usa verbos como lanza, mete, abre o cambia, no despeja."
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
        "Tiene que sonar claramente a gol, con energia, dejando claro que equipo marca y que equipo lo encaja."
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
    event_time_s: float
    player_name: str = ""
    player_position: str = ""
    team_name: str | None = None
    opponent_team_name: str | None = None
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
        self.opponent_team_name = _clean_text(self.opponent_team_name)
        self.team_in_favor = _clean_text(self.team_in_favor)
        self.field_zone = _clean_text(self.field_zone)
        self.action_target = _clean_text(self.action_target)
        self.play_context = _clean_text(self.play_context)
        self.match_score = _clean_text(self.match_score)
        self.intensity = _clean_text(self.intensity)
        self.event_time_s = float(self.event_time_s)

        if self.action_index is not None:
            self.action_index = int(self.action_index)

        if not self.is_intro and not self.player_name:
            raise ValueError("`player_name` no puede ir vacio.")
        if not self.is_intro and not self.player_position:
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
        if (
            self.action in OPPONENT_TEAM_NAME_REQUIRED_ACTIONS
            and not self.opponent_team_name
        ):
            raise ValueError(
                f"`opponent_team_name` es obligatorio para la accion '{self.action}'."
            )
        if (
            self.action == "gol"
            and self.team_name
            and self.opponent_team_name
            and self.team_name.casefold() == self.opponent_team_name.casefold()
        ):
            raise ValueError(
                "`team_name` y `opponent_team_name` no pueden ser el mismo equipo en un gol."
            )

    @property
    def is_intro(self) -> bool:
        return self.action == INTRO_ACTION

    @property
    def is_goal(self) -> bool:
        return self.action == "gol"

    @property
    def should_mention_minute(self) -> bool:
        return self.is_goal

    @property
    def match_minute_text(self) -> str:
        return _format_match_minute(self.event_time_s)

    def to_prompt_payload(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in asdict(self).items()
            if value is not None and value != ""
        }
        if self.action not in {"gol", INTRO_ACTION}:
            payload.pop("opponent_team_name", None)
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


@dataclass(slots=True)
class CommentaryLLMRunResult:
    model: str
    event: CommentaryEvent
    system_prompt: str
    user_prompt: str
    request_payload: dict[str, Any]
    raw_response: dict[str, Any]
    raw_commentary: str
    cleaned_commentary: str
    final_commentary: str
    used_fallback: bool
    total_duration_seconds: float | None = None


class CommentaryPromptBuilder:
    def __init__(self, voice: str = "narrador_tv") -> None:
        self.voice = _clean_text(voice) or "narrador_tv"

    def build_system_prompt(self, event: CommentaryEvent | None = None) -> str:
        if event is not None and event.is_intro:
            return (
                "Eres un narrador de futbol de television en espanol de Espana. "
                "Estas abriendo la retransmision antes de que empiece el partido. "
                "Debe sonar claramente a bienvenida de retransmision. "
                "Puedes sonar cercano, ilusionado y natural. "
                "Hazlo breve y sin inventar sucesos que aun no han pasado. "
                "No uses emojis."
            )
        if event is not None and event.is_goal:
            return (
                "Eres un narrador de futbol de television en espanol de Espana. "
                "Si hay un gol, puedes sonar mas emocional, mas expresivo y algo mas largo que en el resto de acciones. "
                "Debe quedar clarisimo quien marca y a quien se lo marca, sin inventar jugadas que no esten en el evento ni consecuencias posteriores. "
                "No uses emojis."
            )
        return (
            "Eres un narrador de futbol de television en espanol de Espana. "
            "Devuelve una sola frase corta, natural y directa. "
            "La frase debe incluir literalmente la accion indicada. "
            "No expliques nada ni inventes contexto."
        )

    def build_user_prompt(self, event: CommentaryEvent) -> str:
        if event.is_intro:
            rules = [
                "Es el comentario de apertura del partido, antes de que ruede el balon.",
                "Escribe una apertura breve de retransmision, con una o dos frases como maximo y entre 16 y 30 palabras.",
                "Lo natural es una bienvenida tipo bienvenidos, buenas tardes o ya esta todo listo.",
                "Puedes sonar ilusionado, cercano y natural.",
                "No menciones minuto ni una accion de juego que ya haya ocurrido.",
                "No menciones jugador, posicion, dorsal, rol tactico ni placeholders.",
                "No predigas el resultado ni digas que el gol va a caer seguro.",
                "No exageres con frases como historia, legendario, los mejores del mundo o similares.",
                "No uses emojis.",
            ]
            if event.team_name and event.opponent_team_name:
                rules.append(
                    f"Si encaja, menciona que hoy juegan {event.team_name} contra {event.opponent_team_name}."
                )
            elif event.team_name:
                rules.append(f"Si encaja, menciona a {event.team_name}.")
            if event.play_context:
                rules.append(f"Si ayuda, menciona {event.play_context}.")
            if event.match_score:
                rules.append("No inventes marcador si el partido todavia no ha empezado.")
        elif event.is_goal:
            rules = [
                f"El autor del gol es {event.player_name}.",
                "Tiene que sonar claramente a gol.",
                "Puedes escribir una narracion mas larga y emocionante que en el resto de acciones.",
                "Se permiten una o dos frases, con tono de retransmision, y entre 20 y 45 palabras.",
                f"Menciona exactamente una vez el {event.match_minute_text}.",
                "No inventes que decide el partido, que es el mejor gol del ano, ni la respuesta del rival.",
                "No inventes asistencia, remate concreto, jugada previa ni consecuencias si no vienen en el evento.",
                "Usa solo los datos del evento: autor, equipo que marca, equipo que encaja, minuto y zona si existe.",
                "No uses emojis.",
            ]
            if event.team_name and event.opponent_team_name:
                rules.append(f"{event.team_name} marca y {event.opponent_team_name} encaja.")
            if event.field_zone:
                rules.append(f"Si ayuda, menciona {event.field_zone}.")
            if event.play_context:
                rules.append(f"Si ayuda, menciona {event.play_context}.")
            if event.action_target:
                rules.append(f"Si ayuda, menciona {event.action_target}.")
        else:
            rules = [
                f"Usa literalmente esta accion: {event.action}.",
                f"El jugador es {event.player_name}.",
                "Escribe una sola frase corta.",
            ]

            if event.should_mention_minute:
                rules.append(f"Menciona exactamente una vez el {event.match_minute_text}.")
            else:
                rules.append("No menciones el minuto.")
                rules.append("No menciones al equipo contrario.")

            if event.action.startswith(PASS_ACTION_PREFIX):
                rules.append("En un pase, el jugador lo da, no lo recibe.")
                rules.append("No inventes receptor ni jugada posterior.")

            if event.action in SPECIAL_TEAM_FAVOR_ACTIONS and event.team_in_favor:
                rules.append(f"La accion es a favor de {event.team_in_favor}.")
            if event.field_zone:
                rules.append(f"Si encaja, menciona {event.field_zone}.")
            if event.action_target:
                rules.append(f"Si ayuda, menciona {event.action_target}.")
            if event.play_context:
                rules.append(f"Si ayuda, menciona {event.play_context}.")

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
        model: str = "gemma4:e2b",
        base_url: str | None = None,
        temperature: float = 0.4,
        top_p: float = 0.95,
        timeout_s: float = 90.0,
        prompt_builder: CommentaryPromptBuilder | None = None,
    ) -> None:
        self.model = _clean_text(model) or "gemma4:e2b"
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

    def _normalize_raw_response(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(raw_response)
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        ):
            value = normalized.get(key)
            if isinstance(value, (int, float)):
                normalized[f"{key}_seconds"] = float(value) / 1_000_000_000.0
        return normalized

    def check_health(self) -> bool:
        try:
            with request.urlopen(f"{self.base_url}/api/tags", timeout=5.0) as response:
                return response.status == 200
        except Exception:
            return False

    def normalize_event(
        self,
        event: CommentaryEvent | dict[str, Any],
    ) -> CommentaryEvent:
        return (
            event if isinstance(event, CommentaryEvent) else CommentaryEvent.from_dict(event)
        )

    def build_prompts(
        self,
        event: CommentaryEvent | dict[str, Any],
    ) -> tuple[CommentaryEvent, str, str]:
        commentary_event = self.normalize_event(event)
        system_prompt = self.prompt_builder.build_system_prompt(commentary_event)
        user_prompt = self.prompt_builder.build_user_prompt(commentary_event)
        return commentary_event, system_prompt, user_prompt

    def build_chat_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
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

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = self.build_chat_payload(system_prompt, user_prompt)
        raw_response = self._normalize_raw_response(
            self._post_json("/api/chat", payload)
        )
        return payload, raw_response

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
        words = re.findall(r"\b\w+\b", commentary, flags=re.UNICODE)
        if len(words) < 4 or len(words) > 30:
            return True
        if len(commentary) > 220:
            return True
        if any(token in commentary for token in ("{", "}", "[", "]", "\n", "\r")):
            return True
        if commentary.count(":") > 1:
            return True
        if commentary.count('"') > 0 or commentary.count("'") > 2:
            return True
        if commentary.count(" - ") > 0:
            return True
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

    def run_llm(
        self,
        event: CommentaryEvent | dict[str, Any],
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> CommentaryLLMRunResult:
        commentary_event, default_system_prompt, default_user_prompt = self.build_prompts(event)
        system_prompt = str(system_prompt or default_system_prompt)
        user_prompt = str(user_prompt or default_user_prompt)
        request_payload, raw_response = self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        raw_commentary = ((raw_response.get("message", {}) or {}).get("content", ""))
        cleaned_commentary = self._clean_commentary(raw_commentary)
        used_fallback = False
        final_commentary = cleaned_commentary
        if not final_commentary:
            raise RuntimeError(
                f"Ollama devolvio una respuesta vacia: {json.dumps(raw_response, ensure_ascii=False)}"
            )
        return CommentaryLLMRunResult(
            model=self.model,
            event=commentary_event,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request_payload=request_payload,
            raw_response=raw_response,
            raw_commentary=raw_commentary,
            cleaned_commentary=cleaned_commentary,
            final_commentary=final_commentary,
            used_fallback=used_fallback,
            total_duration_seconds=raw_response.get("total_duration_seconds"),
        )

    def generate(self, event: CommentaryEvent | dict[str, Any]) -> CommentaryGenerationResult:
        llm_result = self.run_llm(event)
        return CommentaryGenerationResult(
            commentary=llm_result.final_commentary,
            model=llm_result.model,
            event=llm_result.event,
            raw_response=llm_result.raw_response,
            system_prompt=llm_result.system_prompt,
            user_prompt=llm_result.user_prompt,
        )
