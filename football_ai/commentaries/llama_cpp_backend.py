from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib import request

from .generator import (
    CommentaryEvent,
    CommentaryGenerationResult,
    CommentaryLLMRunResult,
    CommentaryPromptBuilder,
    OllamaCommentaryGenerator,
    _clean_text,
)


def _default_llama_cpp_base_url() -> str:
    host = _clean_text(os.environ.get("LLAMA_CPP_BASE_URL"))
    if host is None:
        return "http://127.0.0.1:8001"
    if re.match(r"^https?://", host, flags=re.IGNORECASE):
        return host.rstrip("/")
    return f"http://{host}".rstrip("/")


class LlamaCppCommentaryGenerator(OllamaCommentaryGenerator):
    def __init__(
        self,
        model: str = "gemma4-q4ks-text",
        base_url: str | None = None,
        temperature: float = 0.4,
        top_p: float = 0.95,
        timeout_s: float = 90.0,
        max_tokens: int = 80,
        prompt_builder: CommentaryPromptBuilder | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url or _default_llama_cpp_base_url(),
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            keep_alive=None,
            prompt_builder=prompt_builder,
        )
        self.max_tokens = int(max_tokens)

    def _backend_display_name(self) -> str:
        return "llama.cpp"

    def _backend_start_hint(self) -> str:
        return (
            "Asegurate de que `llama-server` este corriendo "
            f"y escuchando en {self.base_url}."
        )

    def _normalize_raw_response(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(raw_response)
        timings = normalized.get("timings")
        if isinstance(timings, dict):
            prompt_ms = timings.get("prompt_ms")
            predicted_ms = timings.get("predicted_ms")
            if isinstance(prompt_ms, (int, float)):
                normalized["prompt_eval_duration_seconds"] = float(prompt_ms) / 1000.0
            if isinstance(predicted_ms, (int, float)):
                normalized["eval_duration_seconds"] = float(predicted_ms) / 1000.0
            if isinstance(prompt_ms, (int, float)) or isinstance(predicted_ms, (int, float)):
                normalized["total_duration_seconds"] = (
                    float(prompt_ms or 0.0) + float(predicted_ms or 0.0)
                ) / 1000.0
        return normalized

    def check_health(self) -> bool:
        try:
            with request.urlopen(f"{self.base_url}/v1/models", timeout=5.0) as response:
                return response.status == 200
        except Exception:
            return False

    def build_chat_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = self.build_chat_payload(system_prompt, user_prompt)
        raw_response = self._normalize_raw_response(
            self._post_json("/v1/chat/completions", payload)
        )
        return payload, raw_response

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
        choices = raw_response.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = (first_choice.get("message") or {}) if isinstance(first_choice, dict) else {}
        raw_commentary = str(message.get("content") or "")
        cleaned_commentary = self._clean_commentary(raw_commentary)
        used_fallback = False
        final_commentary = cleaned_commentary
        if not final_commentary:
            raise RuntimeError(
                f"llama.cpp devolvio una respuesta vacia: {json.dumps(raw_response, ensure_ascii=False)}"
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
