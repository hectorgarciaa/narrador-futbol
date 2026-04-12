from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .generator import (
    CommentaryEvent,
    CommentaryGenerationResult,
    CommentaryLLMRunResult,
    CommentaryPromptBuilder,
)


DEFAULT_HYMBA_MODEL = "nvidia/Hymba-1.5B-Instruct"
DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface-no-xet"


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(slots=True)
class _LoadedTransformersArtifacts:
    torch: Any
    tokenizer: Any
    model: Any
    stop_string_criteria: Any
    stopping_criteria_list: Any


class TransformersCommentaryGenerator:
    def __init__(
        self,
        model: str = DEFAULT_HYMBA_MODEL,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_new_tokens: int = 80,
        do_sample: bool = True,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        prompt_builder: CommentaryPromptBuilder | None = None,
        trust_remote_code: bool = True,
        disable_xet: bool = True,
        hf_home: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model = _clean_optional_text(model) or DEFAULT_HYMBA_MODEL
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.device = _clean_optional_text(device) or "cuda"
        self.torch_dtype = (_clean_optional_text(torch_dtype) or "bfloat16").lower()
        self.prompt_builder = prompt_builder or CommentaryPromptBuilder()
        self.trust_remote_code = bool(trust_remote_code)
        self.disable_xet = bool(disable_xet)
        self.hf_home = _clean_optional_text(hf_home)
        self.cache_dir = _clean_optional_text(cache_dir)
        self._loaded: _LoadedTransformersArtifacts | None = None

        if self.max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` debe ser >= 1.")

    def _configure_hf_environment(self) -> str | None:
        if self.disable_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if self.hf_home:
            resolved_hf_home = str(Path(self.hf_home).expanduser())
        else:
            resolved_hf_home = str(DEFAULT_HF_HOME)
        os.environ.setdefault("HF_HOME", resolved_hf_home)
        Path(resolved_hf_home).mkdir(parents=True, exist_ok=True)
        return resolved_hf_home

    def _resolve_torch_dtype(self, torch_module: Any) -> Any:
        dtype_name = self.torch_dtype
        supported = {
            "bfloat16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "float32": torch_module.float32,
        }
        if dtype_name not in supported:
            raise ValueError(
                "`torch_dtype` debe ser uno de: bfloat16, float16, float32."
            )
        return supported[dtype_name]

    def _ensure_loaded(self) -> tuple[_LoadedTransformersArtifacts, float]:
        if self._loaded is not None:
            return self._loaded, 0.0

        self._configure_hf_environment()

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            StopStringCriteria,
            StoppingCriteriaList,
        )

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Se pidio backend transformers en CUDA, pero `torch.cuda.is_available()` es false."
            )

        load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=self._resolve_torch_dtype(torch),
            cache_dir=self.cache_dir,
        )
        model = model.to(self.device)
        load_duration_seconds = time.perf_counter() - load_start

        self._loaded = _LoadedTransformersArtifacts(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            stop_string_criteria=StopStringCriteria,
            stopping_criteria_list=StoppingCriteriaList,
        )
        return self._loaded, load_duration_seconds

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
        *,
        avoid_commentary: str | None = None,
    ) -> tuple[CommentaryEvent, str, str]:
        commentary_event = self.normalize_event(event)
        system_prompt = self.prompt_builder.build_system_prompt(commentary_event)
        user_prompt = self.prompt_builder.build_user_prompt(
            commentary_event,
            avoid_commentary=avoid_commentary,
        )
        return commentary_event, system_prompt, user_prompt

    def _clean_commentary(self, raw_text: str) -> str:
        text = str(raw_text or "").strip()
        text = text.replace("</s>", " ")
        text = text.replace("<s>", " ")
        text = text.replace("<|end_of_text|>", " ")
        text = text.replace("<|eot_id|>", " ")
        text = " ".join(text.split()).strip().strip('"').strip()
        return text

    def run_llm(
        self,
        event: CommentaryEvent | dict[str, Any],
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        avoid_commentary: str | None = None,
    ) -> CommentaryLLMRunResult:
        commentary_event, default_system_prompt, default_user_prompt = self.build_prompts(
            event,
            avoid_commentary=avoid_commentary,
        )
        system_prompt = str(system_prompt or default_system_prompt)
        user_prompt = str(user_prompt or default_user_prompt)

        loaded, load_duration_seconds = self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request_payload = {
            "backend": "transformers",
            "model": self.model,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "messages": messages,
            "generation": {
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "trust_remote_code": self.trust_remote_code,
                "disable_xet": self.disable_xet,
                "hf_home": os.environ.get("HF_HOME"),
                "cache_dir": self.cache_dir,
            },
        }

        tokenized_chat = loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        stopping_criteria = loaded.stopping_criteria_list(
            [
                loaded.stop_string_criteria(
                    tokenizer=loaded.tokenizer,
                    stop_strings="</s>",
                )
            ]
        )
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
            "stopping_criteria": stopping_criteria,
        }
        if self.do_sample:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
        else:
            generate_kwargs["do_sample"] = False

        generation_start = time.perf_counter()
        outputs = loaded.model.generate(tokenized_chat, **generate_kwargs)
        generation_duration_seconds = time.perf_counter() - generation_start
        total_duration_seconds = load_duration_seconds + generation_duration_seconds

        input_length = int(tokenized_chat.shape[1])
        raw_commentary = loaded.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        ).strip()
        cleaned_commentary = self._clean_commentary(raw_commentary)
        if not cleaned_commentary:
            raise RuntimeError(
                "Transformers/Hymba devolvio una respuesta vacia tras la limpieza."
            )

        raw_response = {
            "backend": "transformers",
            "model_id": self.model,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "load_duration_seconds": load_duration_seconds,
            "generation_duration_seconds": generation_duration_seconds,
            "total_duration_seconds": total_duration_seconds,
            "input_token_count": input_length,
            "output_token_count": int(outputs.shape[1]) - input_length,
            "generated_text": raw_commentary,
        }

        return CommentaryLLMRunResult(
            model=self.model,
            event=commentary_event,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request_payload=request_payload,
            raw_response=raw_response,
            raw_commentary=raw_commentary,
            cleaned_commentary=cleaned_commentary,
            final_commentary=cleaned_commentary,
            used_fallback=False,
            total_duration_seconds=total_duration_seconds,
        )

    def generate(
        self,
        event: CommentaryEvent | dict[str, Any],
        *,
        avoid_commentary: str | None = None,
    ) -> CommentaryGenerationResult:
        llm_result = self.run_llm(event, avoid_commentary=avoid_commentary)
        return CommentaryGenerationResult(
            commentary=llm_result.final_commentary,
            model=llm_result.model,
            event=llm_result.event,
            raw_response=llm_result.raw_response,
            system_prompt=llm_result.system_prompt,
            user_prompt=llm_result.user_prompt,
        )
