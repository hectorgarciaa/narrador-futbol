from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ELEVENLABS_API_KEY_ENV = "ELEVENLABS_API_KEY"
ELEVENLABS_VOICE_ID_ENV = "ELEVENLABS_VOICE_ID"
ELEVENLABS_FEMALE_VOICE_ID_ENV = "ELEVENLABS_FEMALE_VOICE_ID"
ELEVENLABS_MODEL_ID_ENV = "ELEVENLABS_MODEL_ID"
ELEVENLABS_OUTPUT_FORMAT_ENV = "ELEVENLABS_OUTPUT_FORMAT"
ELEVENLABS_LANGUAGE_CODE_ENV = "ELEVENLABS_LANGUAGE_CODE"
ELEVENLABS_STABILITY_ENV = "ELEVENLABS_STABILITY"
ELEVENLABS_SIMILARITY_BOOST_ENV = "ELEVENLABS_SIMILARITY_BOOST"
ELEVENLABS_STYLE_ENV = "ELEVENLABS_STYLE"
ELEVENLABS_SPEED_ENV = "ELEVENLABS_SPEED"
ELEVENLABS_USE_SPEAKER_BOOST_ENV = "ELEVENLABS_USE_SPEAKER_BOOST"
ELEVENLABS_OPTIMIZE_STREAMING_LATENCY_ENV = "ELEVENLABS_OPTIMIZE_STREAMING_LATENCY"

DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_ELEVENLABS_LANGUAGE_CODE = "es"
DEFAULT_ELEVENLABS_VOICE_LABEL = "elevenlabs"


def _load_project_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_placeholder(value: str | None) -> bool:
    text = str(value or "").strip().casefold()
    return text in {
        "",
        "your_api_key_here",
        "your_elevenlabs_api_key_here",
        "your_voice_id_here",
        "<your_api_key_here>",
        "<your_voice_id_here>",
    }


def _optional_float(value: str | float | int | None) -> float | None:
    clean = _clean_text(None if value is None else str(value))
    if clean is None:
        return None
    return float(clean)


def _optional_int(value: str | int | None) -> int | None:
    clean = _clean_text(None if value is None else str(value))
    if clean is None:
        return None
    return int(clean)


def _optional_bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    clean = str(value).strip().casefold()
    if not clean:
        return None
    if clean in {"1", "true", "yes", "on"}:
        return True
    if clean in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Valor booleano no valido para ElevenLabs: {value}")


def _suffix_for_output_format(output_format: str) -> str:
    format_key = str(output_format or DEFAULT_ELEVENLABS_OUTPUT_FORMAT).strip().lower()
    if format_key.startswith("mp3_"):
        return ".mp3"
    if format_key.startswith("opus_"):
        return ".opus"
    if format_key.startswith("pcm_"):
        return ".pcm"
    if format_key.startswith("ulaw_"):
        return ".ulaw"
    if format_key.startswith("alaw_"):
        return ".alaw"
    return ".audio"


def mimetype_for_output_format(output_format: str) -> str:
    format_key = str(output_format or DEFAULT_ELEVENLABS_OUTPUT_FORMAT).strip().lower()
    if format_key.startswith("mp3_"):
        return "audio/mpeg"
    if format_key.startswith("opus_"):
        return "audio/ogg"
    if format_key.startswith("pcm_"):
        return "audio/L16"
    if format_key.startswith("ulaw_"):
        return "audio/basic"
    if format_key.startswith("alaw_"):
        return "audio/basic"
    return "application/octet-stream"


class ElevenLabsVoiceSynthesizer:
    """TTS backend backed by ElevenLabs' chunked streaming endpoint."""

    supports_streaming_audio = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_id: str | None = None,
        model_id: str | None = None,
        output_format: str | None = None,
        language_code: str | None = None,
        stability: float | None = None,
        similarity_boost: float | None = None,
        style: float | None = None,
        speed: float | None = None,
        use_speaker_boost: bool | None = None,
        optimize_streaming_latency: int | None = None,
    ) -> None:
        _load_project_dotenv()
        self.api_key = _clean_text(api_key) or _clean_text(
            os.environ.get(ELEVENLABS_API_KEY_ENV)
        )
        self.voice_id = _clean_text(voice_id) or _clean_text(
            os.environ.get(ELEVENLABS_VOICE_ID_ENV)
        )
        self.model_id = (
            _clean_text(model_id)
            or _clean_text(os.environ.get(ELEVENLABS_MODEL_ID_ENV))
            or DEFAULT_ELEVENLABS_MODEL_ID
        )
        self.output_format = (
            _clean_text(output_format)
            or _clean_text(os.environ.get(ELEVENLABS_OUTPUT_FORMAT_ENV))
            or DEFAULT_ELEVENLABS_OUTPUT_FORMAT
        )
        self.language = (
            _clean_text(language_code)
            or _clean_text(os.environ.get(ELEVENLABS_LANGUAGE_CODE_ENV))
            or DEFAULT_ELEVENLABS_LANGUAGE_CODE
        )
        self.stability = (
            stability
            if stability is not None
            else _optional_float(os.environ.get(ELEVENLABS_STABILITY_ENV))
        )
        self.similarity_boost = (
            similarity_boost
            if similarity_boost is not None
            else _optional_float(os.environ.get(ELEVENLABS_SIMILARITY_BOOST_ENV))
        )
        self.style = (
            style
            if style is not None
            else _optional_float(os.environ.get(ELEVENLABS_STYLE_ENV))
        )
        self.speed = (
            speed
            if speed is not None
            else _optional_float(os.environ.get(ELEVENLABS_SPEED_ENV))
        )
        self.use_speaker_boost = (
            use_speaker_boost
            if use_speaker_boost is not None
            else _optional_bool(os.environ.get(ELEVENLABS_USE_SPEAKER_BOOST_ENV))
        )
        self.optimize_streaming_latency = (
            optimize_streaming_latency
            if optimize_streaming_latency is not None
            else _optional_int(os.environ.get(ELEVENLABS_OPTIMIZE_STREAMING_LATENCY_ENV))
        )
        self.model_name = f"elevenlabs:{self.model_id}"
        self.last_voice_label = DEFAULT_ELEVENLABS_VOICE_LABEL
        self._client: Any | None = None

    @property
    def speaker_wavs(self) -> tuple[Path, ...]:
        return ()

    @property
    def output_suffix(self) -> str:
        return _suffix_for_output_format(self.output_format)

    @property
    def audio_mimetype(self) -> str:
        return mimetype_for_output_format(self.output_format)

    def _validate_config(self) -> None:
        if _is_placeholder(self.api_key):
            raise RuntimeError(
                "Falta configurar ELEVENLABS_API_KEY en `.env` o pasar "
                "`--elevenlabs-api-key`."
            )
        if _is_placeholder(self.voice_id):
            raise RuntimeError(
                "Falta configurar ELEVENLABS_VOICE_ID en `.env` o pasar "
                "`--elevenlabs-voice-id`."
            )

    def _get_client(self):
        self._validate_config()
        if self._client is not None:
            return self._client
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as exc:
            raise RuntimeError(
                "Falta instalar `elevenlabs`. Ejecuta "
                "`.venv/bin/python -m pip install elevenlabs`."
            ) from exc
        self._client = ElevenLabs(api_key=self.api_key)
        return self._client

    def _voice_settings(self):
        if all(
            value is None
            for value in (
                self.stability,
                self.similarity_boost,
                self.style,
                self.speed,
                self.use_speaker_boost,
            )
        ):
            return None
        try:
            from elevenlabs import VoiceSettings
        except ImportError as exc:
            raise RuntimeError(
                "Falta instalar `elevenlabs` para configurar VoiceSettings."
            ) from exc
        return VoiceSettings(
            stability=self.stability,
            similarity_boost=self.similarity_boost,
            style=self.style,
            speed=self.speed,
            use_speaker_boost=self.use_speaker_boost,
        )

    def prepare(self, warmup_text: str | None = None) -> "ElevenLabsVoiceSynthesizer":
        del warmup_text
        self._validate_config()
        return self

    def iter_audio_chunks(self, text: str) -> Iterator[bytes]:
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("`text` no puede ir vacio para sintetizar audio.")

        request_kwargs: dict[str, Any] = {
            "text": clean_text,
            "output_format": self.output_format,
            "model_id": self.model_id,
        }
        if self.language:
            request_kwargs["language_code"] = self.language
        voice_settings = self._voice_settings()
        if voice_settings is not None:
            request_kwargs["voice_settings"] = voice_settings
        if self.optimize_streaming_latency is not None:
            request_kwargs["optimize_streaming_latency"] = self.optimize_streaming_latency

        audio_stream = self._get_client().text_to_speech.stream(
            self.voice_id,
            **request_kwargs,
        )
        for chunk in audio_stream:
            if isinstance(chunk, bytes) and chunk:
                yield chunk

    def synthesize_to_file(
        self,
        text: str,
        file_path: str | Path,
        speaker_wavs: Any | None = None,
        language: str | None = None,
        split_sentences: bool | None = None,
    ) -> Path:
        del speaker_wavs, split_sentences
        original_language = self.language
        if language:
            self.language = str(language).strip() or self.language
        try:
            output_path = Path(file_path).expanduser().resolve()
            suffix = self.output_suffix
            if suffix and output_path.suffix.lower() != suffix:
                output_path = output_path.with_suffix(suffix)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            wrote_bytes = False
            with open(output_path, "wb") as f:
                for chunk in self.iter_audio_chunks(text):
                    f.write(chunk)
                    f.flush()
                    wrote_bytes = True
            if not wrote_bytes:
                raise RuntimeError("ElevenLabs no devolvio audio para la peticion.")
            return output_path
        finally:
            self.language = original_language
