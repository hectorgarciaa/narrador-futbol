from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Sequence

from .generator import (
    CommentaryEvent,
    CommentaryGenerationResult,
    OllamaCommentaryGenerator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMENTARIES_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO_OUTPUT_DIR = PROJECT_ROOT / "output" / "commentaries" / "audio"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower())
    return slug.strip("-") or "commentary"


def _detect_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _patch_xtts_audio_loading() -> None:
    try:
        import soundfile as sf
        import torch
        import torchaudio
    except Exception:
        return

    if getattr(torchaudio.load, "__name__", "") == "_soundfile_torchaudio_load":
        return

    def _soundfile_torchaudio_load(
        filepath: str | Path,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        format: str | None = None,
        buffer_size: int = 4096,
        backend: str | None = None,
    ):
        del normalize, format, buffer_size, backend
        wav, sample_rate = sf.read(
            str(filepath),
            dtype="float32",
            always_2d=True,
            start=frame_offset,
            frames=-1 if num_frames == -1 else num_frames,
        )
        audio = torch.from_numpy(wav.T.copy())
        if not channels_first:
            audio = audio.T
        return audio, sample_rate

    torchaudio.load = _soundfile_torchaudio_load


def discover_default_speaker_wavs(
    search_dir: str | Path = COMMENTARIES_ROOT,
) -> list[Path]:
    root = Path(search_dir).expanduser().resolve()
    candidates = sorted(path for path in root.glob("*.wav") if path.is_file())
    preferred = [
        path
        for path in candidates
        if path.name.lower() in {"mi_voz.wav", "mi-voz.wav", "mi voz.wav"}
    ]
    return preferred or candidates


def resolve_speaker_wavs(
    speaker_wavs: Sequence[str | Path] | str | Path | None = None,
) -> tuple[Path, ...]:
    if speaker_wavs is None:
        candidates = discover_default_speaker_wavs()
        if len(candidates) != 1:
            raise ValueError(
                "No se pudo resolver automaticamente `speaker_wav`. "
                "Deja un `mi_Voz.wav` en football_ai/commentaries/ "
                "o pasalo con `--speaker-wav`."
            )
        return (candidates[0],)

    if isinstance(speaker_wavs, (str, Path)):
        items: list[str | Path] = [speaker_wavs]
    else:
        items = list(speaker_wavs)

    resolved = tuple(Path(item).expanduser().resolve() for item in items)
    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "No existen los speaker_wav indicados: " + ", ".join(missing)
        )
    return resolved


@dataclass(slots=True)
class CommentaryAudioResult:
    commentary_result: CommentaryGenerationResult
    audio_path: Path
    speaker_wavs: tuple[Path, ...]
    tts_model: str
    language: str

    @property
    def commentary(self) -> str:
        return self.commentary_result.commentary


class XTTSVoiceSynthesizer:
    _MODEL_CACHE: dict[tuple[str, bool], Any] = {}

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        speaker_wavs: Sequence[str | Path] | str | Path | None = None,
        language: str = "es",
        use_gpu: bool | None = None,
        split_sentences: bool = True,
    ) -> None:
        self.model_name = str(model_name).strip()
        self.speaker_wavs = resolve_speaker_wavs(speaker_wavs)
        self.language = str(language).strip() or "es"
        self.use_gpu = _detect_gpu() if use_gpu is None else bool(use_gpu)
        self.split_sentences = bool(split_sentences)

    def _load_model(self):
        cache_key = (self.model_name, self.use_gpu)
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError(
                "Falta instalar la dependencia de voz. "
                "Instala `coqui-tts` en el entorno del proyecto."
            ) from exc

        _patch_xtts_audio_loading()
        model = TTS(self.model_name)
        if hasattr(model, "to"):
            model = model.to("cuda" if self.use_gpu else "cpu")

        self._MODEL_CACHE[cache_key] = model
        return model

    def synthesize_to_file(
        self,
        text: str,
        file_path: str | Path,
        speaker_wavs: Sequence[str | Path] | str | Path | None = None,
        language: str | None = None,
        split_sentences: bool | None = None,
    ) -> Path:
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("`text` no puede ir vacio para sintetizar audio.")

        output_path = Path(file_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_speaker_wavs = resolve_speaker_wavs(
            speaker_wavs or self.speaker_wavs
        )
        model = self._load_model()
        model.tts_to_file(
            text=clean_text,
            file_path=str(output_path),
            speaker_wav=[str(path) for path in resolved_speaker_wavs],
            language=(language or self.language),
            split_sentences=(
                self.split_sentences
                if split_sentences is None
                else bool(split_sentences)
            ),
        )
        if not output_path.exists():
            raise RuntimeError(
                f"XTTS no genero el fichero de salida esperado: {output_path}"
            )
        return output_path


class CommentaryAudioPipeline:
    def __init__(
        self,
        commentary_generator: OllamaCommentaryGenerator | None = None,
        voice_synthesizer: XTTSVoiceSynthesizer | None = None,
        output_dir: str | Path = DEFAULT_AUDIO_OUTPUT_DIR,
    ) -> None:
        self.commentary_generator = commentary_generator or OllamaCommentaryGenerator()
        self.voice_synthesizer = voice_synthesizer or XTTSVoiceSynthesizer()
        self.output_dir = Path(output_dir).expanduser().resolve()

    def build_output_path(self, event: CommentaryEvent) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = "_".join(
            [
                stamp,
                _slugify(event.action),
                _slugify(event.player_name),
            ]
        )
        return self.output_dir / f"{stem}.wav"

    def generate_to_file(
        self,
        event: CommentaryEvent | dict[str, Any],
        audio_path: str | Path | None = None,
    ) -> CommentaryAudioResult:
        commentary_result = self.commentary_generator.generate(event)
        output_path = (
            Path(audio_path).expanduser().resolve()
            if audio_path is not None
            else self.build_output_path(commentary_result.event)
        )
        audio_path = self.voice_synthesizer.synthesize_to_file(
            commentary_result.commentary,
            output_path,
        )
        return CommentaryAudioResult(
            commentary_result=commentary_result,
            audio_path=audio_path,
            speaker_wavs=self.voice_synthesizer.speaker_wavs,
            tts_model=self.voice_synthesizer.model_name,
            language=self.voice_synthesizer.language,
        )
