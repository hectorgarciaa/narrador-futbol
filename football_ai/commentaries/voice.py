from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Sequence

from .generator import (
    CommentaryEvent,
    CommentaryGenerationResult,
    OllamaCommentaryGenerator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMENTARIES_ROOT = Path(__file__).resolve().parent
DEFAULT_AUDIO_OUTPUT_DIR = PROJECT_ROOT / "output" / "commentaries" / "audio"
DEFAULT_VOICE_CACHE_DIR = PROJECT_ROOT / "output" / "commentaries" / "voices"
DEFAULT_QWEN_VOICE_CACHE_DIR = PROJECT_ROOT / "output" / "commentaries" / "qwen_voices"
DEFAULT_QWEN_CPP_RUNTIME_DIR = PROJECT_ROOT / "output" / "commentaries" / "qwen_cpp_runtime"
DEFAULT_QWEN_CPP_MODEL_DIR = DEFAULT_QWEN_CPP_RUNTIME_DIR / "models"
DEFAULT_QWEN_CPP_THREADS = 6
DEFAULT_TTS_BACKEND = "xtts"
DEFAULT_QWEN_VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_QWEN_VOICE_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_QWEN_USE_FLASH_ATTENTION = False
DEFAULT_QWEN_VOICE_DESIGN_PROMPT = (
    "Design a voice for a professional football commentator speaking in Spanish from Spain. "
    "Male adult voice, warm, clear, charismatic, and emotionally expressive. "
    "The voice should sound like a live match narrator: energetic, passionate, and intense, "
    "with natural excitement, quick rhythm, short pauses, and rising emotion during dangerous plays. "
    "He should sound confident, vivid, and engaging, like a real TV sports broadcaster, "
    "but never cartoonish or overacted. Prioritize emotional delivery, expressive prosody, "
    "dynamic pacing, and natural realism."
)
DEFAULT_QWEN_REFERENCE_TEXT = (
    "Buenas tardes, bienvenidos a una gran noche de futbol. "
    "Ya rueda la emocion y tenemos un partidazo por delante."
)
QWEN_LANGUAGE_ALIASES = {
    "es": "Spanish",
    "es-es": "Spanish",
    "español": "Spanish",
    "espanol": "Spanish",
    "spanish": "Spanish",
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "ru": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "it": "Italian",
    "italian": "Italian",
}
QWEN_CPP_LANGUAGE_IDS = {
    "english": 2050,
    "german": 2053,
    "spanish": 2054,
    "chinese": 2055,
    "japanese": 2058,
    "french": 2061,
    "korean": 2064,
    "russian": 2069,
    "italian": 2070,
    "portuguese": 2071,
}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower())
    return slug.strip("-") or "commentary"


def _detect_gpu() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _flash_attention_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _system_sox_available() -> bool:
    return shutil.which("sox") is not None


def normalize_qwen_language(language: str | None) -> str:
    clean = str(language or "").strip()
    if not clean:
        return "Spanish"
    return QWEN_LANGUAGE_ALIASES.get(clean.casefold(), clean)


def qwen_cpp_language_id(language: str | None) -> int:
    normalized = normalize_qwen_language(language)
    return QWEN_CPP_LANGUAGE_IDS.get(normalized.casefold(), 2054)


def _find_first_existing_path(candidates: Sequence[str | Path]) -> Path | None:
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def _preferred_cmake_command() -> list[str]:
    venv_cmake = Path(sys.executable).resolve().parent / "cmake"
    if venv_cmake.exists():
        return [str(venv_cmake)]
    system_cmake = shutil.which("cmake")
    if system_cmake:
        return [system_cmake]
    if importlib.util.find_spec("cmake") is not None:
        return [sys.executable, "-m", "cmake"]
    raise RuntimeError(
        "No se encontro `cmake`. Instala `cmake` en la `.venv` o en el sistema."
    )


def _nvcc_available() -> bool:
    return shutil.which("nvcc") is not None


def _preferred_cuda_architecture() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:
        pass
    return "86"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)


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


def _patch_qwen_tts_sox() -> None:
    try:
        import numpy as np
        import onnxruntime
        from qwen_tts.core.tokenizer_25hz.vq import speech_vq
    except Exception:
        return

    extractor_cls = speech_vq.XVectorExtractor
    if getattr(extractor_cls, "_narrador_no_sox_patch", False):
        return

    def _patched_init(self, audio_codec_with_xvector):
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        self.ort_session = onnxruntime.InferenceSession(
            audio_codec_with_xvector,
            sess_options=option,
            providers=providers,
        )
        self.tfm = None
        self.mel_ext = speech_vq.MelSpectrogramFeatures(
            filter_length=1024,
            hop_length=160,
            win_length=640,
            n_mel_channels=80,
            mel_fmin=0,
            mel_fmax=8000,
            sampling_rate=16000,
        )

    def _patched_sox_norm(self, audio):
        wav = np.asarray(audio, dtype=np.float32)
        if wav.size == 0:
            return wav
        peak = float(np.max(np.abs(wav)))
        if peak <= 1e-6:
            return wav
        target_peak = 10.0 ** (-6.0 / 20.0)
        scale = target_peak / peak
        return wav * scale

    extractor_cls.__init__ = _patched_init
    extractor_cls.sox_norm = _patched_sox_norm
    extractor_cls._narrador_no_sox_patch = True


def discover_default_speaker_wavs(
    search_dir: str | Path = COMMENTARIES_ROOT,
) -> list[Path]:
    root = Path(search_dir).expanduser().resolve()
    candidates = sorted(path for path in root.glob("*.wav") if path.is_file())
    if not candidates:
        return []

    preferred = []
    remaining = []
    for path in candidates:
        if path.name.lower() in {"mi_voz.wav", "mi-voz.wav", "mi voz.wav"}:
            preferred.append(path)
        else:
            remaining.append(path)
    return preferred + remaining


def resolve_speaker_wavs(
    speaker_wavs: Sequence[str | Path] | str | Path | None = None,
) -> tuple[Path, ...]:
    if speaker_wavs is None:
        candidates = discover_default_speaker_wavs()
        if not candidates:
            raise ValueError(
                "No se pudo resolver automaticamente `speaker_wav`. "
                "Deja uno o varios `.wav` en football_ai/commentaries/ "
                "o pasalo con `--speaker-wav`."
            )
        return tuple(candidates)

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
    llm_seconds: float | None = None
    tts_seconds: float | None = None
    total_seconds: float | None = None

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
        voice_cache_dir: str | Path = DEFAULT_VOICE_CACHE_DIR,
    ) -> None:
        self.model_name = str(model_name).strip()
        self.speaker_wavs = resolve_speaker_wavs(speaker_wavs)
        self.language = str(language).strip() or "es"
        self.use_gpu = _detect_gpu() if use_gpu is None else bool(use_gpu)
        self.split_sentences = bool(split_sentences)
        self.voice_cache_dir = Path(voice_cache_dir).expanduser().resolve()

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

    def prepare(self, warmup_text: str | None = None) -> "XTTSVoiceSynthesizer":
        del warmup_text
        self._load_model()
        return self

    def _voice_cache_speaker_id(self, speaker_wavs: Sequence[Path]) -> str:
        digest = hashlib.sha1()
        digest.update(self.model_name.encode("utf-8"))
        digest.update(self.language.encode("utf-8"))
        for path in speaker_wavs:
            stat = path.stat()
            digest.update(str(path).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        return f"voice-{digest.hexdigest()[:16]}"

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
        self.voice_cache_dir.mkdir(parents=True, exist_ok=True)
        speaker_id = self._voice_cache_speaker_id(resolved_speaker_wavs)
        voice_cache_path = self.voice_cache_dir / f"{speaker_id}.pth"
        tts_kwargs = {
            "text": clean_text,
            "file_path": str(output_path),
            "speaker": speaker_id,
            "voice_dir": str(self.voice_cache_dir),
            "language": (language or self.language),
            "split_sentences": (
                self.split_sentences
                if split_sentences is None
                else bool(split_sentences)
            ),
        }
        speaker_wav_paths = [str(path) for path in resolved_speaker_wavs]
        try:
            if voice_cache_path.exists():
                model.tts_to_file(
                    speaker_wav=None,
                    **tts_kwargs,
                )
            else:
                model.tts_to_file(
                    speaker_wav=speaker_wav_paths,
                    **tts_kwargs,
                )
        except Exception:
            model.tts_to_file(
                speaker_wav=speaker_wav_paths,
                **tts_kwargs,
            )
        if not output_path.exists():
            raise RuntimeError(
                f"XTTS no genero el fichero de salida esperado: {output_path}"
            )
        return output_path


def build_voice_synthesizer(
    *,
    tts_backend: str = DEFAULT_TTS_BACKEND,
    speaker_wavs: Sequence[str | Path] | str | Path | None = None,
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    tts_language: str = "es",
    use_gpu: bool | None = None,
    split_sentences: bool = True,
    qwen_design_model_name: str = DEFAULT_QWEN_VOICE_DESIGN_MODEL,
    qwen_clone_model_name: str = DEFAULT_QWEN_VOICE_CLONE_MODEL,
    qwen_voice_design_prompt: str = DEFAULT_QWEN_VOICE_DESIGN_PROMPT,
    qwen_reference_text: str = DEFAULT_QWEN_REFERENCE_TEXT,
    qwen_use_flash_attention: bool = DEFAULT_QWEN_USE_FLASH_ATTENTION,
    qwen_cpp_threads: int = DEFAULT_QWEN_CPP_THREADS,
    qwen_cpp_repo_dir: str | Path | None = None,
    qwen_cpp_model_dir: str | Path | None = None,
):
    backend = str(tts_backend or DEFAULT_TTS_BACKEND).strip().lower()
    if backend == "xtts":
        return XTTSVoiceSynthesizer(
            model_name=tts_model,
            speaker_wavs=speaker_wavs,
            language=tts_language,
            use_gpu=use_gpu,
            split_sentences=split_sentences,
        )
    if backend == "qwen":
        from .experimental.qwen_voice import QwenVoiceDesignSynthesizer

        return QwenVoiceDesignSynthesizer(
            design_model_name=qwen_design_model_name,
            clone_model_name=qwen_clone_model_name,
            voice_design_prompt=qwen_voice_design_prompt,
            reference_text=qwen_reference_text,
            language=tts_language,
            use_gpu=use_gpu,
            use_flash_attention=qwen_use_flash_attention,
        )
    if backend == "qwen_cpp":
        from .experimental.qwen_voice import QwenCppSynthesizer

        return QwenCppSynthesizer(
            language=tts_language,
            n_threads=qwen_cpp_threads,
            speaker_wavs=speaker_wavs,
            design_model_name=qwen_design_model_name,
            clone_model_name=qwen_clone_model_name,
            voice_design_prompt=qwen_voice_design_prompt,
            reference_text=qwen_reference_text,
            model_dir=qwen_cpp_model_dir,
            repo_dir=qwen_cpp_repo_dir,
            use_gpu=use_gpu,
        )
    raise ValueError(
        f"`tts_backend` no soportado: {backend}. Usa `xtts`, `qwen` o `qwen_cpp`."
    )


class CommentaryAudioPipeline:
    def __init__(
        self,
        commentary_generator: OllamaCommentaryGenerator | None = None,
        voice_synthesizer: Any | None = None,
        output_dir: str | Path = DEFAULT_AUDIO_OUTPUT_DIR,
    ) -> None:
        self.commentary_generator = commentary_generator or OllamaCommentaryGenerator()
        self.voice_synthesizer = voice_synthesizer or build_voice_synthesizer()
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

    def prepare(
        self,
        warmup_event: CommentaryEvent | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        llm_seconds = None
        tts_seconds = None
        commentary = None

        if hasattr(self.commentary_generator, "prepare"):
            llm_start = time.perf_counter()
            warmup_result = self.commentary_generator.prepare(warmup_event)
            llm_seconds = time.perf_counter() - llm_start
            commentary = warmup_result.commentary

        if hasattr(self.voice_synthesizer, "prepare"):
            tts_start = time.perf_counter()
            self.voice_synthesizer.prepare(warmup_text=commentary)
            tts_seconds = time.perf_counter() - tts_start

        return {
            "commentary": commentary,
            "llm_seconds": llm_seconds,
            "tts_seconds": tts_seconds,
            "total_seconds": time.perf_counter() - total_start,
        }

    def generate_to_file(
        self,
        event: CommentaryEvent | dict[str, Any],
        audio_path: str | Path | None = None,
    ) -> CommentaryAudioResult:
        total_start = time.perf_counter()
        llm_start = time.perf_counter()
        commentary_result = self.commentary_generator.generate(event)
        llm_seconds = time.perf_counter() - llm_start
        output_path = (
            Path(audio_path).expanduser().resolve()
            if audio_path is not None
            else self.build_output_path(commentary_result.event)
        )
        tts_start = time.perf_counter()
        audio_path = self.voice_synthesizer.synthesize_to_file(
            commentary_result.commentary,
            output_path,
        )
        tts_seconds = time.perf_counter() - tts_start
        total_seconds = time.perf_counter() - total_start
        return CommentaryAudioResult(
            commentary_result=commentary_result,
            audio_path=audio_path,
            speaker_wavs=self.voice_synthesizer.speaker_wavs,
            tts_model=self.voice_synthesizer.model_name,
            language=self.voice_synthesizer.language,
            llm_seconds=llm_seconds,
            tts_seconds=tts_seconds,
            total_seconds=total_seconds,
        )
