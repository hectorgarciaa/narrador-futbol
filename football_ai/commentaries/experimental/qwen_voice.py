from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from ..voice import (
    DEFAULT_QWEN_CPP_MODEL_DIR,
    DEFAULT_QWEN_CPP_RUNTIME_DIR,
    DEFAULT_QWEN_CPP_THREADS,
    DEFAULT_QWEN_REFERENCE_TEXT,
    DEFAULT_QWEN_VOICE_CACHE_DIR,
    DEFAULT_QWEN_VOICE_CLONE_MODEL,
    DEFAULT_QWEN_VOICE_DESIGN_MODEL,
    DEFAULT_QWEN_VOICE_DESIGN_PROMPT,
    _detect_gpu,
    _find_first_existing_path,
    _flash_attention_available,
    _nvcc_available,
    _patch_qwen_tts_sox,
    _preferred_cmake_command,
    _preferred_cuda_architecture,
    _system_sox_available,
    _write_json,
    discover_default_speaker_wavs,
    normalize_qwen_language,
    PROJECT_ROOT,
    qwen_cpp_language_id,
    resolve_speaker_wavs,
)


class QwenVoiceDesignSynthesizer:
    _MODEL_CACHE: dict[tuple[str, bool, str, bool], Any] = {}

    def __init__(
        self,
        design_model_name: str = DEFAULT_QWEN_VOICE_DESIGN_MODEL,
        clone_model_name: str = DEFAULT_QWEN_VOICE_CLONE_MODEL,
        voice_design_prompt: str = DEFAULT_QWEN_VOICE_DESIGN_PROMPT,
        reference_text: str = DEFAULT_QWEN_REFERENCE_TEXT,
        language: str = "es",
        use_gpu: bool | None = None,
        torch_dtype_name: str = "bfloat16",
        use_flash_attention: bool = True,
        voice_cache_dir: str | Path = DEFAULT_QWEN_VOICE_CACHE_DIR,
    ) -> None:
        self.design_model_name = str(design_model_name).strip()
        self.clone_model_name = str(clone_model_name).strip()
        self.model_name = self.clone_model_name
        self.voice_design_prompt = str(voice_design_prompt).strip()
        self.reference_text = str(reference_text).strip()
        self.language = normalize_qwen_language(language)
        self.use_gpu = _detect_gpu() if use_gpu is None else bool(use_gpu)
        self.torch_dtype_name = str(torch_dtype_name).strip() or "bfloat16"
        self.use_flash_attention = bool(use_flash_attention)
        self.voice_cache_dir = Path(voice_cache_dir).expanduser().resolve()
        self.reference_audio_path = self._voice_asset_dir() / "voice_design_reference.wav"
        self.reference_meta_path = self._voice_asset_dir() / "voice_design_reference.json"
        self.prompt_cache_path = self._voice_asset_dir() / "voice_clone_prompt.pt"
        self.speaker_wavs: tuple[Path, ...] = (self.reference_audio_path,)
        self._voice_clone_prompt: Any | None = None
        self._runtime_warmup_done = False

    def _voice_id(self) -> str:
        digest = hashlib.sha1()
        digest.update(self.design_model_name.encode("utf-8"))
        digest.update(self.clone_model_name.encode("utf-8"))
        digest.update(self.language.encode("utf-8"))
        digest.update(self.voice_design_prompt.encode("utf-8"))
        digest.update(self.reference_text.encode("utf-8"))
        return f"qwen-voice-{digest.hexdigest()[:16]}"

    def _voice_asset_dir(self) -> Path:
        return self.voice_cache_dir / self._voice_id()

    def _resolve_torch_dtype(self):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Falta instalar torch para usar Qwen3-TTS."
            ) from exc

        normalized = self.torch_dtype_name.casefold()
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        return torch.float32

    def _load_model(self, model_name: str):
        cache_key = (
            model_name,
            self.use_gpu,
            self.torch_dtype_name.casefold(),
            self.use_flash_attention,
        )
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Falta instalar la dependencia de voz `qwen-tts`."
            ) from exc

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if not _system_sox_available():
            _patch_qwen_tts_sox()
        resolved_dtype = self._resolve_torch_dtype()
        load_kwargs: dict[str, Any] = {}
        if self.use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            load_kwargs["device_map"] = "cuda:0"
            load_kwargs["dtype"] = resolved_dtype
            if self.use_flash_attention and _flash_attention_available():
                load_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            load_kwargs["device_map"] = "cpu"
            load_kwargs["dtype"] = resolved_dtype

        try:
            model = Qwen3TTSModel.from_pretrained(model_name, **load_kwargs)
        except Exception as first_exc:
            retry_kwargs = dict(load_kwargs)
            retry_kwargs.pop("attn_implementation", None)
            if retry_kwargs != load_kwargs:
                try:
                    model = Qwen3TTSModel.from_pretrained(model_name, **retry_kwargs)
                except Exception:
                    raise first_exc
            else:
                raise

        if self.use_gpu:
            import torch

            try:
                model.device = torch.device("cuda:0")
            except Exception as exc:
                raise RuntimeError(
                    "No se pudo cargar Qwen3-TTS entero en GPU con `device_map=\"cuda:0\"`."
                ) from exc

        self._MODEL_CACHE[cache_key] = model
        return model

    def _load_design_model(self):
        return self._load_model(self.design_model_name)

    def _load_clone_model(self):
        return self._load_model(self.clone_model_name)

    def _release_cached_model(self, model_name: str) -> None:
        keys_to_remove = [
            key for key in self._MODEL_CACHE if key[0] == model_name
        ]
        for key in keys_to_remove:
            model = self._MODEL_CACHE.pop(key, None)
            if model is not None:
                del model
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _prompt_items_to_cpu(self, prompt_items: Sequence[Any]) -> list[Any]:
        from qwen_tts import VoiceClonePromptItem

        cpu_items = []
        for item in prompt_items:
            ref_code = item.ref_code
            if ref_code is not None:
                ref_code = ref_code.detach().cpu()
            cpu_items.append(
                VoiceClonePromptItem(
                    ref_code=ref_code,
                    ref_spk_embedding=item.ref_spk_embedding.detach().cpu(),
                    x_vector_only_mode=bool(item.x_vector_only_mode),
                    icl_mode=bool(item.icl_mode),
                    ref_text=item.ref_text,
                )
            )
        return cpu_items

    def _design_reference_voice(self) -> list[Any]:
        import soundfile as sf
        import torch

        asset_dir = self._voice_asset_dir()
        asset_dir.mkdir(parents=True, exist_ok=True)
        design_model = self._load_design_model()
        wavs, sample_rate = design_model.generate_voice_design(
            text=self.reference_text,
            language=self.language,
            instruct=self.voice_design_prompt,
        )
        sf.write(self.reference_audio_path, wavs[0], sample_rate)
        self._release_cached_model(self.design_model_name)
        clone_model = self._load_clone_model()
        prompt_items = clone_model.create_voice_clone_prompt(
            ref_audio=(wavs[0], sample_rate),
            ref_text=self.reference_text,
        )
        prompt_items_cpu = self._prompt_items_to_cpu(prompt_items)
        torch.save(prompt_items_cpu, self.prompt_cache_path)
        _write_json(
            self.reference_meta_path,
            {
                "design_model_name": self.design_model_name,
                "clone_model_name": self.clone_model_name,
                "language": self.language,
                "voice_design_prompt": self.voice_design_prompt,
                "reference_text": self.reference_text,
                "reference_audio_path": str(self.reference_audio_path),
                "prompt_cache_path": str(self.prompt_cache_path),
            },
        )
        self._voice_clone_prompt = prompt_items_cpu
        return prompt_items_cpu

    def _ensure_voice_clone_prompt(self) -> list[Any]:
        if self._voice_clone_prompt is not None:
            return self._voice_clone_prompt

        if self.prompt_cache_path.exists() and self.reference_audio_path.exists():
            try:
                import torch

                cached_prompt = torch.load(
                    self.prompt_cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if cached_prompt:
                    self._voice_clone_prompt = cached_prompt
                    return cached_prompt
            except Exception:
                pass

        return self._design_reference_voice()

    def _generate_audio(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> tuple[Any, int]:
        voice_clone_prompt = self._ensure_voice_clone_prompt()
        clone_model = self._load_clone_model()
        return clone_model.generate_voice_clone(
            text=text,
            language=normalize_qwen_language(language or self.language),
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=True,
        )

    def _warm_runtime(self, text: str) -> None:
        warm_text = str(text or "").strip()
        if not warm_text or self._runtime_warmup_done:
            return
        self._generate_audio(warm_text, language=self.language)
        self._runtime_warmup_done = True

    def prepare(self, warmup_text: str | None = None) -> "QwenVoiceDesignSynthesizer":
        self._ensure_voice_clone_prompt()
        self._load_clone_model()
        self._warm_runtime(warmup_text or "Pase corto de Modric.")
        return self

    def synthesize_to_file(
        self,
        text: str,
        file_path: str | Path,
        speaker_wavs: Sequence[str | Path] | str | Path | None = None,
        language: str | None = None,
        split_sentences: bool | None = None,
    ) -> Path:
        del speaker_wavs, split_sentences
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("`text` no puede ir vacio para sintetizar audio.")

        import soundfile as sf

        output_path = Path(file_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavs, sample_rate = self._generate_audio(
            clean_text,
            language=language,
        )
        sf.write(output_path, wavs[0], sample_rate)
        if not output_path.exists():
            raise RuntimeError(
                f"Qwen3-TTS no genero el fichero de salida esperado: {output_path}"
            )
        return output_path


class _QwenCppParams(ctypes.Structure):
    _fields_ = [
        ("max_audio_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("repetition_penalty", ctypes.c_float),
        ("language_id", ctypes.c_int32),
    ]


class _QwenCppAudio(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n_samples", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
    ]


class _QwenCppRuntime:
    def __init__(self, library_path: Path, model_dir: Path, n_threads: int) -> None:
        self.library_path = Path(library_path).expanduser().resolve()
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.n_threads = int(max(1, n_threads))
        self.lib = ctypes.CDLL(str(self.library_path))
        self._configure_signatures()
        self.handle = self.lib.qwen3_tts_create(
            str(self.model_dir).encode("utf-8"),
            self.n_threads,
        )
        if not self.handle:
            raise RuntimeError(
                f"No se pudo crear el runtime de qwen3-tts.cpp con {self.model_dir}"
            )

    def _configure_signatures(self) -> None:
        self.lib.qwen3_tts_create.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        self.lib.qwen3_tts_create.restype = ctypes.c_void_p
        self.lib.qwen3_tts_destroy.argtypes = [ctypes.c_void_p]
        self.lib.qwen3_tts_destroy.restype = None
        self.lib.qwen3_tts_default_params.argtypes = [ctypes.POINTER(_QwenCppParams)]
        self.lib.qwen3_tts_default_params.restype = None
        self.lib.qwen3_tts_get_error.argtypes = [ctypes.c_void_p]
        self.lib.qwen3_tts_get_error.restype = ctypes.c_char_p
        self.lib.qwen3_tts_extract_embedding_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
        ]
        self.lib.qwen3_tts_extract_embedding_file.restype = ctypes.c_int32
        self.lib.qwen3_tts_synthesize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_QwenCppParams),
        ]
        self.lib.qwen3_tts_synthesize.restype = ctypes.POINTER(_QwenCppAudio)
        self.lib.qwen3_tts_synthesize_with_embedding.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.POINTER(_QwenCppParams),
        ]
        self.lib.qwen3_tts_synthesize_with_embedding.restype = ctypes.POINTER(
            _QwenCppAudio
        )
        self.lib.qwen3_tts_free_audio.argtypes = [ctypes.POINTER(_QwenCppAudio)]
        self.lib.qwen3_tts_free_audio.restype = None

    def __del__(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            try:
                self.lib.qwen3_tts_destroy(handle)
            except Exception:
                pass
            self.handle = None

    def _last_error(self) -> str:
        raw = self.lib.qwen3_tts_get_error(self.handle)
        return raw.decode("utf-8", errors="replace") if raw else ""

    def default_params(self) -> _QwenCppParams:
        params = _QwenCppParams()
        self.lib.qwen3_tts_default_params(ctypes.byref(params))
        params.n_threads = self.n_threads
        return params

    def extract_embedding_file(self, reference_audio_path: Path) -> list[float]:
        max_size = 4096
        buffer = (ctypes.c_float * max_size)()
        size = self.lib.qwen3_tts_extract_embedding_file(
            self.handle,
            str(reference_audio_path).encode("utf-8"),
            buffer,
            max_size,
        )
        if size <= 0:
            raise RuntimeError(
                self._last_error()
                or f"No se pudo extraer el speaker embedding de {reference_audio_path}"
            )
        return [float(buffer[index]) for index in range(size)]

    def synthesize(
        self,
        text: str,
        *,
        language_id: int,
        embedding: Sequence[float] | None = None,
    ) -> tuple[Any, int]:
        import numpy as np

        params = self.default_params()
        params.language_id = int(language_id)
        text_bytes = str(text).encode("utf-8")

        if embedding:
            embedding_buffer = (ctypes.c_float * len(embedding))(*embedding)
            audio_ptr = self.lib.qwen3_tts_synthesize_with_embedding(
                self.handle,
                text_bytes,
                embedding_buffer,
                len(embedding),
                ctypes.byref(params),
            )
        else:
            audio_ptr = self.lib.qwen3_tts_synthesize(
                self.handle,
                text_bytes,
                ctypes.byref(params),
            )

        if not audio_ptr:
            raise RuntimeError(
                self._last_error() or "qwen3-tts.cpp no pudo sintetizar el audio."
            )

        try:
            audio = audio_ptr.contents
            samples = np.ctypeslib.as_array(
                audio.samples,
                shape=(audio.n_samples,),
            ).copy()
            return samples, int(audio.sample_rate)
        finally:
            self.lib.qwen3_tts_free_audio(audio_ptr)


class QwenCppSynthesizer:
    _RUNTIME_CACHE: dict[tuple[str, str, int], _QwenCppRuntime] = {}

    def __init__(
        self,
        *,
        language: str = "es",
        n_threads: int = DEFAULT_QWEN_CPP_THREADS,
        speaker_wavs: Sequence[str | Path] | str | Path | None = None,
        design_model_name: str = DEFAULT_QWEN_VOICE_DESIGN_MODEL,
        clone_model_name: str = DEFAULT_QWEN_VOICE_CLONE_MODEL,
        voice_design_prompt: str = DEFAULT_QWEN_VOICE_DESIGN_PROMPT,
        reference_text: str = DEFAULT_QWEN_REFERENCE_TEXT,
        voice_cache_dir: str | Path = DEFAULT_QWEN_VOICE_CACHE_DIR,
        runtime_dir: str | Path = DEFAULT_QWEN_CPP_RUNTIME_DIR,
        model_dir: str | Path | None = None,
        repo_dir: str | Path | None = None,
        use_gpu: bool | None = None,
    ) -> None:
        self.language = normalize_qwen_language(language)
        self.n_threads = int(max(1, n_threads))
        self.design_model_name = str(design_model_name).strip()
        self.clone_model_name = str(clone_model_name).strip()
        self.voice_design_prompt = str(voice_design_prompt).strip()
        self.reference_text = str(reference_text).strip()
        self.voice_cache_dir = Path(voice_cache_dir).expanduser().resolve()
        self.runtime_dir = Path(runtime_dir).expanduser().resolve()
        self.use_gpu = _detect_gpu() if use_gpu is None else bool(use_gpu)
        self.model_dir = (
            Path(model_dir).expanduser().resolve()
            if model_dir is not None
            else DEFAULT_QWEN_CPP_MODEL_DIR
        )
        self.repo_dir = (
            Path(repo_dir).expanduser().resolve()
            if repo_dir is not None
            else self._resolve_repo_dir()
        )
        self.model_name = "qwen3-tts.cpp/qwen3-tts-0.6b-q8_0"
        self.reference_audio_path = self._resolve_reference_audio(speaker_wavs)
        self.speaker_wavs = (
            (self.reference_audio_path,) if self.reference_audio_path is not None else ()
        )
        self.embedding_cache_path = self._embedding_cache_path()
        self._speaker_embedding: list[float] | None = None
        self._runtime_warmup_done = False
        self._last_runtime_mode: str | None = None

    def _resolve_repo_dir(self) -> Path:
        env_repo = os.environ.get("QWEN_CPP_REPO_DIR")
        candidates: list[str | Path] = []
        if env_repo:
            candidates.append(env_repo)
        candidates.extend(
            [
                PROJECT_ROOT / ".cache" / "qwen3-tts.cpp",
                PROJECT_ROOT / "third_party" / "qwen3-tts.cpp",
                Path("/tmp/qwen3-tts.cpp"),
            ]
        )
        repo_dir = _find_first_existing_path(candidates)
        if repo_dir is None:
            raise RuntimeError(
                "No se encontro el repo de `qwen3-tts.cpp`. "
                "Define `QWEN_CPP_REPO_DIR` o deja el clone en `/tmp/qwen3-tts.cpp`."
            )
        return repo_dir

    def _resolve_reference_audio(
        self,
        speaker_wavs: Sequence[str | Path] | str | Path | None,
    ) -> Path | None:
        if speaker_wavs is not None:
            resolved = resolve_speaker_wavs(speaker_wavs)
            return resolved[0]

        qwen_voice = QwenVoiceDesignSynthesizer(
            design_model_name=self.design_model_name,
            clone_model_name=self.clone_model_name,
            voice_design_prompt=self.voice_design_prompt,
            reference_text=self.reference_text,
            language=self.language,
            voice_cache_dir=self.voice_cache_dir,
        ).reference_audio_path
        if qwen_voice.exists():
            return qwen_voice

        candidates = sorted(
            self.voice_cache_dir.glob("qwen-voice-*/voice_design_reference.wav"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        if candidates:
            return candidates[0]

        discovered = discover_default_speaker_wavs()
        return discovered[0] if discovered else None

    def _embedding_cache_path(self) -> Path | None:
        if self.reference_audio_path is None:
            return None
        digest = hashlib.sha1()
        stat = self.reference_audio_path.stat()
        digest.update(str(self.reference_audio_path).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(self.language.encode("utf-8"))
        cache_dir = self.runtime_dir / "voice_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"embedding-{digest.hexdigest()[:16]}.json"

    def _required_model_files(self) -> tuple[Path, Path]:
        return (
            self.model_dir / "qwen3-tts-0.6b-q8_0.gguf",
            self.model_dir / "qwen3-tts-tokenizer-f16.gguf",
        )

    def _find_source_asset_dir(self, env_name: str, *fallbacks: str | Path) -> Path | None:
        candidates: list[str | Path] = []
        value = os.environ.get(env_name)
        if value:
            candidates.append(value)
        candidates.extend(fallbacks)
        return _find_first_existing_path(candidates)

    def _run_command(self, cmd: Sequence[str], cwd: Path) -> None:
        subprocess.run(
            list(cmd),
            cwd=str(cwd),
            check=True,
        )

    def _prefer_cuda_build(self) -> bool:
        if not self.use_gpu:
            return False
        if os.name != "posix":
            return False
        return _nvcc_available()

    def _ensure_model_artifacts(self) -> None:
        tts_model_path, tokenizer_model_path = self._required_model_files()
        if tts_model_path.exists() and tokenizer_model_path.exists():
            return

        self.model_dir.mkdir(parents=True, exist_ok=True)
        base_assets_dir = self._find_source_asset_dir(
            "QWEN_CPP_BASE_ASSETS_DIR",
            Path("/tmp/qwen3_tts_models/Qwen3-TTS-12Hz-0.6B-Base"),
            Path("/tmp/qwen3_tts_models/Qwen3-TTS-0.6B-Base"),
        )
        tokenizer_assets_dir = self._find_source_asset_dir(
            "QWEN_CPP_TOKENIZER_ASSETS_DIR",
            Path("/tmp/qwen3_tts_models/Qwen3-TTS-Tokenizer-12Hz"),
        )
        if base_assets_dir is None or tokenizer_assets_dir is None:
            raise RuntimeError(
                "Faltan los assets de Hugging Face para convertir `qwen3-tts.cpp`. "
                "Define `QWEN_CPP_BASE_ASSETS_DIR` y `QWEN_CPP_TOKENIZER_ASSETS_DIR` "
                "o deja los modelos descargados en `/tmp/qwen3_tts_models/`."
            )

        python_executable = sys.executable
        if not tts_model_path.exists():
            self._run_command(
                [
                    python_executable,
                    str(self.repo_dir / "scripts" / "convert_tts_to_gguf.py"),
                    "--input",
                    str(base_assets_dir),
                    "--output",
                    str(tts_model_path),
                    "--type",
                    "q8_0",
                ],
                cwd=self.repo_dir,
            )
        if not tokenizer_model_path.exists():
            self._run_command(
                [
                    python_executable,
                    str(self.repo_dir / "scripts" / "convert_tokenizer_to_gguf.py"),
                    "--input",
                    str(tokenizer_assets_dir),
                    "--output",
                    str(tokenizer_model_path),
                    "--type",
                    "f16",
                ],
                cwd=self.repo_dir,
            )

    def _qwen_cpp_build_root(self, mode: str) -> Path:
        return self.runtime_dir / "native" / mode

    def _write_wrapper_cmakelists(self, wrapper_dir: Path, use_cuda: bool) -> Path:
        repo_dir = self.repo_dir.as_posix()
        cuda_block = ""
        if use_cuda:
            ggml_cuda_lib = (
                self.repo_dir / "ggml" / "build" / "src" / "ggml-cuda" / "libggml-cuda.a"
            ).as_posix()
            cuda_block = f"""
find_package(CUDAToolkit REQUIRED)
set(QWEN3_TTS_GGML_CUDA_LIB "{ggml_cuda_lib}")
if(EXISTS "${{QWEN3_TTS_GGML_CUDA_LIB}}")
  foreach(target_name IN ITEMS text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder)
    target_link_libraries(${{target_name}} PUBLIC
      "${{QWEN3_TTS_GGML_CUDA_LIB}}"
      CUDA::cudart
      CUDA::cublas
      CUDA::cuda_driver
    )
  endforeach()
endif()
"""
        content = f"""cmake_minimum_required(VERSION 3.18)
project(qwen3_tts_wrapper LANGUAGES CXX)

add_subdirectory("{repo_dir}" qwen3_tts_src)
{cuda_block}
"""
        wrapper_path = wrapper_dir / "CMakeLists.txt"
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        wrapper_path.write_text(content, encoding="utf-8")
        return wrapper_path

    def _configure_ggml(self, *, use_cuda: bool, cmake_command: Sequence[str]) -> None:
        cmd = [
            *cmake_command,
            "-S",
            str(self.repo_dir / "ggml"),
            "-B",
            str(self.repo_dir / "ggml" / "build"),
            "-DGGML_BUILD_TESTS=OFF",
            "-DGGML_BUILD_EXAMPLES=OFF",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DGGML_OPENMP=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        ]
        if use_cuda:
            cmd.extend(
                [
                    "-DGGML_CUDA=ON",
                    f"-DCMAKE_CUDA_ARCHITECTURES={_preferred_cuda_architecture()}",
                ]
            )
        self._run_command(cmd, cwd=self.repo_dir)

    def _ggml_artifacts_ready(self, *, use_cuda: bool) -> bool:
        build_root = self.repo_dir / "ggml" / "build" / "src"
        required = [
            build_root / "libggml.a",
            build_root / "libggml-base.a",
            build_root / "libggml-cpu.a",
        ]
        if use_cuda:
            required.append(build_root / "ggml-cuda" / "libggml-cuda.a")
        return all(path.exists() for path in required)

    def _build_wrapper_targets(
        self,
        *,
        build_dir: Path,
        cmake_command: Sequence[str],
    ) -> None:
        self._run_command(
            [
                *cmake_command,
                "-S",
                str(build_dir),
                "-B",
                str(build_dir / "cmake-build"),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            ],
            cwd=build_dir,
        )
        self._run_command(
            [
                *cmake_command,
                "--build",
                str(build_dir / "cmake-build"),
                "--target",
                "qwen3tts_shared",
                "-j4",
            ],
            cwd=build_dir,
        )

    def _ensure_library_for_mode(self, mode: str) -> Path:
        library_path = (
            self._qwen_cpp_build_root(mode)
            / "cmake-build"
            / "qwen3_tts_src"
            / "libqwen3tts.so"
        )
        if library_path.exists():
            return library_path
        cmake_command = _preferred_cmake_command()
        use_cuda = mode == "cuda"
        if not self._ggml_artifacts_ready(use_cuda=use_cuda):
            self._configure_ggml(use_cuda=use_cuda, cmake_command=cmake_command)
            self._run_command(
                [
                    *cmake_command,
                    "--build",
                    str(self.repo_dir / "ggml" / "build"),
                    "--target",
                    "ggml",
                    "ggml-base",
                    "ggml-cpu",
                    *(["ggml-cuda"] if use_cuda else []),
                    "-j4",
                ],
                cwd=self.repo_dir,
            )
        build_root = self._qwen_cpp_build_root(mode)
        self._write_wrapper_cmakelists(build_root, use_cuda=use_cuda)
        self._build_wrapper_targets(build_dir=build_root, cmake_command=cmake_command)
        if not library_path.exists():
            raise RuntimeError(
                f"No se pudo construir `libqwen3tts.so` en {library_path}"
            )
        return library_path

    def _ensure_library(self) -> Path:
        preferred_modes = ["cuda", "cpu"] if self._prefer_cuda_build() else ["cpu"]
        last_error: Exception | None = None
        for mode in preferred_modes:
            try:
                library_path = self._ensure_library_for_mode(mode)
                self._last_runtime_mode = mode
                return library_path
            except Exception as exc:
                last_error = exc
                if mode == "cpu":
                    break
        assert last_error is not None
        raise last_error

    def _runtime_cache_key(self, library_path: Path) -> tuple[str, str, int]:
        return (str(library_path), str(self.model_dir), self.n_threads)

    def _load_runtime(self) -> _QwenCppRuntime:
        self._ensure_model_artifacts()
        library_path = self._ensure_library()
        cache_key = self._runtime_cache_key(library_path)
        cached = self._RUNTIME_CACHE.get(cache_key)
        if cached is not None:
            return cached
        runtime = _QwenCppRuntime(
            library_path=library_path,
            model_dir=self.model_dir,
            n_threads=self.n_threads,
        )
        self._RUNTIME_CACHE[cache_key] = runtime
        return runtime

    def _load_cached_embedding(self) -> list[float] | None:
        if self.embedding_cache_path is None or not self.embedding_cache_path.exists():
            return None
        try:
            with open(self.embedding_cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            values = payload.get("embedding")
            if isinstance(values, list) and values:
                return [float(item) for item in values]
        except Exception:
            return None
        return None

    def _ensure_embedding(self) -> list[float] | None:
        if self.reference_audio_path is None:
            return None
        if self._speaker_embedding is not None:
            return self._speaker_embedding

        cached = self._load_cached_embedding()
        if cached:
            self._speaker_embedding = cached
            return cached

        runtime = self._load_runtime()
        embedding = runtime.extract_embedding_file(self.reference_audio_path)
        if self.embedding_cache_path is not None:
            _write_json(
                self.embedding_cache_path,
                {
                    "reference_audio_path": str(self.reference_audio_path),
                    "embedding_size": len(embedding),
                    "embedding": embedding,
                },
            )
        self._speaker_embedding = embedding
        return embedding

    def _warm_runtime(self, text: str) -> None:
        warm_text = str(text or "").strip()
        if not warm_text or self._runtime_warmup_done:
            return
        runtime = self._load_runtime()
        runtime.synthesize(
            warm_text,
            language_id=qwen_cpp_language_id(self.language),
            embedding=self._ensure_embedding(),
        )
        self._runtime_warmup_done = True

    def prepare(self, warmup_text: str | None = None) -> "QwenCppSynthesizer":
        self._load_runtime()
        self._ensure_embedding()
        self._warm_runtime(warmup_text or "Pase corto de Modric.")
        return self

    def synthesize_to_file(
        self,
        text: str,
        file_path: str | Path,
        speaker_wavs: Sequence[str | Path] | str | Path | None = None,
        language: str | None = None,
        split_sentences: bool | None = None,
    ) -> Path:
        del split_sentences
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("`text` no puede ir vacio para sintetizar audio.")

        import soundfile as sf

        if speaker_wavs is not None:
            resolved = resolve_speaker_wavs(speaker_wavs)
            self.reference_audio_path = resolved[0]
            self.speaker_wavs = (self.reference_audio_path,)
            self.embedding_cache_path = self._embedding_cache_path()
            self._speaker_embedding = None

        runtime = self._load_runtime()
        samples, sample_rate = runtime.synthesize(
            clean_text,
            language_id=qwen_cpp_language_id(language or self.language),
            embedding=self._ensure_embedding(),
        )
        output_path = Path(file_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, samples, sample_rate)
        if not output_path.exists():
            raise RuntimeError(
                f"qwen3-tts.cpp no genero el fichero de salida esperado: {output_path}"
            )
        return output_path
