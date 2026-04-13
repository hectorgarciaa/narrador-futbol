"""Synthetic football commentary generation on top of Ollama, llama.cpp or Transformers."""

from .generator import (
    CommentaryEvent,
    CommentaryGenerationResult,
    CommentaryLLMRunResult,
    CommentaryPromptBuilder,
    OllamaCommentaryGenerator,
)
from .llama_cpp_backend import LlamaCppCommentaryGenerator
from .transformers_backend import (
    DEFAULT_HYMBA_MODEL,
    TransformersCommentaryGenerator,
)
from .server import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    CommentaryHTTPServer,
    CommentaryHTTPService,
    CommentaryServiceResult,
    create_http_server,
)
from .deferred_media import (
    DeferredCommentaryAssemblyResult,
    assemble_deferred_commentary_video,
    build_commentary_track_from_manifest,
    mux_commentary_track_into_video,
    probe_video_duration_seconds,
)
from .experimental import QwenCppSynthesizer, QwenVoiceDesignSynthesizer
from .voice import (
    AlternatingVoiceSynthesizer,
    CommentaryAudioPipeline,
    CommentaryAudioResult,
    DEFAULT_QWEN_FEMALE_VOICE_DESIGN_PROMPT,
    DEFAULT_QWEN_VOICE_DESIGN_PROMPT,
    XTTSVoiceSynthesizer,
    build_voice_synthesizer,
    discover_default_speaker_wavs,
    qwen_voice_reference_path,
    resolve_speaker_wavs,
)

__all__ = [
    "CommentaryEvent",
    "CommentaryGenerationResult",
    "CommentaryLLMRunResult",
    "CommentaryPromptBuilder",
    "OllamaCommentaryGenerator",
    "LlamaCppCommentaryGenerator",
    "DEFAULT_HYMBA_MODEL",
    "TransformersCommentaryGenerator",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "CommentaryHTTPServer",
    "CommentaryHTTPService",
    "CommentaryServiceResult",
    "create_http_server",
    "DeferredCommentaryAssemblyResult",
    "assemble_deferred_commentary_video",
    "build_commentary_track_from_manifest",
    "mux_commentary_track_into_video",
    "probe_video_duration_seconds",
    "AlternatingVoiceSynthesizer",
    "CommentaryAudioPipeline",
    "CommentaryAudioResult",
    "DEFAULT_QWEN_FEMALE_VOICE_DESIGN_PROMPT",
    "DEFAULT_QWEN_VOICE_DESIGN_PROMPT",
    "QwenCppSynthesizer",
    "QwenVoiceDesignSynthesizer",
    "XTTSVoiceSynthesizer",
    "build_voice_synthesizer",
    "discover_default_speaker_wavs",
    "qwen_voice_reference_path",
    "resolve_speaker_wavs",
]
