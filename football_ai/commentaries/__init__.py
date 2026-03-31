"""Synthetic football commentary generation on top of Ollama or Transformers."""

from .generator import (
    CommentaryEvent,
    CommentaryGenerationResult,
    CommentaryLLMRunResult,
    CommentaryPromptBuilder,
    OllamaCommentaryGenerator,
)
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
from .voice import (
    CommentaryAudioPipeline,
    CommentaryAudioResult,
    XTTSVoiceSynthesizer,
    discover_default_speaker_wavs,
    resolve_speaker_wavs,
)

__all__ = [
    "CommentaryEvent",
    "CommentaryGenerationResult",
    "CommentaryLLMRunResult",
    "CommentaryPromptBuilder",
    "OllamaCommentaryGenerator",
    "DEFAULT_HYMBA_MODEL",
    "TransformersCommentaryGenerator",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "CommentaryHTTPServer",
    "CommentaryHTTPService",
    "CommentaryServiceResult",
    "create_http_server",
    "CommentaryAudioPipeline",
    "CommentaryAudioResult",
    "XTTSVoiceSynthesizer",
    "discover_default_speaker_wavs",
    "resolve_speaker_wavs",
]
