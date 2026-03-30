"""Synthetic football commentary generation on top of Ollama."""

from .generator import (
    CommentaryEvent,
    CommentaryGenerationResult,
    CommentaryPromptBuilder,
    OllamaCommentaryGenerator,
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
    "CommentaryPromptBuilder",
    "OllamaCommentaryGenerator",
    "CommentaryAudioPipeline",
    "CommentaryAudioResult",
    "XTTSVoiceSynthesizer",
    "discover_default_speaker_wavs",
    "resolve_speaker_wavs",
]
