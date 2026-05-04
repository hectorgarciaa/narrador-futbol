"""Synthetic football commentary generation using llama.cpp (Gemma 4 GGUF)."""

from .generator import (
    CommentaryEvent,
    CommentaryGenerationResult,
    CommentaryGenerator,
    CommentaryLLMRunResult,
    CommentaryPromptBuilder,
)
from .deferred_media import (
    DeferredCommentaryAssemblyResult,
    assemble_deferred_commentary_video,
    build_commentary_track_from_manifest,
    mux_commentary_track_into_video,
    probe_video_duration_seconds,
)
from .experimental import QwenCppSynthesizer, QwenVoiceDesignSynthesizer
from .elevenlabs_tts import ElevenLabsVoiceSynthesizer
from .voice import (
    AlternatingVoiceSynthesizer,
    CommentaryAudioPipeline,
    CommentaryAudioResult,
    DEFAULT_ELEVENLABS_MODEL_ID,
    DEFAULT_ELEVENLABS_OUTPUT_FORMAT,
    DEFAULT_QWEN_FEMALE_VOICE_DESIGN_PROMPT,
    DEFAULT_QWEN_VOICE_DESIGN_PROMPT,
    TTS_BACKEND_CHOICES,
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
    "CommentaryGenerator",
    "DeferredCommentaryAssemblyResult",
    "assemble_deferred_commentary_video",
    "build_commentary_track_from_manifest",
    "mux_commentary_track_into_video",
    "probe_video_duration_seconds",
    "AlternatingVoiceSynthesizer",
    "CommentaryAudioPipeline",
    "CommentaryAudioResult",
    "DEFAULT_ELEVENLABS_MODEL_ID",
    "DEFAULT_ELEVENLABS_OUTPUT_FORMAT",
    "DEFAULT_QWEN_FEMALE_VOICE_DESIGN_PROMPT",
    "DEFAULT_QWEN_VOICE_DESIGN_PROMPT",
    "ElevenLabsVoiceSynthesizer",
    "QwenCppSynthesizer",
    "QwenVoiceDesignSynthesizer",
    "TTS_BACKEND_CHOICES",
    "XTTSVoiceSynthesizer",
    "build_voice_synthesizer",
    "discover_default_speaker_wavs",
    "qwen_voice_reference_path",
    "resolve_speaker_wavs",
]
