"""Backends experimentales y utilidades de benchmarking para comentarios."""

from .qwen_voice import QwenCppSynthesizer, QwenVoiceDesignSynthesizer

__all__ = [
    "QwenCppSynthesizer",
    "QwenVoiceDesignSynthesizer",
]
