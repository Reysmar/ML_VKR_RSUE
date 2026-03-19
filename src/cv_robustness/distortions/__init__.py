"""Искажения входных изображений для тестирования устойчивости."""

from .base import BaseDistortion
from .implementations import (
    GaussianNoiseDistortion,
    GaussianBlurDistortion,
    JPEGCompressionDistortion,
    BrightnessContrastDistortion,
)

__all__ = [
    "BaseDistortion",
    "GaussianNoiseDistortion",
    "GaussianBlurDistortion",
    "JPEGCompressionDistortion",
    "BrightnessContrastDistortion",
]
