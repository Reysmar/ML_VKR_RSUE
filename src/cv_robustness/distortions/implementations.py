"""Реализации искажений для тестирования устойчивости моделей."""

import io
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .base import BaseDistortion


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Привести изображение к uint8 RGB (H, W, 3)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    return image


class GaussianNoiseDistortion(BaseDistortion):
    """Гауссов шум."""

    name = "Гауссов шум"
    description = "Добавление гауссова шума к изображению"
    params_spec = [
        ("sigma", "float", 0.08, 0.01, 0.5, "Стандартное отклонение шума (доля от 255)"),
    ]

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        image = _ensure_uint8_rgb(image)
        sigma = float(params.get("sigma", 0.08))
        noise_std = sigma * 255
        noise = np.random.randn(*image.shape).astype(np.float64) * noise_std
        out = image.astype(np.float64) + noise
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out


class GaussianBlurDistortion(BaseDistortion):
    """Размытие по Гауссу."""

    name = "Размытие по Гауссу"
    description = "Gaussian blur"
    params_spec = [
        ("kernel_size", "int", 11, 1, 31, "Размер ядра (нечётный)"),
        ("sigma", "float", 2.0, 0.5, 15.0, "Сигма размытия"),
    ]

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        image = _ensure_uint8_rgb(image)
        k = int(params.get("kernel_size", 11))
        sigma = float(params.get("sigma", 2.0))
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(image, (k, k), sigma)


class JPEGCompressionDistortion(BaseDistortion):
    """Сжатие JPEG."""

    name = "Сжатие JPEG"
    description = "Имитация потери при сжатии JPEG"
    params_spec = [
        ("quality", "int", 70, 5, 95, "Качество JPEG (5–95)"),
    ]

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        image = _ensure_uint8_rgb(image)
        quality = int(params.get("quality", 70))
        pil = Image.fromarray(image)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        out = np.array(Image.open(buf).convert("RGB"))
        return out


class BrightnessContrastDistortion(BaseDistortion):
    """Яркость и контраст."""

    name = "Яркость и контраст"
    description = "Линейное изменение яркости и контраста"
    params_spec = [
        ("brightness", "float", 0.0, -0.5, 0.5, "Смещение яркости"),
        ("contrast", "float", 0.0, -0.5, 0.5, "Изменение контраста (0 = без изменений)"),
    ]

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        image = _ensure_uint8_rgb(image)
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        # contrast: factor around 1; brightness: add to pixel
        factor = 1.0 + contrast
        delta = brightness * 255
        out = image.astype(np.float64) * factor + delta
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out
