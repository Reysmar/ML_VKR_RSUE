"""Базовый класс обёртки модели для единообразного API."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModelWrapper(ABC):
    """Абстрактная обёртка: загрузка модели и предсказание по изображению."""

    name: str = ""

    @abstractmethod
    def load(self) -> None:
        """Загрузить веса/модель (вызывается один раз перед использованием)."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """
        Предсказание по одному изображению.

        Args:
            image: RGB изображение (H, W, 3), uint8 [0, 255] или float [0, 1].

        Returns:
            Словарь с ключами:
            - "top_classes": list[int] — индексы топ-N классов (от самого вероятного).
            - "top_labels": list[str] — метки классов (например названия).
            - "top_confidences": list[float] — уверенности для топ-N.
            - Опционально: "logits", "embeddings" для расширенных метрик.
        """
        pass
