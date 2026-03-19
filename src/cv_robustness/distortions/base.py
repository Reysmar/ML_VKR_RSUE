"""Базовый класс для искажений изображений."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDistortion(ABC):
    """Абстрактный искажатель: применяет преобразование к изображению с заданными параметрами."""

    name: str = ""
    description: str = ""

    # Список параметров для UI: (имя, тип, default, min, max, label)
    # Например: [("sigma", "float", 0.05, 0.0, 0.5, "Сила шума")]
    params_spec: list[tuple[str, str, Any, Any, Any, str]] = []

    @abstractmethod
    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        """
        Применить искажение к изображению.

        Args:
            image: RGB изображение, shape (H, W, 3), dtype uint8 или float [0,1].
            **params: параметры искажения (имена из params_spec).

        Returns:
            Искажённое изображение в том же формате и диапазоне.
        """
        pass

    def get_default_params(self) -> dict[str, Any]:
        """Параметры по умолчанию для UI."""
        return {spec[0]: spec[2] for spec in self.params_spec}
