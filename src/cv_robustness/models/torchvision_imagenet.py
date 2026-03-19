"""Универсальная обёртка для моделей torchvision, предобученных на ImageNet."""

from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from .base import BaseModelWrapper
from .resnet_imagenet import _load_imagenet_labels


class TorchvisionImageNetWrapper(BaseModelWrapper):
    """Обёртка для любой модели torchvision с весами ImageNet."""

    top_k = 5

    def __init__(
        self,
        name: str,
        model_fn: Callable[..., torch.nn.Module],
        weights_enum: models.WeightsEnum,
    ) -> None:
        """
        Args:
            name: Человекочитаемое имя модели для отображения в UI.
            model_fn: фабрика модели, например torchvision.models.resnet18.
            weights_enum: соответствующий Weights (например ResNet18_Weights.IMAGENET1K_V1).
        """
        self.name = name
        self._model_fn = model_fn
        self._weights_enum = weights_enum
        self._model: torch.nn.Module | None = None
        self._labels: list[str] = []
        self._transform = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._model = self._model_fn(weights=self._weights_enum)
        self._model.eval()
        self._labels = _load_imagenet_labels()
        # Используем стандартные transforms из weights
        self._transform = self._weights_enum.transforms()

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        if self._model is None:
            self.load()
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(image)
        x = self._transform(pil).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        confidences, indices = torch.topk(probs, min(self.top_k, probs.shape[0]))
        indices_list = indices.cpu().numpy().tolist()
        confidences_list = confidences.cpu().numpy().tolist()
        labels = [self._labels[i] for i in indices_list]
        return {
            "top_classes": indices_list,
            "top_labels": labels,
            "top_confidences": confidences_list,
        }

