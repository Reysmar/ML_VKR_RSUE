"""Обёртка для предобученного ResNet на ImageNet."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from .base import BaseModelWrapper

# URL с человекочитаемыми метками ImageNet (1000 строк, индекс = номер строки)
IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet_simple_labels.txt"
)


def _load_imagenet_labels() -> list[str]:
    """Загрузить список меток ImageNet (1000 классов)."""
    try:
        import urllib.request
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=10) as resp:
            text = resp.read().decode("utf-8")
        labels = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if len(labels) >= 1000:
            return labels[:1000]
    except Exception:
        pass
    return [f"Class {i}" for i in range(1000)]


class ResNetImageNetWrapper(BaseModelWrapper):
    """ResNet-50, предобученный на ImageNet (1000 классов)."""

    name = "ResNet-50 ImageNet"
    top_k = 5

    def __init__(self) -> None:
        self._model = None
        self._labels: list[str] = []
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self) -> None:
        if self._model is not None:
            return
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self._model = models.resnet50(weights=weights)
        self._model.eval()
        self._labels = _load_imagenet_labels()

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        if self._model is None:
            self.load()
        # image: (H, W, 3) uint8 or float
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(image)
        x = self._transform(pil).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        confidences, indices = torch.topk(probs, min(self.top_k, probs.shape[0]))
        indices = indices.cpu().numpy().tolist()
        confidences = confidences.cpu().numpy().tolist()
        labels = [self._labels[i] for i in indices]
        return {
            "top_classes": indices,
            "top_labels": labels,
            "top_confidences": confidences,
        }
