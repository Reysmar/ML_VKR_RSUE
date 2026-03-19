"""Обёртки моделей для единообразного predict API."""

from .base import BaseModelWrapper
from .resnet_imagenet import ResNetImageNetWrapper
from .torchvision_imagenet import TorchvisionImageNetWrapper

__all__ = [
    "BaseModelWrapper",
    "ResNetImageNetWrapper",
    "TorchvisionImageNetWrapper",
]
