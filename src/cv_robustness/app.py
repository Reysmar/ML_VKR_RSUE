"""Gradio-интерфейс MVP: загрузка изображения, искажение, сравнение предсказаний и метрик."""

from __future__ import annotations

import os
from typing import Any, Callable

import gradio as gr
import numpy as np
from torchvision import models as tv_models

from .auto_study import run_auto_study, save_results_to_csv
from .distortions import (
    BrightnessContrastDistortion,
    GaussianBlurDistortion,
    GaussianNoiseDistortion,
    JPEGCompressionDistortion,
)
from .metrics import compute_robustness_metrics
from .models import ResNetImageNetWrapper, TorchvisionImageNetWrapper


# Реестр моделей и искажений для UI
MODELS: dict[str, Callable[[], Any]] = {
    "ResNet-50 ImageNet": ResNetImageNetWrapper,
    "ResNet-18 ImageNet": lambda: TorchvisionImageNetWrapper(
        "ResNet-18 ImageNet",
        tv_models.resnet18,
        tv_models.ResNet18_Weights.IMAGENET1K_V1,
    ),
    "MobileNetV3-Small ImageNet": lambda: TorchvisionImageNetWrapper(
        "MobileNetV3-Small ImageNet",
        tv_models.mobilenet_v3_small,
        tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    ),
    "EfficientNet-B0 ImageNet": lambda: TorchvisionImageNetWrapper(
        "EfficientNet-B0 ImageNet",
        tv_models.efficientnet_b0,
        tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    ),
}

DISTORTIONS = {
    "Гауссов шум": GaussianNoiseDistortion(),
    "Размытие по Гауссу": GaussianBlurDistortion(),
    "Сжатие JPEG": JPEGCompressionDistortion(),
    "Яркость и контраст": BrightnessContrastDistortion(),
}


def _get_model(name: str):
    factory = MODELS.get(name)
    if factory is None:
        factory = ResNetImageNetWrapper
    model = factory()
    model.load()
    return model


def _image_to_numpy(image) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image
    # Gradio Image can be PIL or path
    from PIL import Image

    if isinstance(image, Image.Image):
        return np.array(image)
    return np.array(Image.open(image).convert("RGB"))


def _params_from_sliders(distortion_name: str, p1: float, p2: float, p3: float) -> dict[str, Any]:
    """Преобразовать слайдеры 0..1 в параметры выбранного искажения."""
    d = DISTORTIONS.get(distortion_name)
    if d is None:
        return {}
    spec = d.params_spec
    params: dict[str, Any] = {}
    values = [p1, p2, p3]
    for i, (name, typ, default, lo, hi, _label) in enumerate(spec):
        if i >= len(values):
            params[name] = default
            continue
        v = values[i]
        if typ == "float":
            params[name] = lo + v * (hi - lo)
        else:
            params[name] = int(lo + v * (hi - lo))
    return params


def analyze(
    image,
    model_name: str,
    distortion_name: str,
    param1: float,
    param2: float,
    param3: float,
):
    """Интерактивный анализ одного набора параметров искажения."""
    img = _image_to_numpy(image)
    if img is None:
        return None, None, "Загрузите изображение."
    model = _get_model(model_name)
    distortion = DISTORTIONS.get(distortion_name)
    if distortion is None:
        return None, None, "Выберите искажение."
    params = _params_from_sliders(distortion_name, param1, param2, param3)
    distorted = distortion.apply(img, **params)
    pred_orig = model.predict(img)
    pred_dist = model.predict(distorted)
    metrics = compute_robustness_metrics(pred_orig, pred_dist)
    lines = [
        "**Исходное изображение:**",
        f"Класс: {pred_orig['top_labels'][0]}",
        f"Уверенность: {pred_orig['top_confidences'][0]:.2%}",
        "",
        "**Искажённое изображение:**",
        f"Класс: {pred_dist['top_labels'][0]}",
        f"Уверенность: {pred_dist['top_confidences'][0]:.2%}",
        "",
        "**Метрики устойчивости:**",
        f"Топ-1 совпадает: {'Да' if metrics['top1_match'] else 'Нет'}",
        f"Изменение уверенности: {metrics['confidence_delta']:+.2%}",
        f"Класс сменился: {'Да' if metrics['class_changed'] else 'Нет'}",
    ]
    if metrics["class_changed"]:
        lines.append(f"Было: {metrics['label_original']} → Стало: {metrics['label_distorted']}")
    return img, distorted, "\n".join(lines)


def run_auto_ui(
    image,
    model_name: str,
    distortion_name: str,
    step: float,
):
    """Обработчик Gradio для автоисследования и сохранения результатов в CSV."""
    img = _image_to_numpy(image)
    if img is None:
        return [], "Загрузите изображение."
    if step <= 0:
        return [], "Шаг должен быть больше 0."

    model = _get_model(model_name)
    distortion = DISTORTIONS.get(distortion_name)
    if distortion is None:
        return [], "Выберите искажение."

    try:
        rows = run_auto_study(img, model, distortion, step)
    except ValueError as exc:
        return [], f"Ошибка автоисследования: {exc}"

    if not rows:
        return [], "Нет результатов (проверьте шаг и параметры)."

    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    csv_path = save_results_to_csv(rows, os.path.abspath(output_dir))

    preview_rows = rows[:50]
    columns = list(preview_rows[0].keys())
    data = [[row.get(col) for col in columns] for row in preview_rows]

    info_lines = [
        f"Комбинаций обработано: {len(rows)}.",
        f"Файл CSV сохранён по пути: `{csv_path}`.",
        "Откройте его в Excel для построения графиков и дальнейшего анализа.",
    ]
    # Gradio 4.x: return value directly (no .update()); first row = headers
    table_value = [columns] + data
    return table_value, "\n".join(info_lines)


def build_ui():
    """Собрать интерфейс Gradio с вкладками интерактивного и авто анализа."""
    with gr.Blocks(title="Анализ устойчивости моделей CV к искажениям") as demo:
        gr.Markdown("# Анализ устойчивости моделей компьютерного зрения к искажениям")
        with gr.Tabs():
            with gr.Tab("Интерактивный анализ"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_in = gr.Image(label="Изображение", type="numpy")
                        model_name = gr.Dropdown(
                            choices=list(MODELS.keys()),
                            value="ResNet-50 ImageNet",
                            label="Модель",
                        )
                        distortion_name = gr.Dropdown(
                            choices=list(DISTORTIONS.keys()),
                            value="Гауссов шум",
                            label="Тип искажения",
                        )
                        gr.Markdown("Параметры искажения (0–1, интерпретируются по типу):")
                        param1 = gr.Slider(0, 1, value=0.2, label="Параметр 1")
                        param2 = gr.Slider(0, 1, value=0.5, label="Параметр 2")
                        param3 = gr.Slider(0, 1, value=0.5, label="Параметр 3")
                        run_btn = gr.Button("Анализировать")
                    with gr.Column(scale=1):
                        img_orig_out = gr.Image(label="Исходное", type="numpy")
                        img_dist_out = gr.Image(label="Искажённое", type="numpy")
                        text_out = gr.Markdown(label="Результат")
                run_btn.click(
                    fn=analyze,
                    inputs=[
                        image_in,
                        model_name,
                        distortion_name,
                        param1,
                        param2,
                        param3,
                    ],
                    outputs=[img_orig_out, img_dist_out, text_out],
                )

            with gr.Tab("Автоисследование"):
                gr.Markdown(
                    "Этот режим перебирает все комбинации параметров выбранного искажения "
                    "с заданным шагом и сохраняет результаты в CSV-файл для Excel."
                )
                image_auto = gr.Image(label="Изображение", type="numpy")
                model_auto = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="ResNet-50 ImageNet",
                    label="Модель",
                )
                distortion_auto = gr.Dropdown(
                    choices=list(DISTORTIONS.keys()),
                    value="Гауссов шум",
                    label="Тип искажения",
                )
                step = gr.Slider(
                    minimum=0.05,
                    maximum=0.5,
                    value=0.1,
                    step=0.05,
                    label="Шаг по параметрам (в условных единицах диапазона)",
                )
                run_auto_btn = gr.Button("Запустить автоисследование")
                table_out = gr.Dataframe(label="Пример результатов (первые строки)")
                info_out = gr.Markdown(label="Информация и путь к CSV")

                run_auto_btn.click(
                    fn=run_auto_ui,
                    inputs=[image_auto, model_auto, distortion_auto, step],
                    outputs=[table_out, info_out],
                )
    return demo


def main():
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
