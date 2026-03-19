"""Метрики устойчивости: сравнение предсказаний на исходном и искажённом изображении."""

from typing import Any


def top1_match(pred_original: dict[str, Any], pred_distorted: dict[str, Any]) -> bool:
    """Совпадает ли предсказанный класс (топ-1) для исходного и искажённого изображения."""
    c1 = pred_original.get("top_classes", [None])[0]
    c2 = pred_distorted.get("top_classes", [None])[0]
    return c1 == c2


def confidence_change(
    pred_original: dict[str, Any],
    pred_distorted: dict[str, Any],
) -> float:
    """Изменение уверенности топ-1: confidence_distorted - confidence_original."""
    conf_orig = pred_original.get("top_confidences", [0.0])[0]
    conf_dist = pred_distorted.get("top_confidences", [0.0])[0]
    return float(conf_dist - conf_orig)


def top1_class_switch(
    pred_original: dict[str, Any],
    pred_distorted: dict[str, Any],
) -> tuple[bool, str, str]:
    """
    Проверка смены топ-1 класса и метки.

    Returns:
        (class_changed, label_original, label_distorted)
    """
    labels_orig = pred_original.get("top_labels", ["?"])
    labels_dist = pred_distorted.get("top_labels", ["?"])
    classes_orig = pred_original.get("top_classes", [])
    classes_dist = pred_distorted.get("top_classes", [])
    c_orig = classes_orig[0] if classes_orig else None
    c_dist = classes_dist[0] if classes_dist else None
    changed = c_orig != c_dist
    return (changed, labels_orig[0], labels_dist[0])


def compute_robustness_metrics(
    pred_original: dict[str, Any],
    pred_distorted: dict[str, Any],
) -> dict[str, Any]:
    """
    Сводка метрик устойчивости для отображения в UI.

    Returns:
        - top1_match: bool
        - confidence_delta: float
        - class_changed: bool
        - label_original: str
        - label_distorted: str
    """
    match = top1_match(pred_original, pred_distorted)
    delta = confidence_change(pred_original, pred_distorted)
    changed, label_orig, label_dist = top1_class_switch(pred_original, pred_distorted)
    return {
        "top1_match": match,
        "confidence_delta": delta,
        "class_changed": changed,
        "label_original": label_orig,
        "label_distorted": label_dist,
    }
