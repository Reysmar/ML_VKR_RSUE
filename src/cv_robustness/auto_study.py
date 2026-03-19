"""Автоматическое исследование устойчивости: перебор параметров искажений и сохранение в CSV."""

from __future__ import annotations

import csv
import itertools
import os
from datetime import datetime
from typing import Any

import numpy as np

from .distortions.base import BaseDistortion
from .metrics import compute_robustness_metrics
from .models.base import BaseModelWrapper


MAX_COMBINATIONS = 10_000


def _build_param_grid(distortion: BaseDistortion, step: float) -> list[dict[str, Any]]:
    """Сгенерировать полную решётку параметров по params_spec и шагу."""
    if step <= 0:
        raise ValueError("step must be > 0")

    specs = distortion.params_spec
    if not specs:
        return [{}]

    all_values: list[list[tuple[str, Any]]] = []
    for name, typ, default, lo, hi, _label in specs:
        lo_val = float(lo)
        hi_val = float(hi)
        if hi_val < lo_val:
            lo_val, hi_val = hi_val, lo_val
        # количество шагов, включая границы
        n_steps = int((hi_val - lo_val) / step) + 1
        n_steps = max(1, min(n_steps, 1000))
        vals: list[tuple[str, Any]] = []
        for k in range(n_steps):
            raw = lo_val + k * step
            if typ == "int":
                v = int(round(raw))
            else:
                v = float(raw)
            vals.append((name, v))
        all_values.append(vals)

    combos: list[dict[str, Any]] = []
    for prod in itertools.product(*all_values):
        params: dict[str, Any] = {}
        for key, value in prod:
            params[key] = value
        combos.append(params)

    return combos


def run_auto_study(
    image: np.ndarray,
    model_wrapper: BaseModelWrapper,
    distortion: BaseDistortion,
    step: float,
) -> list[dict[str, Any]]:
    """Перебрать сетку параметров, посчитать предсказания и метрики."""
    param_grid = _build_param_grid(distortion, step)
    total = len(param_grid)
    if total > MAX_COMBINATIONS:
        param_grid = param_grid[:MAX_COMBINATIONS]

    results: list[dict[str, Any]] = []
    pred_orig = model_wrapper.predict(image)

    for params in param_grid:
        distorted = distortion.apply(image, **params)
        pred_dist = model_wrapper.predict(distorted)
        metrics = compute_robustness_metrics(pred_orig, pred_dist)

        row: dict[str, Any] = {
            "distortion_name": getattr(distortion, "name", distortion.__class__.__name__),
            "model_name": getattr(model_wrapper, "name", model_wrapper.__class__.__name__),
            "orig_label": pred_orig.get("top_labels", ["?"])[0],
            "orig_conf": pred_orig.get("top_confidences", [0.0])[0],
            "dist_label": pred_dist.get("top_labels", ["?"])[0],
            "dist_conf": pred_dist.get("top_confidences", [0.0])[0],
            "top1_match": metrics["top1_match"],
            "confidence_delta": metrics["confidence_delta"],
            "class_changed": metrics["class_changed"],
        }
        for name, value in params.items():
            row[f"param_{name}"] = value
        results.append(row)

    return results


def save_results_to_csv(rows: list[dict[str, Any]], output_dir: str) -> str:
    """Сохранить результаты автоисследования в CSV и вернуть путь к файлу."""
    if not rows:
        raise ValueError("no rows to save")

    os.makedirs(output_dir, exist_ok=True)

    distortion_name = str(rows[0].get("distortion_name", "distortion")).replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"auto_study_{distortion_name}_{timestamp}.csv"
    path = os.path.join(output_dir, filename)

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return os.path.abspath(path)

