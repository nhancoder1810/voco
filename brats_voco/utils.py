from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.utils import set_determinism
from sklearn.decomposition import PCA


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(history: list[dict[str, float]], path: str | Path, metrics: list[str]) -> None:
    if not history:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    epochs = [row["epoch"] for row in history]
    for axis, metric in zip(axes, metrics):
        axis.plot(epochs, [row.get(f"train_{metric}", np.nan) for row in history], label=f"train_{metric}")
        axis.plot(epochs, [row.get(f"val_{metric}", np.nan) for row in history], label=f"val_{metric}")
        axis.set_title(metric)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_text(text: str, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: list[str],
    path: str | Path,
    title: str,
) -> None:
    if embeddings.size == 0:
        return

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(embeddings)
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")

    figure, axis = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(unique_labels):
        mask = np.array([item == label for item in labels], dtype=bool)
        axis.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.8, label=label, color=cmap(index % 10))
    axis.set_title(title)
    axis.set_xlabel("PCA-1")
    axis.set_ylabel("PCA-2")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_comparison_bars(
    baseline_dice: float,
    pretrained_dice: float,
    baseline_hd95: float,
    pretrained_hd95: float,
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].bar(["baseline", "pretrained"], [baseline_dice, pretrained_dice], color=["#5B8FF9", "#5AD8A6"])
    axes[0].set_title("Test Dice")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25, axis="y")

    axes[1].bar(["baseline", "pretrained"], [baseline_hd95, pretrained_hd95], color=["#5B8FF9", "#5AD8A6"])
    axes[1].set_title("Test HD95")
    axes[1].grid(alpha=0.25, axis="y")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
