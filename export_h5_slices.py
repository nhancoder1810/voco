from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all slices from an H5 volume into a single montage image."
    )
    parser.add_argument("input_h5", type=Path, help="Path to the input H5 file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to outputs/<stem>_axial_montage.png",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=8,
        help="Number of columns in the montage grid",
    )
    return parser.parse_args()


def load_volume(input_h5: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(input_h5, "r") as handle:
        image = handle["image"][:]
        label = handle["label"][:]
    return image, label


def normalize_image(volume: np.ndarray) -> np.ndarray:
    lower = np.percentile(volume, 1)
    upper = np.percentile(volume, 99)
    if upper <= lower:
        return np.zeros_like(volume, dtype=np.float32)
    clipped = np.clip(volume, lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def save_montage(image: np.ndarray, label: np.ndarray, output_path: Path, cols: int) -> None:
    num_slices = image.shape[-1]
    rows = int(np.ceil(num_slices / cols))

    normalized = normalize_image(image)
    figure, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1))
    axes = np.atleast_1d(axes).ravel()

    for index in range(num_slices):
        axis = axes[index]
        axis.imshow(normalized[:, :, index], cmap="gray")

        mask = label[:, :, index]
        if np.any(mask):
            overlay = np.ma.masked_where(mask == 0, mask)
            axis.imshow(overlay, cmap="autumn", alpha=0.55, interpolation="none")

        axis.set_title(f"z={index}", fontsize=7)
        axis.axis("off")

    for index in range(num_slices, len(axes)):
        axes[index].axis("off")

    figure.suptitle(
        f"{output_path.stem} | slices={num_slices} | image+label overlay",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_path = args.output or Path("outputs") / f"{args.input_h5.stem}_axial_montage.png"
    image, label = load_volume(args.input_h5)
    save_montage(image, label, output_path, args.cols)
    print(output_path.resolve())


if __name__ == "__main__":
    main()