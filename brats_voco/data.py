from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandomizableTransform,
    RandSpatialCropd,
    SpatialPadd,
)
from sklearn.model_selection import train_test_split

from .utils import read_json, write_json


def prepare_data_split(
    data_dir: str | Path,
    output_path: str | Path,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    if output_path.exists():
        return read_json(output_path)

    files = sorted(str(path.as_posix()) for path in data_dir.glob("*.h5"))
    if len(files) < 3:
        raise ValueError("Need at least 3 H5 files to create train/val/test splits.")

    train_files, holdout_files = train_test_split(files, train_size=train_ratio, random_state=seed, shuffle=True)
    val_size = val_ratio / (1.0 - train_ratio)
    val_files, test_files = train_test_split(holdout_files, train_size=val_size, random_state=seed, shuffle=True)

    split = {
        "seed": seed,
        "train": sorted(train_files),
        "val": sorted(val_files),
        "test": sorted(test_files),
    }
    write_json(split, output_path)
    return split


class LoadBraTSH5d(MapTransform):
    def __init__(self, source_key: str = "h5_path", image_key: str = "image", label_key: str = "label") -> None:
        super().__init__([source_key])
        self.source_key = source_key
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        path = Path(output[self.source_key])
        with h5py.File(path, "r") as handle:
            output[self.image_key] = handle[self.image_key][:].astype(np.float32)
            if self.label_key in handle:
                output[self.label_key] = handle[self.label_key][:].astype(np.int64)
        output["case_id"] = path.stem
        return output


class CreateVocoViewsd(RandomizableTransform, MapTransform):
    def __init__(
        self,
        image_key: str = "image",
        global_size: Sequence[int] = (96, 96, 96),
        local_size: Sequence[int] = (64, 64, 64),
        grid_size: int = 4,
        patch_modes: Sequence[str] = ("local", "grid", "drop"),
        train: bool = True,
    ) -> None:
        MapTransform.__init__(self, keys=[image_key])
        RandomizableTransform.__init__(self, prob=1.0)
        self.image_key = image_key
        self.global_size = tuple(int(v) for v in global_size)
        self.local_size = tuple(int(v) for v in local_size)
        self.grid_size = int(grid_size)
        self.patch_modes = tuple(patch_modes)
        self.train = train

    def _pad_to_size(self, image: np.ndarray, size: Sequence[int]) -> np.ndarray:
        _, depth, height, width = image.shape
        pad_depth = max(0, size[0] - depth)
        pad_height = max(0, size[1] - height)
        pad_width = max(0, size[2] - width)
        if pad_depth == pad_height == pad_width == 0:
            return image
        return np.pad(
            image,
            (
                (0, 0),
                (pad_depth // 2, pad_depth - pad_depth // 2),
                (pad_height // 2, pad_height - pad_height // 2),
                (pad_width // 2, pad_width - pad_width // 2),
            ),
            mode="constant",
        )

    def _crop(self, image: np.ndarray, size: Sequence[int], centered: bool) -> np.ndarray:
        image = self._pad_to_size(image, size)
        _, depth, height, width = image.shape
        if centered:
            starts = [
                max(0, (depth - size[0]) // 2),
                max(0, (height - size[1]) // 2),
                max(0, (width - size[2]) // 2),
            ]
        else:
            starts = [
                int(self.R.randint(0, max(1, depth - size[0] + 1))),
                int(self.R.randint(0, max(1, height - size[1] + 1))),
                int(self.R.randint(0, max(1, width - size[2] + 1))),
            ]
        return image[
            :,
            starts[0] : starts[0] + size[0],
            starts[1] : starts[1] + size[1],
            starts[2] : starts[2] + size[2],
        ]

    def _augment(self, image: np.ndarray) -> np.ndarray:
        output = image.copy()
        if not self.train:
            return output
        for axis in (1, 2, 3):
            if self.R.rand() < 0.5:
                output = np.flip(output, axis=axis).copy()
        scale = float(self.R.uniform(0.9, 1.1))
        shift = float(self.R.uniform(-0.1, 0.1))
        noise_std = float(self.R.uniform(0.0, 0.05))
        output = output * scale + shift
        output = output + self.R.normal(0.0, noise_std, size=output.shape).astype(np.float32)
        return output.astype(np.float32)

    def _resize(self, image: np.ndarray, target_size: Sequence[int]) -> np.ndarray:
        tensor = torch.as_tensor(image[None, ...], dtype=torch.float32)
        resized = F.interpolate(tensor, size=tuple(target_size), mode="trilinear", align_corners=False)
        return resized.squeeze(0).numpy().astype(np.float32)

    def _grid_crop(self, image: np.ndarray) -> np.ndarray:
        image = self._pad_to_size(image, self.global_size)
        _, depth, height, width = image.shape
        step_h = max(1, height // self.grid_size)
        step_w = max(1, width // self.grid_size)

        if self.train:
            grid_i = int(self.R.randint(0, self.grid_size))
            grid_j = int(self.R.randint(0, self.grid_size))
        else:
            grid_i = self.grid_size // 2
            grid_j = self.grid_size // 2

        center_h = int((grid_i + 0.5) * step_h)
        center_w = int((grid_j + 0.5) * step_w)
        start_h = int(np.clip(center_h - self.global_size[1] // 2, 0, max(0, height - self.global_size[1])))
        start_w = int(np.clip(center_w - self.global_size[2] // 2, 0, max(0, width - self.global_size[2])))
        start_d = max(0, (depth - self.global_size[0]) // 2)

        crop = image[
            :,
            start_d : start_d + self.global_size[0],
            start_h : start_h + self.global_size[1],
            start_w : start_w + self.global_size[2],
        ]
        return self._augment(crop)

    def _random_drop(self, image: np.ndarray, max_drop_ratio: float = 0.3, max_block_ratio: float = 0.25) -> np.ndarray:
        output = image.copy()
        if not self.train:
            return output

        _, depth, height, width = output.shape
        total_voxels = depth * height * width
        target_drop = int(float(self.R.uniform(0.0, max_drop_ratio)) * total_voxels)
        max_d = max(2, int(depth * max_block_ratio))
        max_h = max(2, int(height * max_block_ratio))
        max_w = max(2, int(width * max_block_ratio))

        dropped = 0
        while dropped < target_drop:
            d0 = int(self.R.randint(0, max(1, depth - 1)))
            h0 = int(self.R.randint(0, max(1, height - 1)))
            w0 = int(self.R.randint(0, max(1, width - 1)))
            d1 = min(depth, d0 + int(self.R.randint(1, max_d)))
            h1 = min(height, h0 + int(self.R.randint(1, max_h)))
            w1 = min(width, w0 + int(self.R.randint(1, max_w)))
            noise = self.R.normal(loc=0.0, scale=1.0, size=(1, d1 - d0, h1 - h0, w1 - w0)).astype(np.float32)
            output[:, d0:d1, h0:h1, w0:w1] = noise
            dropped += (d1 - d0) * (h1 - h0) * (w1 - w0)
        return output

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        output = dict(data)
        image = output[self.image_key]
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        image = np.asarray(image, dtype=np.float32)
        centered = not self.train

        global_view_1 = self._augment(self._crop(image, self.global_size, centered=centered))
        global_view_2 = self._augment(self._crop(image, self.global_size, centered=centered))
        if "local" in self.patch_modes:
            local_view = self._augment(self._crop(image, self.local_size, centered=centered))
            local_view = self._resize(local_view, self.global_size)
        else:
            local_view = global_view_1.copy()

        if "grid" in self.patch_modes:
            grid_view = self._grid_crop(image)
        else:
            grid_view = global_view_1.copy()

        if "drop" in self.patch_modes:
            drop_view = self._random_drop(global_view_1)
        else:
            drop_view = global_view_1.copy()

        output["global_view_1"] = global_view_1
        output["global_view_2"] = global_view_2
        output["local_view"] = local_view
        output["grid_view"] = grid_view
        output["drop_view"] = drop_view
        return output


def _dataset_items(file_list: Sequence[str]) -> list[dict[str, str]]:
    return [{"h5_path": str(path)} for path in file_list]


def build_pretrain_loaders(config: dict[str, Any], split: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    roi_size = config["pretrain"]["global_crop_size"]
    local_size = config["pretrain"]["local_crop_size"]
    grid_size = config["pretrain"].get("grid_size", 4)
    patch_modes = config["pretrain"].get("patch_modes", ["local", "grid", "drop"])
    cache_rate = config.get("cache_rate", 0.0)
    num_workers = config.get("num_workers", 0)

    train_transforms = Compose(
        [
            LoadBraTSH5d(),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            CreateVocoViewsd(global_size=roi_size, local_size=local_size, grid_size=grid_size, patch_modes=patch_modes, train=True),
            DeleteItemsd(keys=["image", "label"]),
            EnsureTyped(keys=["global_view_1", "global_view_2", "local_view", "grid_view", "drop_view"], dtype=torch.float32),
        ]
    )
    val_transforms = Compose(
        [
            LoadBraTSH5d(),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            CreateVocoViewsd(global_size=roi_size, local_size=local_size, grid_size=grid_size, patch_modes=patch_modes, train=False),
            DeleteItemsd(keys=["image", "label"]),
            EnsureTyped(keys=["global_view_1", "global_view_2", "local_view", "grid_view", "drop_view"], dtype=torch.float32),
        ]
    )

    if cache_rate > 0:
        train_ds = CacheDataset(data=_dataset_items(split["train"]), transform=train_transforms, cache_rate=cache_rate)
        val_ds = CacheDataset(data=_dataset_items(split["val"]), transform=val_transforms, cache_rate=cache_rate)
    else:
        train_ds = Dataset(data=_dataset_items(split["train"]), transform=train_transforms)
        val_ds = Dataset(data=_dataset_items(split["val"]), transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["pretrain"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def build_segmentation_loaders(
    config: dict[str, Any],
    split: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    roi_size = config["segmentation"]["roi_size"]
    cache_rate = config.get("cache_rate", 0.0)
    num_workers = config.get("num_workers", 0)

    train_transforms = Compose(
        [
            LoadBraTSH5d(),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )
    eval_transforms = Compose(
        [
            LoadBraTSH5d(),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    if cache_rate > 0:
        train_ds = CacheDataset(data=_dataset_items(split["train"]), transform=train_transforms, cache_rate=cache_rate)
        val_ds = CacheDataset(data=_dataset_items(split["val"]), transform=eval_transforms, cache_rate=cache_rate)
        test_ds = CacheDataset(data=_dataset_items(split["test"]), transform=eval_transforms, cache_rate=cache_rate)
    else:
        train_ds = Dataset(data=_dataset_items(split["train"]), transform=train_transforms)
        val_ds = Dataset(data=_dataset_items(split["val"]), transform=eval_transforms)
        test_ds = Dataset(data=_dataset_items(split["test"]), transform=eval_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["segmentation"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader


def infer_input_size(config: dict[str, Any]) -> tuple[int, int, int]:
    size = config["segmentation"]["roi_size"]
    if len(size) != 3:
        raise ValueError("roi_size must have exactly 3 dimensions.")
    return tuple(int(v) for v in size)
