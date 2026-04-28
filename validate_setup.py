from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from monai.losses import DiceCELoss

from brats_voco.data import build_pretrain_loaders, build_segmentation_loaders, prepare_data_split
from brats_voco.models import VocoStylePretrainer, build_swinunetr, load_pretrained_encoder
from brats_voco.utils import get_device, read_json, save_checkpoint, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick end-to-end setup validation for BraTS VOCO pipeline.")
    parser.add_argument("--config", default="experiments/brats2020_voco_swinunetr.json")
    parser.add_argument("--output", default="results/setup_validation.json")
    parser.add_argument(
        "--cache-rate",
        type=float,
        default=0.0,
        help="Override cache_rate for validation to avoid out-of-memory in one-process checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    config["cache_rate"] = float(args.cache_rate)
    split = prepare_data_split(
        data_dir=config["data_dir"],
        output_path=config["split_path"],
        seed=config["seed"],
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
    )

    device = get_device()

    pretrain_train_loader, _ = build_pretrain_loaders(config, split)
    pretrain_batch = next(iter(pretrain_train_loader))
    pretrain_batch_shapes = {
        "global_view_1": list(pretrain_batch["global_view_1"].shape),
        "global_view_2": list(pretrain_batch["global_view_2"].shape),
        "local_view": list(pretrain_batch["local_view"].shape),
        "grid_view": list(pretrain_batch["grid_view"].shape),
        "drop_view": list(pretrain_batch["drop_view"].shape),
    }
    pretrainer = VocoStylePretrainer(
        feature_size=config["model"]["feature_size"],
        projection_dim=config["pretrain"]["projection_dim"],
        temperature=config["pretrain"]["temperature"],
        local_weight=config["pretrain"]["local_weight"],
        grid_weight=config["pretrain"].get("grid_weight", 0.5),
        drop_weight=config["pretrain"].get("drop_weight", 0.5),
        use_checkpoint=config["model"].get("use_checkpoint", False),
        use_v2=config["model"].get("use_v2", True),
    ).to(device)

    pretrain_outputs = pretrainer(
        pretrain_batch["global_view_1"].to(device),
        pretrain_batch["global_view_2"].to(device),
        pretrain_batch["local_view"].to(device),
        pretrain_batch["grid_view"].to(device),
        pretrain_batch["drop_view"].to(device),
    )

    del pretrain_train_loader
    del pretrain_batch
    gc.collect()

    seg_train_loader, _, _ = build_segmentation_loaders(config, split)
    seg_batch = next(iter(seg_train_loader))
    seg_model = build_swinunetr(config).to(device)
    seg_logits = seg_model(seg_batch["image"].to(device))
    seg_loss = DiceCELoss(to_onehot_y=True, softmax=True)(seg_logits, seg_batch["label"].long().to(device))

    tmp_checkpoint = Path("results") / "tmp_pretrain_for_validation.pt"
    save_checkpoint({"model": pretrainer.state_dict()}, tmp_checkpoint)
    loaded_keys, skipped_keys = load_pretrained_encoder(seg_model, str(tmp_checkpoint))

    report = {
        "device": str(device),
        "cache_rate_used": config["cache_rate"],
        "split_sizes": {"train": len(split["train"]), "val": len(split["val"]), "test": len(split["test"])},
        "pretrain_batch_shapes": pretrain_batch_shapes,
        "pretrain_forward": {
            "loss": float(pretrain_outputs["loss"].detach().cpu()),
            "global_loss": float(pretrain_outputs["global_loss"].detach().cpu()),
            "local_loss": float(pretrain_outputs["local_loss"].detach().cpu()),
            "grid_loss": float(pretrain_outputs["grid_loss"].detach().cpu()),
            "drop_loss": float(pretrain_outputs["drop_loss"].detach().cpu()),
        },
        "segmentation_batch_shape": list(seg_batch["image"].shape),
        "segmentation_logits_shape": list(seg_logits.shape),
        "segmentation_loss": float(seg_loss.detach().cpu()),
        "encoder_transfer": {"loaded_keys": len(loaded_keys), "skipped_keys": len(skipped_keys)},
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, output_path)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
