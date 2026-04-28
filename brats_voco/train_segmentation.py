from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose

from .data import build_segmentation_loaders, infer_input_size, prepare_data_split
from .models import build_swinunetr, load_pretrained_encoder
from .utils import count_parameters, get_device, plot_curves, read_json, save_checkpoint, set_seed, write_csv, write_json, write_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BraTS2020 segmentation with or without VOCO-style pretraining.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config")
    parser.add_argument("--mode", choices=["baseline", "pretrained"], required=True)
    parser.add_argument("--pretrained-checkpoint", default=None, help="Path to pretraining checkpoint for encoder initialization")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--resume-from", default=None, help="Path to a segmentation checkpoint to resume")
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable automatic resume from output-dir/checkpoint_last.pt when present",
    )
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, loss_fn, device: torch.device, amp_enabled: bool, scaler: GradScaler) -> float:
    model.train()
    running_loss = 0.0
    steps = 0
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].long().to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits = model(image)
            loss = loss_fn(logits, label)
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += float(loss.detach().cpu())
        steps += 1
    return running_loss / max(1, steps)


@torch.no_grad()
def evaluate(model, loader, device: torch.device, roi_size: tuple[int, int, int], sw_batch_size: int, overlap: float) -> tuple[float, float, list[dict[str, float]]]:
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    case_rows: list[dict[str, float]] = []
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].long().to(device)
        case_id = batch["case_id"][0]
        logits = sliding_window_inference(
            inputs=image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
        )
        pred_list = [post_pred(item) for item in decollate_batch(logits)]
        label_list = [post_label(item) for item in decollate_batch(label)]
        dice_metric(y_pred=pred_list, y=label_list)
        hd95_metric(y_pred=pred_list, y=label_list)
        case_dice = float(dice_metric.aggregate().cpu())
        case_hd95 = float(hd95_metric.aggregate().cpu())
        case_rows.append({"case_id": case_id, "dice": case_dice, "hd95": case_hd95})
        dice_metric.reset()
        hd95_metric.reset()

    mean_dice = float(sum(row["dice"] for row in case_rows) / max(1, len(case_rows)))
    mean_hd95 = float(sum(row["hd95"] for row in case_rows) / max(1, len(case_rows)))
    return mean_dice, mean_hd95, case_rows


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    set_seed(config["seed"])
    split = prepare_data_split(
        data_dir=config["data_dir"],
        output_path=config["split_path"],
        seed=config["seed"],
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
    )
    output_dir = Path(args.output_dir or Path(config["output_root"]) / f"segmentation_{args.mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config, output_dir / "config.json")
    if Path(config["split_path"]).exists():
        shutil.copyfile(config["split_path"], output_dir / "split.json")

    device = get_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader, val_loader, test_loader = build_segmentation_loaders(config, split)
    model = build_swinunetr(config).to(device)
    write_text(str(model), output_dir / "model_architecture.txt")
    resume_path = args.resume_from
    if resume_path is None and not args.no_auto_resume:
        candidate = output_dir / "checkpoint_last.pt"
        if candidate.exists():
            resume_path = str(candidate)

    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    if args.mode == "pretrained" and resume_path is None:
        if not args.pretrained_checkpoint:
            raise ValueError("--pretrained-checkpoint is required when mode=pretrained")
        loaded_keys, skipped_keys = load_pretrained_encoder(model, args.pretrained_checkpoint)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["segmentation"]["lr"],
        weight_decay=config["segmentation"]["weight_decay"],
    )
    amp_enabled = bool(config["segmentation"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["segmentation"]["epochs"])
    epoch_checkpoints_dir = output_dir / "checkpoints"
    epoch_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_every = int(config["segmentation"].get("checkpoint_every", 1))
    save_epoch_checkpoints = bool(config["segmentation"].get("save_epoch_checkpoints", True))
    roi_size = infer_input_size(config)

    history: list[dict[str, float]] = []
    best_val_dice = -1.0
    start_epoch = 1

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        best_val_dice = float(checkpoint.get("best_val_dice", best_val_dice))
        history = list(checkpoint.get("history", history))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[seg-{args.mode}] resumed from {resume_path} at epoch={start_epoch}")

    print(
        f"device={device} mode={args.mode} train_cases={len(split['train'])} val_cases={len(split['val'])} "
        f"test_cases={len(split['test'])} params={count_parameters(model):,} loaded_encoder_keys={len(loaded_keys)}"
    )

    for epoch in range(start_epoch, config["segmentation"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, amp_enabled=amp_enabled, scaler=scaler)
        val_dice, val_hd95, _ = evaluate(
            model,
            val_loader,
            device,
            roi_size=roi_size,
            sw_batch_size=config["segmentation"]["sw_batch_size"],
            overlap=config["segmentation"]["overlap"],
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_dice": val_dice,
            "val_hd95": val_hd95,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[seg-{args.mode}] epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"val_dice={val_dice:.4f} val_hd95={val_hd95:.4f}"
        )

        write_csv(history, output_dir / "history.csv")

        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "config": config,
            "best_val_dice": best_val_dice,
            "history": history,
            "amp_enabled": amp_enabled,
            "mode": args.mode,
        }

        save_checkpoint(
            checkpoint_state,
            output_dir / "checkpoint_last.pt",
        )
        if save_epoch_checkpoints and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(checkpoint_state, epoch_checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint_state["best_val_dice"] = best_val_dice
            save_checkpoint(
                checkpoint_state,
                output_dir / "checkpoint_best.pt",
            )

    best_checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_dice, test_hd95, case_rows = evaluate(
        model,
        test_loader,
        device,
        roi_size=roi_size,
        sw_batch_size=config["segmentation"]["sw_batch_size"],
        overlap=config["segmentation"]["overlap"],
    )
    write_csv(history, output_dir / "history.csv")
    write_csv(case_rows, output_dir / "test_case_metrics.csv")
    plot_curves(history, output_dir / "curves.png", metrics=["loss", "dice", "hd95"])
    write_json(
        {
            "mode": args.mode,
            "best_val_dice": best_val_dice,
            "test_dice": test_dice,
            "test_hd95": test_hd95,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "loaded_encoder_keys": len(loaded_keys),
            "skipped_encoder_keys": len(skipped_keys),
            "loaded_encoder_key_samples": loaded_keys[:20],
            "skipped_encoder_key_samples": skipped_keys[:20],
        },
        output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
