from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from .data import build_pretrain_loaders, prepare_data_split
from .models import VocoStylePretrainer
from .utils import (
    count_parameters,
    get_device,
    plot_curves,
    plot_embedding_projection,
    read_json,
    save_checkpoint,
    set_seed,
    write_csv,
    write_json,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VOCO-style contrastive pretraining for BraTS2020 H5 volumes.")
    parser.add_argument("--config", required=True, help="Path to the experiment JSON config")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional limit for train steps per epoch (debug)")
    parser.add_argument("--max-val-steps", type=int, default=None, help="Optional limit for val steps per epoch (debug)")
    parser.add_argument(
        "--embedding-viz-every",
        type=int,
        default=10,
        help="Run embedding diagnostics every N epochs and at the final epoch",
    )
    parser.add_argument(
        "--embedding-viz-batches",
        type=int,
        default=6,
        help="Number of validation batches used for embedding diagnostics",
    )
    parser.add_argument("--resume-from", default=None, help="Path to a checkpoint to resume pretraining")
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable automatic resume from output-dir/checkpoint_last.pt when present",
    )
    return parser.parse_args()


def run_epoch(
    model: VocoStylePretrainer,
    loader,
    optimizer,
    device: torch.device,
    training: bool,
    max_steps: int | None,
    amp_enabled: bool,
    scaler: GradScaler,
) -> dict[str, float]:
    model.train(training)
    stats = {"loss": 0.0, "global_loss": 0.0, "local_loss": 0.0, "grid_loss": 0.0, "drop_loss": 0.0, "steps": 0}
    for step, batch in enumerate(loader, start=1):
        global_view_1 = batch["global_view_1"].to(device)
        global_view_2 = batch["global_view_2"].to(device)
        local_view = batch["local_view"].to(device)
        grid_view = batch.get("grid_view")
        drop_view = batch.get("drop_view")
        if grid_view is not None:
            grid_view = grid_view.to(device)
        if drop_view is not None:
            drop_view = drop_view.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast(enabled=amp_enabled):
                outputs = model(global_view_1, global_view_2, local_view, grid_view=grid_view, drop_view=drop_view)
                loss = outputs["loss"]
            if training:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        stats["loss"] += float(loss.detach().cpu())
        stats["global_loss"] += float(outputs["global_loss"].cpu())
        stats["local_loss"] += float(outputs["local_loss"].cpu())
        stats["grid_loss"] += float(outputs["grid_loss"].cpu())
        stats["drop_loss"] += float(outputs["drop_loss"].cpu())
        stats["steps"] += 1

        if max_steps is not None and step >= max_steps:
            break

    return {
        "loss": stats["loss"] / max(1, stats["steps"]),
        "global_loss": stats["global_loss"] / max(1, stats["steps"]),
        "local_loss": stats["local_loss"] / max(1, stats["steps"]),
        "grid_loss": stats["grid_loss"] / max(1, stats["steps"]),
        "drop_loss": stats["drop_loss"] / max(1, stats["steps"]),
    }


@torch.no_grad()
def embedding_diagnostics(
    model: VocoStylePretrainer,
    loader,
    device: torch.device,
    max_batches: int,
) -> tuple[dict[str, float], np.ndarray, list[str]]:
    model.eval()
    pos_global_list: list[float] = []
    pos_local_list: list[float] = []
    neg_list: list[float] = []
    norm_list: list[float] = []

    embeddings: list[np.ndarray] = []
    labels: list[str] = []

    for batch_idx, batch in enumerate(loader, start=1):
        global_view_1 = batch["global_view_1"].to(device)
        global_view_2 = batch["global_view_2"].to(device)
        local_view = batch["local_view"].to(device)
        grid_view = batch.get("grid_view")
        drop_view = batch.get("drop_view")
        if grid_view is not None:
            grid_view = grid_view.to(device)
        if drop_view is not None:
            drop_view = drop_view.to(device)

        z1 = model.encode(global_view_1)
        z2 = model.encode(global_view_2)
        zl = model.encode(local_view)

        pos_global = torch.sum(z1 * z2, dim=1)
        pos_local = torch.sum(zl * z1, dim=1)
        neg = torch.sum(z1 * z2[torch.randperm(z2.shape[0])], dim=1)
        norms = torch.norm(z1, dim=1)

        pos_global_list.extend(pos_global.detach().cpu().tolist())
        pos_local_list.extend(pos_local.detach().cpu().tolist())
        neg_list.extend(neg.detach().cpu().tolist())
        norm_list.extend(norms.detach().cpu().tolist())

        embeddings.extend(z1.detach().cpu().numpy())
        labels.extend(["global_view_1"] * z1.shape[0])
        embeddings.extend(z2.detach().cpu().numpy())
        labels.extend(["global_view_2"] * z2.shape[0])
        embeddings.extend(zl.detach().cpu().numpy())
        labels.extend(["local_view"] * zl.shape[0])
        if grid_view is not None:
            zgd = model.encode(grid_view)
            embeddings.extend(zgd.detach().cpu().numpy())
            labels.extend(["grid_view"] * zgd.shape[0])
        if drop_view is not None:
            zdp = model.encode(drop_view)
            embeddings.extend(zdp.detach().cpu().numpy())
            labels.extend(["drop_view"] * zdp.shape[0])

        if batch_idx >= max_batches:
            break

    stats = {
        "emb_pos_global_mean": float(np.mean(pos_global_list)) if pos_global_list else float("nan"),
        "emb_pos_local_mean": float(np.mean(pos_local_list)) if pos_local_list else float("nan"),
        "emb_neg_mean": float(np.mean(neg_list)) if neg_list else float("nan"),
        "emb_norm_mean": float(np.mean(norm_list)) if norm_list else float("nan"),
        "emb_norm_std": float(np.std(norm_list)) if norm_list else float("nan"),
    }
    array_embeddings = np.asarray(embeddings, dtype=np.float32) if embeddings else np.empty((0, 0), dtype=np.float32)
    return stats, array_embeddings, labels


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

    output_dir = Path(args.output_dir or Path(config["output_root"]) / "pretrain_voco")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config, output_dir / "config.json")
    if Path(config["split_path"]).exists():
        shutil.copyfile(config["split_path"], output_dir / "split.json")

    device = get_device()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader, val_loader = build_pretrain_loaders(config, split)
    model = VocoStylePretrainer(
        feature_size=config["model"]["feature_size"],
        projection_dim=config["pretrain"]["projection_dim"],
        temperature=config["pretrain"]["temperature"],
        local_weight=config["pretrain"]["local_weight"],
        grid_weight=config["pretrain"].get("grid_weight", 0.5),
        drop_weight=config["pretrain"].get("drop_weight", 0.5),
        use_checkpoint=config["model"].get("use_checkpoint", False),
        use_v2=config["model"].get("use_v2", True),
    ).to(device)
    write_text(str(model), output_dir / "model_architecture.txt")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["pretrain"]["lr"],
        weight_decay=config["pretrain"]["weight_decay"],
    )
    amp_enabled = bool(config["pretrain"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["pretrain"]["epochs"])
    epoch_checkpoints_dir = output_dir / "checkpoints"
    epoch_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_every = int(config["pretrain"].get("checkpoint_every", 1))
    save_epoch_checkpoints = bool(config["pretrain"].get("save_epoch_checkpoints", True))

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    start_epoch = 1

    resume_path = args.resume_from
    if resume_path is None and not args.no_auto_resume:
        candidate = output_dir / "checkpoint_last.pt"
        if candidate.exists():
            resume_path = str(candidate)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        history = list(checkpoint.get("history", history))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[pretrain] resumed from {resume_path} at epoch={start_epoch}")

    print(f"device={device} train_cases={len(split['train'])} val_cases={len(split['val'])} params={count_parameters(model):,}")
    for epoch in range(start_epoch, config["pretrain"]["epochs"] + 1):
        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True,
            max_steps=args.max_train_steps,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            training=False,
            max_steps=args.max_val_steps,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )

        need_diagnostics = epoch == config["pretrain"]["epochs"] or (args.embedding_viz_every > 0 and epoch % args.embedding_viz_every == 0)
        emb_stats = {
            "emb_pos_global_mean": float("nan"),
            "emb_pos_local_mean": float("nan"),
            "emb_neg_mean": float("nan"),
            "emb_norm_mean": float("nan"),
            "emb_norm_std": float("nan"),
        }
        if need_diagnostics:
            emb_stats, emb_array, emb_labels = embedding_diagnostics(
                model=model,
                loader=val_loader,
                device=device,
                max_batches=args.embedding_viz_batches,
            )
            plot_embedding_projection(
                emb_array,
                emb_labels,
                output_dir / f"embedding_pca_epoch_{epoch:03d}.png",
                title=f"Embedding PCA epoch {epoch}",
            )

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_global_loss": train_stats["global_loss"],
            "train_local_loss": train_stats["local_loss"],
            "train_grid_loss": train_stats["grid_loss"],
            "train_drop_loss": train_stats["drop_loss"],
            "val_loss": val_stats["loss"],
            "val_global_loss": val_stats["global_loss"],
            "val_local_loss": val_stats["local_loss"],
            "val_grid_loss": val_stats["grid_loss"],
            "val_drop_loss": val_stats["drop_loss"],
            "val_emb_pos_global_mean": emb_stats["emb_pos_global_mean"],
            "val_emb_pos_local_mean": emb_stats["emb_pos_local_mean"],
            "val_emb_neg_mean": emb_stats["emb_neg_mean"],
            "val_emb_norm_mean": emb_stats["emb_norm_mean"],
            "val_emb_norm_std": emb_stats["emb_norm_std"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[pretrain] epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"g={row['val_global_loss']:.4f} l={row['val_local_loss']:.4f} gd={row['val_grid_loss']:.4f} dp={row['val_drop_loss']:.4f} "
            f"emb_pos={row['val_emb_pos_global_mean']:.4f} emb_neg={row['val_emb_neg_mean']:.4f} "
            f"lr={row['lr']:.6f}"
        )

        write_csv(history, output_dir / "history.csv")

        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "config": config,
            "best_val_loss": best_val_loss,
            "history": history,
            "amp_enabled": amp_enabled,
        }
        save_checkpoint(
            checkpoint_state,
            output_dir / "checkpoint_last.pt",
        )
        if save_epoch_checkpoints and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(checkpoint_state, epoch_checkpoints_dir / f"checkpoint_epoch_{epoch:04d}.pt")
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            checkpoint_state["best_val_loss"] = best_val_loss
            save_checkpoint(
                checkpoint_state,
                output_dir / "checkpoint_best.pt",
            )

    write_csv(history, output_dir / "history.csv")
    plot_curves(history, output_dir / "curves.png", metrics=["loss", "global_loss", "local_loss", "grid_loss", "drop_loss"])
    write_json(
        {
            "best_val_loss": best_val_loss,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "train_cases": len(split["train"]),
            "val_cases": len(split["val"]),
            "embedding_projection_files": sorted(path.name for path in output_dir.glob("embedding_pca_epoch_*.png")),
        },
        output_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
