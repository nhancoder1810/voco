from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import ContrastiveLoss
from monai.networks.nets import SwinUNETR


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(x), dim=1)


class VocoStylePretrainer(nn.Module):
    def __init__(
        self,
        feature_size: int = 24,
        projection_dim: int = 256,
        temperature: float = 0.1,
        local_weight: float = 0.5,
        grid_weight: float = 0.5,
        drop_weight: float = 0.5,
        use_checkpoint: bool = False,
        use_v2: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = SwinUNETR(
            in_channels=1,
            out_channels=2,
            feature_size=feature_size,
            spatial_dims=3,
            use_checkpoint=use_checkpoint,
            use_v2=use_v2,
        )
        self.projector = ProjectionHead(in_dim=feature_size * 16, out_dim=projection_dim)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.local_weight = local_weight
        self.grid_weight = grid_weight
        self.drop_weight = drop_weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.swinViT(x, normalize=True)[-1]
        pooled = F.adaptive_avg_pool3d(features, output_size=1).flatten(1)
        return self.projector(pooled)

    def forward(
        self,
        global_view_1: torch.Tensor,
        global_view_2: torch.Tensor,
        local_view: torch.Tensor,
        grid_view: torch.Tensor | None = None,
        drop_view: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if global_view_1.shape[0] < 2:
            raise ValueError("VOCO-style contrastive pretraining needs batch_size >= 2 for useful negatives.")
        z1 = self.encode(global_view_1)
        z2 = self.encode(global_view_2)
        zl = self.encode(local_view)
        zg = F.normalize((z1 + z2) / 2.0, dim=1)
        global_loss = self.contrastive_loss(z1, z2)
        local_loss = self.contrastive_loss(zl, zg)

        if grid_view is not None:
            zgd = self.encode(grid_view)
            grid_loss = self.contrastive_loss(zgd, zg)
        else:
            grid_loss = torch.zeros_like(global_loss)

        if drop_view is not None:
            zdp = self.encode(drop_view)
            drop_loss = self.contrastive_loss(zdp, zg)
        else:
            drop_loss = torch.zeros_like(global_loss)

        total_loss = global_loss + self.local_weight * local_loss + self.grid_weight * grid_loss + self.drop_weight * drop_loss
        return {
            "loss": total_loss,
            "global_loss": global_loss.detach(),
            "local_loss": local_loss.detach(),
            "grid_loss": grid_loss.detach(),
            "drop_loss": drop_loss.detach(),
        }


def build_swinunetr(config: dict[str, Any]) -> SwinUNETR:
    model_config = config["model"]
    return SwinUNETR(
        in_channels=1,
        out_channels=2,
        feature_size=model_config["feature_size"],
        spatial_dims=3,
        use_checkpoint=model_config.get("use_checkpoint", False),
        use_v2=model_config.get("use_v2", True),
    )


def load_pretrained_encoder(model: SwinUNETR, checkpoint_path: str) -> tuple[list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    target_state = model.state_dict()

    updated: dict[str, torch.Tensor] = {}
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    for key, value in state_dict.items():
        new_key = None
        if key.startswith("backbone.swinViT."):
            new_key = key[len("backbone.") :]
        elif key.startswith("swinViT."):
            new_key = key
        if new_key is None:
            continue
        if new_key in target_state and target_state[new_key].shape == value.shape:
            updated[new_key] = value
            loaded_keys.append(new_key)
        else:
            skipped_keys.append(key)

    target_state.update(updated)
    model.load_state_dict(target_state)
    return loaded_keys, skipped_keys
