# BraTS2020 VOCO-Style Pretraining Study

## Research Question

Can VOCO-style self-supervised contrastive pretraining on single-channel BraTS2020 volumes improve downstream binary brain tumor segmentation compared with training SwinUNETR from random initialization?

## Hypothesis

VOCO-style pretraining that aligns global and local 3D crops from the same BraTS volume will improve representation quality and yield higher test Dice and lower HD95 than a no-pretraining baseline.

## Dataset Setting

- Dataset: 369 BraTS2020 `.h5` volumes in the workspace.
- Input: single-channel 3D MRI volume per case.
- Target: binary segmentation mask with labels `{0, 1}`.
- Split: deterministic train/validation/test hold-out controlled by seed.

## Pretraining Design

- Backbone: MONAI `SwinUNETR` encoder (`feature_size=24`).
- Input views: two random global crops `(96, 96, 96)` and one local crop `(64, 64, 64)` resized to `(96, 96, 96)`.
- Objective:
  - global-global contrastive alignment
  - local-global contrastive alignment
- Loss: MONAI `ContrastiveLoss` with temperature scaling.
- Initialization transfer: only encoder `swinViT.*` weights are loaded into downstream segmentation.

## Downstream Segmentation Design

- Model: MONAI `SwinUNETR`, binary output with background and tumor.
- Loss: `DiceCELoss`.
- Validation/Test inference: `sliding_window_inference`.
- Main metrics:
  - Dice (foreground only)
  - HD95

## Comparison Protocol

1. Baseline: train segmentation from scratch.
2. Pretrained: initialize encoder from VOCO-style pretraining, then fine-tune full segmentation model.
3. Keep split, seed, patch size, optimizer family, and evaluation pipeline identical.

## Reproducibility Controls

- Fixed random seed.
- Saved split manifest.
- Saved JSON config per experiment.
- Saved best and last checkpoints.
- Saved learning curves and per-case metrics.

## Recommended Paper Tables

1. Main comparison table: Dice and HD95 for baseline vs pretrained.
2. Training efficiency table: epochs-to-best and wall-clock time if GPU logs are available.
3. Ablation table:
   - local loss weight
   - crop sizes
   - pretraining epochs
   - feature size

## Notes On Current Machine

- The current detected environment is CPU-only, so the code is ready but full 3D training should be run on your actual GPU machine.
- `num_workers=0` is intentionally conservative for Windows + HDF5 stability. Increase later if your runtime is stable.