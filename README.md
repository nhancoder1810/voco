# BraTS2020 VOCO-Style Pretraining Pipeline

This repository is prepared for a research workflow that compares:

- `SwinUNETR` trained from scratch for binary BraTS2020 segmentation
- `SwinUNETR` initialized from a VOCO-style self-supervised contrastive pretraining stage

The dataset is expected to be the current `data/` folder of `.h5` files, where each file contains:

- `image`: single-channel 3D volume
- `label`: binary 3D segmentation mask

## Project Layout

- `brats_voco/data.py`: H5 loading, deterministic split creation, pretraining loader, segmentation loader
- `brats_voco/models.py`: VOCO-style pretrainer, SwinUNETR builder, checkpoint transfer
- `brats_voco/train_voco_pretrain.py`: self-supervised pretraining entry point
- `brats_voco/train_segmentation.py`: supervised segmentation training entry point
- `prepare_brats_split.py`: creates the deterministic hold-out split manifest
- `compare_experiments.py`: compares baseline and pretrained summaries
- `run_experiment.py`: unified launcher for the whole workflow
- `validate_setup.py`: one-shot sanity check for data, VOCO forward, segmentation forward, and encoder transfer
- `experiments/brats2020_voco_swinunetr.json`: main experiment config
- `RESEARCH_PLAN.md`: paper-style study design

## Reproducibility

- Seed is fixed in the JSON config and applied through `set_seed(...)`.
- Train/validation/test split is saved to `experiments/brats2020_split.json`.
- Each experiment writes its own copied config, checkpoints, CSV logs, and summary JSON.

## Recommended Environment For Real Training

For the current local machine, the code has been validated structurally, but the detected PyTorch runtime is CPU-only.

For actual training on a VM, use:

- Python 3.10 to 3.12 recommended
- CUDA-enabled PyTorch
- 1 GPU with 16 GB to 24 GB VRAM

Suggested setup on the GPU VM:

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Adjust the CUDA wheel index to match the VM driver/CUDA stack.

Run a quick setup check before starting long training jobs:

```bash
python run_experiment.py validate
```

This writes `results/setup_validation.json`.

### VM Config Tuning Guide

- 16 GB VRAM:
	- keep `segmentation.batch_size=1`
	- keep `segmentation.roi_size=[96,96,96]`
	- set `model.use_checkpoint=true`
	- keep `model.feature_size=48` only if memory allows; otherwise reduce to `24`
	- keep `pretrain.batch_size>=2` (contrastive needs negatives)
- `num_workers`:
	- current default is `8` for strong machines
	- reduce to `2-4` if dataloader is unstable on Windows
- AMP / TF32:
	- `pretrain.amp=true`, `segmentation.amp=true` are enabled by default when CUDA is available
	- TF32 acceleration is enabled automatically on CUDA in training scripts
- 24 GB VRAM:
	- keep current defaults and optionally increase `segmentation.sw_batch_size` to 3 or 4
- CPU-only debug:
	- reduce epochs and use `--max-train-steps` / `--max-val-steps` for quick code checks

## Main Commands

Create split:

```bash
python run_experiment.py split
```

Run VOCO-style pretraining:

```bash
python run_experiment.py pretrain
```

Resume VOCO pretraining from last checkpoint in output dir (auto):

```bash
python run_experiment.py pretrain
```

Resume VOCO pretraining from a specific checkpoint:

```bash
python run_experiment.py pretrain --resume-from results/pretrain_voco/checkpoints/checkpoint_epoch_0020.pt
```

Run segmentation baseline from scratch:

```bash
python run_experiment.py baseline
```

Resume baseline segmentation training:

```bash
python run_experiment.py baseline --resume-from results/segmentation_baseline/checkpoint_last.pt
```

Run segmentation fine-tuning from the pretrained encoder:

```bash
python run_experiment.py finetune
```

Resume finetuning segmentation training:

```bash
python run_experiment.py finetune --resume-from results/segmentation_pretrained/checkpoint_last.pt
```

Compare final results:

```bash
python run_experiment.py compare
```

Validate full setup:

```bash
python run_experiment.py validate
```

## Direct Module Commands

If you want full control instead of the launcher:

```bash
python -m brats_voco.train_voco_pretrain --config experiments/brats2020_voco_swinunetr.json
python -m brats_voco.train_segmentation --config experiments/brats2020_voco_swinunetr.json --mode baseline
python -m brats_voco.train_segmentation --config experiments/brats2020_voco_swinunetr.json --mode pretrained --pretrained-checkpoint results/pretrain_voco/checkpoint_best.pt
python compare_experiments.py --baseline results/segmentation_baseline/summary.json --pretrained results/segmentation_pretrained/summary.json --output results/comparison.md
python validate_setup.py --config experiments/brats2020_voco_swinunetr.json
```

## Research Defaults In The Current Config

- Seed: `42`
- Split: `70 / 15 / 15`
- Pretraining patch size: `96 x 96 x 96`
- Local crop size: `64 x 64 x 64`
- Grid patch mode: enabled with `grid_size=4`
- Random drop patch mode: enabled
- Segmentation ROI size: `96 x 96 x 96`
- Segmentation model: MONAI `SwinUNETR`
- `SwinUNETR use_v2=true`, `feature_size=48`, `use_checkpoint=true`
- Loss: `DiceCELoss`
- Metrics: foreground Dice and HD95
- Cache rate: `1.0` (full RAM caching)

## Notes About The VOCO-Style Implementation

This code uses a practical VOCO-style contrastive design for 3D medical data:

- two global crops from the same volume
- one local crop aligned to the same global semantic content
- one grid-based patch view
- one random-drop patch view
- global-global, local-global, grid-global, and drop-global contrastive losses

The pretraining loss is:

`L = L_global + w_local * L_local + w_grid * L_grid + w_drop * L_drop`

where defaults are `w_local=w_grid=w_drop=0.5`.

This keeps the implementation lightweight and directly transferable to downstream segmentation while remaining close to the intended self-supervised representation-learning objective.

## Expected Outputs

Pretraining writes under `results/pretrain_voco/`:

- `checkpoint_best.pt`
- `checkpoint_last.pt`
- `checkpoints/checkpoint_epoch_XXXX.pt`
- `history.csv`
- `curves.png`
- `summary.json`
- `model_architecture.txt`
- `embedding_pca_epoch_*.png`

Segmentation writes under `results/segmentation_baseline/` and `results/segmentation_pretrained/`:

- `checkpoint_best.pt`
- `checkpoint_last.pt`
- `checkpoints/checkpoint_epoch_XXXX.pt`
- `history.csv`
- `test_case_metrics.csv`
- `curves.png`
- `summary.json`

Comparison writes:

- `results/comparison.md`
- `results/comparison.csv`
- `results/comparison.json`
- `results/comparison_plot.png`

Validation writes:

- `results/setup_validation.json`

## Checkpoint And Resume Behavior

- Training auto-resumes from `checkpoint_last.pt` in the output directory when present.
- Use `--no-auto-resume` to force a fresh run.
- Each checkpoint stores model, optimizer, scheduler, AMP scaler state, best metric, and training history.
- Checkpoint frequency is controlled by:
	- `pretrain.checkpoint_every`, `pretrain.save_epoch_checkpoints`
	- `segmentation.checkpoint_every`, `segmentation.save_epoch_checkpoints`

## Practical Next Step On The VM

1. Copy this folder to the GPU VM.
2. Create a fresh environment with CUDA-enabled PyTorch.
3. Run `python run_experiment.py split`.
4. Run `python run_experiment.py pretrain`.
5. Run both downstream experiments and compare.