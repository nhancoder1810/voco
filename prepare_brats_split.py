from __future__ import annotations

import argparse

from brats_voco.data import prepare_data_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic train/val/test split for BraTS2020 H5 data.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="experiments/brats2020_split.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = prepare_data_split(
        data_dir=args.data_dir,
        output_path=args.output,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(
        f"split saved to {args.output} | train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}"
    )


if __name__ == "__main__":
    main()
