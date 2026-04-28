from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from brats_voco.utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified launcher for the BraTS2020 VOCO-style research pipeline.")
    parser.add_argument(
        "stage",
        choices=["split", "pretrain", "baseline", "finetune", "compare", "validate"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--config", default="experiments/brats2020_voco_swinunetr.json")
    parser.add_argument("--pretrained-checkpoint", default=None)
    parser.add_argument("--resume-from", default=None, help="Path to checkpoint to resume for pretrain/baseline/finetune")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto-resume from checkpoint_last.pt")
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--pretrained-summary", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    print(" ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    python = sys.executable

    if args.stage == "split":
        run_command(
            [
                python,
                "prepare_brats_split.py",
                "--data-dir",
                config["data_dir"],
                "--output",
                config["split_path"],
                "--seed",
                str(config["seed"]),
                "--train-ratio",
                str(config["train_ratio"]),
                "--val-ratio",
                str(config["val_ratio"]),
            ]
        )
        return

    if args.stage == "validate":
        command = [python, "validate_setup.py", "--config", args.config]
        if args.output:
            command.extend(["--output", args.output])
        run_command(command)
        return

    if args.stage == "pretrain":
        command = [python, "-m", "brats_voco.train_voco_pretrain", "--config", args.config]
        if args.output:
            command.extend(["--output-dir", args.output])
        if args.resume_from:
            command.extend(["--resume-from", args.resume_from])
        if args.no_auto_resume:
            command.append("--no-auto-resume")
        run_command(command)
        return

    if args.stage in {"baseline", "finetune"}:
        command = [
            python,
            "-m",
            "brats_voco.train_segmentation",
            "--config",
            args.config,
            "--mode",
            "baseline" if args.stage == "baseline" else "pretrained",
        ]
        if args.stage == "finetune":
            checkpoint = args.pretrained_checkpoint or str(Path(config["output_root"]) / "pretrain_voco" / "checkpoint_best.pt")
            command.extend(["--pretrained-checkpoint", checkpoint])
        if args.output:
            command.extend(["--output-dir", args.output])
        if args.resume_from:
            command.extend(["--resume-from", args.resume_from])
        if args.no_auto_resume:
            command.append("--no-auto-resume")
        run_command(command)
        return

    if args.stage == "compare":
        baseline_summary = args.baseline_summary or str(Path(config["output_root"]) / "segmentation_baseline" / "summary.json")
        pretrained_summary = args.pretrained_summary or str(Path(config["output_root"]) / "segmentation_pretrained" / "summary.json")
        output = args.output or str(Path(config["output_root"]) / "comparison.md")
        run_command(
            [
                python,
                "compare_experiments.py",
                "--baseline",
                baseline_summary,
                "--pretrained",
                pretrained_summary,
                "--output",
                output,
            ]
        )


if __name__ == "__main__":
    main()
