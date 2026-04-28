from __future__ import annotations

import argparse
from pathlib import Path

from brats_voco.utils import plot_comparison_bars, read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and pretrained segmentation experiment summaries.")
    parser.add_argument("--baseline", required=True, help="Path to baseline summary.json")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained summary.json")
    parser.add_argument("--output", default="results/comparison.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = read_json(args.baseline)
    pretrained = read_json(args.pretrained)
    dice_gain = pretrained["test_dice"] - baseline["test_dice"]
    hd95_gain = baseline["test_hd95"] - pretrained["test_hd95"]

    rows = [
        {"setting": "baseline", "test_dice": baseline["test_dice"], "test_hd95": baseline["test_hd95"]},
        {"setting": "pretrained", "test_dice": pretrained["test_dice"], "test_hd95": pretrained["test_hd95"]},
    ]

    summary = {
        "baseline": rows[0],
        "pretrained": rows[1],
        "dice_gain": dice_gain,
        "hd95_gain": hd95_gain,
    }

    report = f"""# BraTS2020 VOCO-style Pretraining Comparison

| Setting | Test Dice | Test HD95 |
|---|---:|---:|
| No pretraining | {baseline['test_dice']:.4f} | {baseline['test_hd95']:.4f} |
| VOCO-style pretrained encoder | {pretrained['test_dice']:.4f} | {pretrained['test_hd95']:.4f} |

Dice improvement: {dice_gain:+.4f}

HD95 improvement: {hd95_gain:+.4f}
"""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    write_csv(rows, output_path.with_suffix(".csv"))
    write_json(summary, output_path.with_suffix(".json"))
    plot_comparison_bars(
        baseline_dice=baseline["test_dice"],
        pretrained_dice=pretrained["test_dice"],
        baseline_hd95=baseline["test_hd95"],
        pretrained_hd95=pretrained["test_hd95"],
        path=output_path.with_name("comparison_plot.png"),
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
