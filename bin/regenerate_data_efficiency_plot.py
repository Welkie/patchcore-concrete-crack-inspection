"""
regenerate_data_efficiency_plot.py
-----------------------------------
Regenerate ONLY plot_data_efficiency.png from already-saved CSV results.
Run this on Kaggle as a standalone script or paste its body into a new notebook cell.

Usage:
    python bin/regenerate_data_efficiency_plot.py --results_path results
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16})


def regenerate_plot(results_path: str) -> None:
    pc_csv  = os.path.join(results_path, "patchcore_ablation_results.csv")
    sup_csv = os.path.join(results_path, "supervised", "results.csv")
    out_png = os.path.join(results_path, "plot_data_efficiency.png")

    assert os.path.exists(pc_csv),  f"Missing PatchCore CSV: {pc_csv}"
    assert os.path.exists(sup_csv), f"Missing supervised CSV: {sup_csv}"

    # ── Load PatchCore ablation results ───────────────────────────────────────
    # CSV columns: Backbone, Coreset Ratio, Image AUROC, ...
    pc_results: dict[str, dict[float, float]] = {}
    with open(pc_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            backbone  = row["Backbone"]
            ratio_str = row["Coreset Ratio"]          # e.g. "10%"
            ratio     = float(ratio_str.strip("%")) / 100  # 0.10
            auroc     = float(row["Image AUROC"])
            pc_results.setdefault(backbone, {})[ratio] = auroc

    # ── Load supervised results ───────────────────────────────────────────────
    # CSV columns: Model, Num Positives, Epochs, Batch Size, LR, Accuracy, AUROC, ...
    sup_data: dict[str, list[tuple[int, float]]] = {}
    with open(sup_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model   = row["Model"]
            num_pos = int(row["Num Positives"])
            auroc   = float(row["AUROC"])
            sup_data.setdefault(model, []).append((num_pos, auroc))

    # ── Draw plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    # Supervised curves
    for model, pts in sup_data.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ax.plot(xs, ys, marker="v", linewidth=2.5, label=f"Supervised {model}")

    # PatchCore horizontal reference lines
    # ⚠ BUG FIX: use sorted() instead of set() to guarantee both backbones are
    #   drawn in a deterministic order with their correct, fixed colors.
    #   Also use distinct linestyles + linewidths because resnet18 AUROC (0.9969)
    #   and resnet50 AUROC (0.9971) differ by only ~0.0002 — they overlap if
    #   drawn identically.
    backbone_colors = {"resnet18": "red", "resnet50": "green"}
    for backbone in sorted(pc_results.keys()):      # resnet18 drawn first
        color = backbone_colors.get(backbone, "purple")
        ratios = pc_results[backbone]
        # Prefer 10% coreset ratio; fall back to 1% for quick_run mode
        auroc_val = ratios.get(0.10) or ratios.get(0.01)
        if auroc_val is None:
            print(f"[WARN] No 10%/1% coreset result found for {backbone}, skipping.")
            continue

        lw = 2.5 if backbone == "resnet18" else 1.5
        ls = "--"  if backbone == "resnet18" else "-."
        ax.axhline(
            y=auroc_val, color=color, linestyle=ls, linewidth=lw,
            label=f"PatchCore ({backbone}, 0 defects)"
        )
        # Right-side annotation makes lines distinguishable even when nearly equal
        ax.annotate(
            f"{auroc_val:.4f}",
            xy=(1.01, auroc_val),
            xycoords=("axes fraction", "data"),
            fontsize=9, color=color, va="center"
        )

    ax.set_xlabel("Number of Labeled Defect (Positive) Samples in Training")
    ax.set_ylabel("Test Image AUROC")
    ax.set_title("Data Efficiency Comparison: Unsupervised vs. Supervised")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 50, 100, 200])
    ax.set_xticklabels(["5", "10", "50", "100", "200"])
    ax.legend(loc="lower right")
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\n✅  Saved fixed plot → {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default="results",
                        help="Directory containing patchcore_ablation_results.csv "
                             "and supervised/results.csv")
    args = parser.parse_args()
    regenerate_plot(args.results_path)
