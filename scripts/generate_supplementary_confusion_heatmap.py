#!/usr/bin/env python3
"""Generate Figure S1 confusion-matrix heatmaps for the FABLE-500 supplement."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT
INPUT_CSV = (
    DATASET_DIR
    / "benchmark"
    / "analysis"
    / "api_reference_test100"
    / "confusion_matrix_long.csv"
)
FIGURE_DIR = DATASET_DIR / "figures"

LABELS = [
    "Cataract",
    "Vitreous hemorrhage",
    "High myopia",
    "Refractive error",
    "Retinal detachment",
]

LABEL_ABBR = {
    "Cataract": "Cat",
    "Vitreous hemorrhage": "VH",
    "High myopia": "HM",
    "Refractive error": "RE",
    "Retinal detachment": "RD",
}

SYSTEM_ORDER = [
    "report_text_only",
    "fundus_only",
    "bscan_only",
    "fundus_bscan_image_only",
    "full_case_multimodal",
    "api_agentic_full_case_workflow",
]

SYSTEM_TITLES = {
    "report_text_only": "B-scan report text",
    "fundus_only": "Fundus",
    "bscan_only": "B-scan",
    "fundus_bscan_image_only": "Fundus+B-scan images",
    "full_case_multimodal": "Full case",
    "api_agentic_full_case_workflow": "Workflow",
}


def matrix_for(df: pd.DataFrame, system_id: str) -> np.ndarray:
    sub = df[df["system_id"] == system_id]
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for i, true_label in enumerate(LABELS):
        for j, pred_label in enumerate(LABELS):
            val = sub[(sub["true_label"] == true_label) & (sub["predicted_label"] == pred_label)]["n"]
            mat[i, j] = int(val.iloc[0]) if not val.empty else 0
    return mat


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    fig = plt.figure(figsize=(7.6, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(len(SYSTEM_ORDER))]

    vmax = 20
    image = None
    tick_labels = [LABEL_ABBR[x] for x in LABELS]
    panel_letters = ["A", "B", "C", "D", "E", "F"]

    for ax, system_id, letter in zip(axes, SYSTEM_ORDER, panel_letters):
        mat = matrix_for(df, system_id)
        image = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(tick_labels)
        ax.set_title(f"{letter}. {SYSTEM_TITLES[system_id]}", loc="left", pad=4, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        ax.set_xticks(np.arange(-0.5, len(LABELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(LABELS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                color = "white" if value >= 12 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)

    if image is not None:
        cb = fig.colorbar(image, ax=axes, location="right", shrink=0.92, pad=0.02)
        cb.set_label("Cases, n", rotation=90)
        cb.set_ticks([0, 5, 10, 15, 20])

    outputs = [
        FIGURE_DIR / "Figure_S1_confusion_matrices.png",
        FIGURE_DIR / "Figure_S1_confusion_matrices.tif",
    ]
    for out_path in outputs:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    for out_path in outputs:
        print(out_path)


if __name__ == "__main__":
    main()
