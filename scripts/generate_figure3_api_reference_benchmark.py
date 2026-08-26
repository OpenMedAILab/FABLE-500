#!/usr/bin/env python3
"""Generate Figure 3 for the API-based reference benchmark."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT
ANALYSIS_DIR = DATASET_DIR / "benchmark" / "analysis" / "api_reference_test100"
FIGURE_DIR = DATASET_DIR / "figures"

DISEASE_ORDER = [
    "Cataract",
    "Vitreous hemorrhage",
    "High myopia",
    "Refractive error",
    "Retinal detachment",
]

SYSTEM_ORDER = [
    "report_text_only",
    "fundus_only",
    "bscan_only",
    "fundus_bscan_image_only",
    "full_case_multimodal",
    "api_agentic_full_case_workflow",
]

PLOT_LABELS = {
    "report_text_only": "Report\ntext",
    "fundus_only": "Fundus",
    "bscan_only": "B-scan",
    "fundus_bscan_image_only": "Image\nfusion",
    "full_case_multimodal": "Full\ncase",
    "api_agentic_full_case_workflow": "Workflow",
}

BAR_COLORS = {
    "report_text_only": "#9aa1a9",
    "fundus_only": "#9aa1a9",
    "bscan_only": "#9aa1a9",
    "fundus_bscan_image_only": "#9aa1a9",
    "full_case_multimodal": "#4c78a8",
    "api_agentic_full_case_workflow": "#f28e2b",
}


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(ANALYSIS_DIR / "summary.csv")
    paired = pd.read_csv(ANALYSIS_DIR / "paired_comparisons.csv")
    per_class = pd.read_csv(ANALYSIS_DIR / "per_class.csv")
    return summary, paired, per_class


def ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return summary.set_index("system_id").loc[SYSTEM_ORDER].reset_index()


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.07, label, transform=ax.transAxes, fontweight="bold", fontsize=13)


def format_p_value(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def panel_metric_bar(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    lo_col: str,
    hi_col: str,
    ylabel: str,
    ylim: tuple[float, float],
    label_offset: float,
) -> None:
    x = np.arange(len(summary))
    values = summary[metric].to_numpy()
    err = np.vstack(
        [
            values - summary[lo_col].to_numpy(),
            summary[hi_col].to_numpy() - values,
        ]
    )
    colors = [BAR_COLORS[s] for s in summary["system_id"]]
    ax.bar(x, values, yerr=err, capsize=3, color=colors, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([PLOT_LABELS[s] for s in summary["system_id"]])
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    for i, (value, hi) in enumerate(zip(values, summary[hi_col].to_numpy())):
        ax.text(i, hi + label_offset, f"{value:.2f}", ha="center", va="bottom", fontsize=8)


def figure3(summary: pd.DataFrame, paired: pd.DataFrame, per_class: pd.DataFrame) -> plt.Figure:
    summary = ordered_summary(summary)
    fig = plt.figure(figsize=(10.2, 7.6))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.16],
        width_ratios=[1.06, 0.94],
        hspace=0.52,
        wspace=0.45,
    )

    ax = fig.add_subplot(gs[0, 0])
    panel_metric_bar(
        ax,
        summary,
        metric="accuracy",
        lo_col="accuracy_ci95_low",
        hi_col="accuracy_ci95_high",
        ylabel="Accuracy",
        ylim=(0, 0.60),
        label_offset=0.018,
    )
    add_panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    panel_metric_bar(
        ax,
        summary,
        metric="macro_f1",
        lo_col="macro_f1_ci95_low",
        hi_col="macro_f1_ci95_high",
        ylabel="Macro-F1",
        ylim=(0, 0.48),
        label_offset=0.014,
    )
    add_panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, 0])
    heat = (
        per_class.pivot(index="class", columns="system_id", values="sensitivity")
        .reindex(DISEASE_ORDER)
        .loc[:, SYSTEM_ORDER]
    )
    im = ax.imshow(heat.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(SYSTEM_ORDER)))
    ax.set_xticklabels([PLOT_LABELS[s] for s in SYSTEM_ORDER])
    ax.set_yticks(np.arange(len(DISEASE_ORDER)))
    ax.set_yticklabels([wrap(d, 18) for d in DISEASE_ORDER])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = float(heat.values[i, j])
            color = "white" if val > 0.55 else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.05)
    cbar.set_label("Sensitivity")
    add_panel_label(ax, "C")

    ax = fig.add_subplot(gs[1, 1])
    paired_order = [
        "report_text_only",
        "fundus_only",
        "bscan_only",
        "fundus_bscan_image_only",
        "api_agentic_full_case_workflow",
    ]
    paired = paired.set_index("comparison_system").loc[paired_order].reset_index()
    y = np.arange(len(paired))
    delta = paired["accuracy_delta_reference_minus_comparison"].to_numpy()
    lo = paired["accuracy_delta_ci95_low"].to_numpy()
    hi = paired["accuracy_delta_ci95_high"].to_numpy()
    xerr = np.vstack([delta - lo, hi - delta])
    ax.axvline(0, color="#555555", lw=1)
    ax.errorbar(delta, y, xerr=xerr, fmt="o", color="#e15759", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([PLOT_LABELS[s].replace("\n", " ") for s in paired["comparison_system"]])
    ax.set_xlabel("Accuracy delta: full case minus comparator")
    ax.set_xlim(-0.15, 0.39)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    for i, row in paired.iterrows():
        ax.text(
            row["accuracy_delta_ci95_high"] + 0.015,
            i,
            f"P={format_p_value(float(row['accuracy_mcnemar_exact_p']))}",
            va="center",
            fontsize=8,
        )
    add_panel_label(ax, "D")
    return fig


def save_outputs(fig: plt.Figure) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / "Figure_3_api_reference_benchmark.png"
    tif = FIGURE_DIR / "Figure_3_api_reference_benchmark.tif"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(tif, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {png}")
    print(f"Saved {tif}")


def main() -> int:
    setup_matplotlib()
    summary, paired, per_class = read_inputs()
    fig = figure3(summary, paired, per_class)
    save_outputs(fig)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
