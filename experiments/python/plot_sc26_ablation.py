#!/usr/bin/env python3
"""Plot SC26 ablation study figure from experiments/results/ablation/ JSON files.

Produces a 1×2 panel figure:
  (a) Pure-CV QAOA  — Full vs w/o Gaussian Symbolic vs Eager Materialization
  (b) Hybrid JCH    — Full vs w/o Gaussian Symbolic vs Eager Materialization

Usage:
    python experiments/python/plot_sc26_ablation.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile

SCRIPT_PATH = pathlib.Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

MPL_CONFIG_DIR = pathlib.Path(tempfile.gettempdir()) / "hybridcvdv_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(REPO_ROOT))
from experiments.configs.paper_style import (
    SINGLE_COLUMN_PT,
    apply_paper_style,
    save_figure,
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
ABLATION_DIR = REPO_ROOT / "experiments" / "results" / "ablation_v2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "SC26submission" / "expplots"

MODE_SWEEP = [2, 3, 4, 5]

# ---------------------------------------------------------------------------
# Ablation configurations — each maps to a file naming suffix
# ---------------------------------------------------------------------------
CONFIGS = [
    {
        "suffix": "full",
        "label": "Gantry",
        "color": "#1b9e77",   # teal
        "marker": "o",
        "linestyle": "-",
    },
    {
        "suffix": "no_symbolic",
        "label": "w/o Gaussian Track",
        "color": "#945034",   # brown
        "marker": "s",
        "linestyle": "--",
    },
    {
        "suffix": "no_symbolic_dense_fock",
        "label": "w/o Gaussian Track + w/o Fock Optimization",
        "color": "#FFA800",   # orange
        "marker": "D",
        "linestyle": ":",
    },
]

# ---------------------------------------------------------------------------
# Two panels
# ---------------------------------------------------------------------------
PANELS = [
    {
        "title": "(a) Pure-CV QAOA",
        "file_prefix": "sc26_cv_qaoa_nm{nm}_c16",
    },
    {
        "title": "(b) Hybrid JCH (4 qubits)",
        "file_prefix": "sc26_jch_nq4_nm{nm}_c16",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_metric_ms(path: pathlib.Path, metric: str = "median_total_ms") -> float | None:
    """Extract a timing metric from an ablation JSON file."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        return None
    metrics = results[0].get("metrics", {})
    val = metrics.get(metric)
    if val is None:
        val = metrics.get("median_total_ms")
    return float(val) if val is not None else None


def build_series(
    panel: dict,
    config: dict,
    modes: list[int],
    ablation_dir: pathlib.Path,
    metric: str = "median_total_ms",
) -> list[float | None]:
    """Build a list of metric values for one (panel, config) combination."""
    values: list[float | None] = []
    for nm in modes:
        prefix = panel["file_prefix"].format(nm=nm)
        filename = f"{prefix}__{config['suffix']}.json"
        values.append(load_metric_ms(ablation_dir / filename, metric))
    return values


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------
def plot_ablation(
    ablation_dir: pathlib.Path,
    output_dir: pathlib.Path,
    metric: str = "median_total_ms",
) -> None:
    figsize = apply_paper_style(
        width_pt=SINGLE_COLUMN_PT,
        ncols=2,
        nrows=1,
        panel_aspect=4.0 / 3.0,
        font_size=7,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    for ax, panel in zip(axes, PANELS):
        # ax.set_title(panel["title"], fontweight="bold", pad=4)
        ax.set_xlabel("Number of Qumode")
        ax.set_yscale("log")
        ax.set_xticks(MODE_SWEEP)
        ax.set_xticklabels([str(m) for m in MODE_SWEEP])
        ax.grid(True, axis="y", alpha=0.35)
        ax.grid(True, axis="x", alpha=0.15, linestyle=":")
        ## set title under the figure
        ax.text(
            0.5, -0.5,
            panel["title"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
        ax.set_xlim(MODE_SWEEP[0] - 0.5, MODE_SWEEP[-1] + 0.5)
        for cfg in CONFIGS:
            values = build_series(panel, cfg, MODE_SWEEP, ablation_dir, metric)
            xs = [m for m, v in zip(MODE_SWEEP, values) if v is not None]
            ys = [v for v in values if v is not None]
            if not ys:
                print(f"  WARNING: no data for {panel['title']} / {cfg['label']}")
                continue

            ax.plot(
                xs,
                ys,
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                linestyle=cfg["linestyle"],
                linewidth=1.0,
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )

            # Annotate the slowdown ratio at the rightmost point
            if len(ys) >= 2:
                full_vals = build_series(panel, CONFIGS[0], MODE_SWEEP, ablation_dir, metric)
                full_ys = [v for v in full_vals if v is not None]
                if cfg["suffix"] != "full" and full_ys and ys:
                    ratio = ys[-1] / full_ys[-1] if full_ys[-1] else 0
                    if ratio > 1.5:
                        ax.annotate(
                            f"{ratio:.0f}×",
                            xy=(xs[-1]-0.3, ys[-1]),
                            xytext=(4, 2),
                            textcoords="offset points",
                            fontsize=7,
                            color=cfg["color"],
                        )

    axes[0].set_ylabel("Runtime (ms)")

    # Format y-axis ticks nicely
    for ax in axes:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v:.1f}" if v < 10 else f"{v:.0f}"
        ))

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.1),
        frameon=False,
        fontsize=7,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.30, left=0.08, right=0.97)
    saved = save_figure(fig, output_dir, "sc26_ablation")
    plt.close(fig)
    for p in saved:
        print(f"  Saved: {p}")


# ---------------------------------------------------------------------------
# Print a summary table to stdout
# ---------------------------------------------------------------------------
def print_summary(ablation_dir: pathlib.Path, metric: str = "median_total_ms") -> None:
    print(f"\n=== Ablation Summary ({metric}) ===")
    header = f"{'Panel':<24} {'nm':>3}"
    for cfg in CONFIGS:
        header += f"  {cfg['label']:>26}"
    print(header)
    print("-" * len(header))

    for panel in PANELS:
        for nm in MODE_SWEEP:
            row = f"{panel['title']:<24} {nm:>3}"
            full_ms = None
            for cfg in CONFIGS:
                val = build_series(panel, cfg, [nm], ablation_dir, metric)[0]
                if cfg["suffix"] == "full":
                    full_ms = val
                if val is not None:
                    ratio = val / full_ms if full_ms and full_ms > 0 else 0
                    row += f"  {val:>18.3f} ({ratio:>4.1f}×)"
                else:
                    row += f"  {'N/A':>26}"
            print(row)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SC26 ablation figure")
    parser.add_argument(
        "--ablation-dir",
        type=pathlib.Path,
        default=ABLATION_DIR,
        help="Directory containing ablation JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary table, skip plotting",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="median_total_ms",
        choices=["median_total_ms", "median_compute_ms"],
        help="Which timing metric to plot (default: median_total_ms)",
    )
    args = parser.parse_args()

    print_summary(args.ablation_dir, args.metric)

    if not args.summary_only:
        plot_ablation(args.ablation_dir, args.output_dir, args.metric)


if __name__ == "__main__":
    main()
