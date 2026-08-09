"""
Plotting Functions for FSL Research Framework.
Generates Bar Charts, Line Charts, Scatter Plots, Heatmaps, and Reliability Error Bar Charts.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.theme import apply_publication_theme, save_figure, PALETTE
from utilities.logger import get_logger

logger = get_logger("visualization_charts")
apply_publication_theme()


def plot_bar_chart(
    categories: List[str],
    values: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path_stem: Union[str, Path],
    errors: Optional[List[float]] = None,
    color: str = PALETTE[0],
) -> List[Path]:
    """Generates a publication-quality bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_pos = np.arange(len(categories))
    bars = ax.bar(
        x_pos,
        values,
        yerr=errors,
        capsize=4,
        color=color,
        edgecolor="#1E1B4B",
        alpha=0.85,
        width=0.55,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=15 if len(categories) > 4 else 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, fontweight="bold")

    # Value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}" if height < 10 else f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    logger.info(f"Generated Bar Chart: '{title}'")
    return save_figure(fig, output_path_stem)


def plot_line_chart(
    x_values: List[Any],
    series_dict: Dict[str, List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path_stem: Union[str, Path],
) -> List[Path]:
    """Generates a line chart showing scaling trajectories across conditions."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    markers = ["o", "s", "^", "D", "v"]
    for idx, (label, y_vals) in enumerate(series_dict.items()):
        color = PALETTE[idx % len(PALETTE)]
        marker = markers[idx % len(markers)]
        ax.plot(
            x_values,
            y_vals,
            marker=marker,
            linewidth=2.2,
            markersize=7,
            color=color,
            label=label,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB")

    logger.info(f"Generated Line Chart: '{title}'")
    return save_figure(fig, output_path_stem)


def plot_scatter_chart(
    x_values: List[float],
    y_values: List[float],
    labels: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path_stem: Union[str, Path],
) -> List[Path]:
    """Generates a scatter plot for efficiency vs accuracy trade-offs."""
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.scatter(
        x_values,
        y_values,
        c=PALETTE[0],
        s=80,
        alpha=0.9,
        edgecolor="#1E1B4B",
        zorder=3,
    )

    for i, txt in enumerate(labels):
        ax.annotate(
            txt,
            (x_values[i], y_values[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, fontweight="bold")

    logger.info(f"Generated Scatter Plot: '{title}'")
    return save_figure(fig, output_path_stem)


def plot_heatmap(
    matrix_df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path_stem: Union[str, Path],
) -> List[Path]:
    """Generates a 2D parameter interaction heatmap."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    cax = ax.matshow(matrix_df.values, cmap="YlGnBu")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_xticklabels(matrix_df.columns)
    ax.set_yticklabels(matrix_df.index)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=20, fontweight="bold")

    # Values in cells
    for i in range(len(matrix_df.index)):
        for j in range(len(matrix_df.columns)):
            val = matrix_df.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="black" if val < 0.7 else "white", fontsize=9)

    logger.info(f"Generated Heatmap: '{title}'")
    return save_figure(fig, output_path_stem)


def plot_reliability_chart(
    categories: List[str],
    values: List[float],
    errors: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path_stem: Union[str, Path],
) -> List[Path]:
    """Generates a reliability and stability chart with error bars."""
    return plot_bar_chart(
        categories=categories,
        values=values,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        output_path_stem=output_path_stem,
        errors=errors,
        color=PALETTE[1],
    )
