"""
Publication Theme & Styling Rules for FSL Research Framework Visualizations.
Establishes 300 DPI high-resolution PNG & SVG exports with publication color palettes.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, List

# Publication Palette (Indigo, Teal, Amber, Rose, Emerald, Blue, Purple)
PALETTE = [
    "#4338CA",  # Indigo
    "#0D9488",  # Teal
    "#D97706",  # Amber
    "#E11D48",  # Rose
    "#059669",  # Emerald
    "#2563EB",  # Blue
    "#7C3AED",  # Purple
]


def apply_publication_theme():
    """Sets matplotlib global rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


def save_figure(fig: plt.Figure, output_path_stem: Union[str, Path]) -> List[Path]:
    """
    Saves figure in both PNG (300 DPI) and vector SVG formats.
    """
    stem_path = Path(output_path_stem)
    stem_path.parent.mkdir(parents=True, exist_ok=True)

    png_path = stem_path.with_suffix(".png")
    svg_path = stem_path.with_suffix(".svg")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")

    plt.close(fig)
    return [png_path, svg_path]
