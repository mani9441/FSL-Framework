"""
Visualization Engine Package for FSL Research Framework.
Provides publication-quality theme styling, bar/line/scatter/heatmap plotters, and master VisualizationEngine.
"""

from visualization.theme import apply_publication_theme, save_figure, PALETTE
from visualization.charts import (
    plot_bar_chart,
    plot_line_chart,
    plot_scatter_chart,
    plot_heatmap,
    plot_reliability_chart,
)
from visualization.engine import VisualizationEngine

__all__ = [
    "apply_publication_theme",
    "save_figure",
    "PALETTE",
    "plot_bar_chart",
    "plot_line_chart",
    "plot_scatter_chart",
    "plot_heatmap",
    "plot_reliability_chart",
    "VisualizationEngine",
]
