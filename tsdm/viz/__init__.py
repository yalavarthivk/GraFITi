r"""Plotting Functionality."""

__all__ = [
    # Constants
    "USE_TEX",
    # Functions
    "shared_grid_plot",
    "visualize_distribution",
    "plot_spectrum",
    "kernel_heatmap",
    "rasterize",
    "center_axes",
]

from tsdm.viz._config import USE_TEX
from tsdm.viz._image import kernel_heatmap
from tsdm.viz._plotting import (
    center_axes,
    plot_spectrum,
    rasterize,
    shared_grid_plot,
    visualize_distribution,
)
