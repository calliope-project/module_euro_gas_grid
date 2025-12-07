"""Plot functions used in one or more rules."""

import geopandas as gpd
import pandas as pd
from matplotlib.axes import Axes


def get_padded_bounds(gdf: gpd.GeoDataFrame, pad_frac: float = 0.02):
    """Get a square with some padding."""
    minx, miny, maxx, maxy = gdf.total_bounds
    dx, dy = (maxx - minx), (maxy - miny)
    side = max(dx, dy)
    side = side if side > 0 else 1.0
    pad = pad_frac * side
    half = 0.5 * side + pad
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    return (cx - half, cx + half), (cy - half, cy + half)


def plot_density(ax: Axes, values: pd.Series, title: str):
    """Plot a simple density kernel of a numeric value."""
    values.plot.density(ax=ax)
    ax.set_xlim(values.min(), values.max())
    ax.tick_params(left=False, labelleft=False)
    ax.set_title(title)


def style_map_plot(
    ax: Axes, title: str, xlim: tuple | None = None, ylim: tuple | None = None
):
    """Standardise map plots (needs projected CRS)."""
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", anchor="C")
    ax.set_title(title)

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
