"""Shape standardisation functions."""

import sys
from typing import TYPE_CHECKING, Any

import _plots
import _schemas
import _utils
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely import make_valid
from shapely.ops import transform, unary_union

if TYPE_CHECKING:
    snakemake: Any


def _swap_xy(g: gpd.GeoSeries) -> gpd.GeoSeries:
    return transform(lambda x, y, z=None: (y, x), g)


def _polygons_only(g):
    if g is None or g.is_empty or not g.is_valid:
        raise ValueError("Invalid/empty geometry.")
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    if g.geom_type != "GeometryCollection":
        raise ValueError(f"Non-area geometry: {g.geom_type}")
    polys = [p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")]
    if not polys:
        raise ValueError("No polygonal area in GeometryCollection.")
    return unary_union(polys)


def prepare_shapes(shape_file: str) -> gpd.GeoDataFrame:
    """Build a standardised shapefile for the module."""
    shapes = _schemas.ShapesSchema.validate(gpd.read_parquet(shape_file))
    # Drop maritime shapes (not supported)
    return shapes[shapes["shape_class"] == "land"]


def add_north_sea(shapes: gpd.GeoDataFrame, north_sea_file: str):
    """Append a North Sea polygon, fitting our shapes."""
    # Load and ensure a single North Sea shape
    raw = gpd.read_file(north_sea_file).dissolve().reset_index(drop=True)
    raw.geometry = raw.geometry.apply(_swap_xy)  # Fix inverted X / Y
    raw = _utils.to_crs(raw, shapes.crs)

    # Get a single valid geometry for the North Sea
    geom_ns = raw.geometry.iloc[0]
    if geom_ns is None or geom_ns.is_empty:
        raise RuntimeError("North sea geometry dissolved into nothingness!")
    geom_ns = make_valid(geom_ns, method="structure", keep_collapsed=False)

    # Clip giving land priority
    land_union = shapes.geometry.union_all()
    geom_ns = geom_ns.difference(land_union)
    geom_ns = make_valid(geom_ns, method="structure", keep_collapsed=False)
    geom_ns = _polygons_only(geom_ns)
    if geom_ns.is_empty:
        raise RuntimeError("North sea geometry was clipped into nothingness!")

    # Convert to our standard
    north_sea = gpd.GeoDataFrame(
        [
            {
                "shape_id": "UKN_marineregions_2350",
                "country_id": "UKN",
                "shape_class": "maritime",
                "geometry": geom_ns,
            }
        ],
        crs=shapes.crs,
    )
    return pd.concat([shapes, north_sea], ignore_index=True)


def plot(shape_file: gpd.GeoDataFrame) -> tuple[Figure, Axes]:
    """A simple plot of the produced shapes."""
    shapes = gpd.read_parquet(shape_file)
    fig, ax = plt.subplots(layout="compressed")
    shapes.plot("shape_class", ax=ax, categorical=True)
    _plots.style_map_plot(ax, "Processed shapes")
    return fig, ax


def main():
    """Build a valid shapes dataset for further processing.

    Has an option to append the North Sea.
    """
    shapes = prepare_shapes(snakemake.input.shapes)

    north_sea_file = snakemake.input.north_sea
    if north_sea_file:
        shapes = add_north_sea(shapes, north_sea_file)

    shapes_out_file = snakemake.output.shapes
    _schemas.ShapesSchema.validate(shapes).to_parquet(shapes_out_file)
    fig, _ = plot(shapes_out_file)
    fig.savefig(snakemake.output.fig, dpi=300)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")
    main()
