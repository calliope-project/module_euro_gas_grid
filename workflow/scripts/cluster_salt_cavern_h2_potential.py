"""Gas network clustering to shapes."""

import sys
from typing import TYPE_CHECKING, Any

import _plots
import _schemas
import _utils
import cmap
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    snakemake: Any


def get_area_km2(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Area of geometries in km^2."""
    factor = _utils.get_crs_meter_conversion_factor(gdf.crs)
    area_native = gdf.geometry.area
    area_m2 = area_native * (factor**2)
    return area_m2 / 1e6


def salt_cavern_potential_gwh(
    caverns: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    *,
    shape_id_col: str = "shape_id",
    storage_type_col: str = "storage_type",
    capacity_per_area_col: str = "gwh_per_km2",
    min_gwh: float = 1e-6,
) -> pd.DataFrame:
    """Compute salt cavern storage potential per shape_id in GWh."""
    storage_types = pd.Index(caverns[storage_type_col].dropna().unique())

    caverns = _utils.to_crs(caverns, shapes.crs)

    cav = caverns[[capacity_per_area_col, storage_type_col, "geometry"]].copy()
    shp = shapes[[shape_id_col, "geometry"]].copy()

    cav["area_caverns_km2"] = get_area_km2(cav)
    overlay = gpd.overlay(
        shp.reset_index(drop=True), cav.reset_index(drop=True), how="intersection"
    )

    overlay["share"] = get_area_km2(overlay) / overlay["area_caverns_km2"]
    overlay["e_nom_gwh"] = (
        overlay[capacity_per_area_col] * overlay["share"] * overlay["area_caverns_km2"]
    )

    out = (
        overlay.groupby([shape_id_col, storage_type_col])["e_nom_gwh"]
        .sum()
        .unstack(storage_type_col)
        .reindex(columns=storage_types)  # ensure all storage types always exist
        .fillna(0.0)  # clears NaNs introduced by unstack
        .rename(columns=lambda c: f"{c}_gwh")
        .reset_index()
    )

    # Remove negative numbers and small values.
    gwh_cols = out.columns.drop(shape_id_col)

    out[gwh_cols] = out[gwh_cols].mask(
        (out[gwh_cols] < 0) | (out[gwh_cols].abs() < min_gwh)
    )
    out = out.dropna(subset=gwh_cols, how="all")
    out[gwh_cols] = out[gwh_cols].fillna(0.0)

    out["total_gwh"] = out[gwh_cols].sum(axis="columns")
    return out


def plot(
    shapes,
    potential,
    *,
    shape_id_col="shape_id",
    value_col="total_gwh",
    colormap="chrisluts:I_Purple",
):
    """Plot saltcavern potential."""
    gdf = shapes[[shape_id_col, "geometry"]].merge(
        potential[[shape_id_col, value_col]], on=shape_id_col, how="left"
    )

    fig, ax = plt.subplots(figsize=(6, 6), layout="compressed")
    gdf.plot(
        ax=ax,
        column=value_col,
        cmap=cmap.Colormap(colormap).to_mpl(),
        legend=True,
        linewidth=0,
        missing_kwds={"color": "lightgrey"}
    )
    gdf.boundary.plot(ax=ax, color="black", lw=0.5)
    _plots.style_map_plot(ax, "Salt cavern H2 potential ($GWh$)")
    return fig, ax


def main():
    """Main snakemake process."""
    proj_crs = snakemake.params.projected_crs
    _utils.check_projected_crs(proj_crs)
    shapes = _utils.to_crs(gpd.read_parquet(snakemake.input.shapes), proj_crs)
    shapes = _schemas.ShapesSchema.validate(shapes)
    caverns = _utils.to_crs(gpd.read_parquet(snakemake.input.salt_caverns), proj_crs)

    min_gwh_tolerance = snakemake.params.min_gwh_tolerance
    potential = salt_cavern_potential_gwh(caverns, shapes, min_gwh=min_gwh_tolerance)
    potential = _schemas.H2Potential.validate(potential)
    potential.to_parquet(snakemake.output.salt_cavern_h2_potential)

    fig, _ = plot(shapes, potential)
    fig.savefig(snakemake.output.fig, dpi=300)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")
    main()
