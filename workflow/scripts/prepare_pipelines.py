"""Filter SciGrid pipelines to fit our data standards.

Some portions of this code were adapted from PyPSA-Eur (https://github.com/pypsa/pypsa-eur)
Copyright (c) 2017-2024 The PyPSA-Eur Authors
Licensed under the MIT License
Commit: 822a92729e6973aa3aff741d6c94f1da2c75e8b2
"""

import sys
from typing import TYPE_CHECKING, Any

import _schemas
import country_converter as coco
import geopandas as gpd
import numpy as np
import pandas as pd
from cmap import Colormap
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    snakemake: Any
sys.stderr = open(snakemake.log[0], "w")

# Density at normal temperature and pressure (closer to operating conditions than STP)
# https://www.engineeringtoolbox.com/gas-density-d_158.html
CH4_KG_M3 = 0.668
# Typical values for natural gas (CH4)
# https://ocw.tudelft.nl/wp-content/uploads/Summary_table_with_heating_values_and_CO2_emissions.pdf
CH4_HHV_MJ_PER_KG = 55


def _build_country_translator(pipes: pd.Series) -> dict[str, str]:
    """Build a dictionary to translate country names to ISO3."""
    scigrid_countries: set = set()
    for i in pipes:
        if isinstance(i, str):
            scigrid_countries.update([i])
        else:
            scigrid_countries.update(set(i))
    # Special case
    converter = {
        k: coco.convert(k, to="iso3", not_found=np.nan)
        for k in scigrid_countries
        if k != "XX"
    }
    converter["XX"] = "XXX"
    return converter


def _line_midpoint_safe(geom):
    """Fallback function to get the midpoint in the line."""
    if geom is None or geom.is_empty:
        return None
    g = geom
    if g.geom_type == "MultiLineString":
        g = max(g.geoms, key=lambda ls: ls.length)
    if getattr(g, "length", 0) == 0:
        return g.centroid
    return g.interpolate(0.5, normalized=True)


def _diameter_to_capacity(pipe_diameter_mm: float) -> float:
    """Estimate natural gas (CH4) pipeline capacity from diameter.

    These values have been back-converted from the European Hydrogen Backbone report,
    using the stated  0.8*MW_CH4 = MW_H2 conversion assumption in p.15.
    https://ehb.eu/files/downloads/2020_European-Hydrogen-Backbone_Report.pdf

    Adapted from PyPSA-Eur (https://github.com/pypsa/pypsa-eur).

    Args:
        pipe_diameter_mm (float): Pipe diameter [mm]

    Returns:
        float: Pipeline capacity [MW]
    """
    # Anchor points: (diameter_mm, capacity_MW)
    p0 = (0.0, 0.0)
    p1 = (500.0, 1500.0)
    p2 = (600.0, 5000.0)
    p3 = (900.0, 11250.0)
    p4 = (1200.0, 21700.0)

    def line_through(pa, pb):
        """Return (m, a) for y = a + m*x through points pa,pb."""
        (x1, y1), (x2, y2) = pa, pb
        m = (y2 - y1) / (x2 - x1)  # [MW/mm]
        a = y1 - m * x1  # [MW]
        return m, a

    d = pipe_diameter_mm
    if d < p1[0]:
        m0, a0 = line_through(p0, p1)
        return a0 + m0 * d
    elif d < p2[0]:
        m1, a1 = line_through(p1, p2)
        return a1 + m1 * d
    elif d < p3[0]:
        m2, a2 = line_through(p2, p3)
        return a2 + m2 * d
    else:
        m3, a3 = line_through(p3, p4)
        return a3 + m3 * d


def standardise_pipelines(pipelines_file: str) -> gpd.GeoDataFrame:
    """Clean the SciGrid dataset."""
    pipes = gpd.read_file(pipelines_file)
    country_translator = _build_country_translator(pipes["country_code"])
    param_cols = [
        "diameter_mm",
        "max_cap_M_m3_per_d",
        "is_bothDirection",
        "length_km",
        "max_pressure_bar",
    ]
    param = pipes.param.apply(pd.Series)[param_cols]
    method_cols = {
        "diameter_mm": "diameter_method",
        "max_cap_M_m3_per_d": "max_cap_method",
    }
    method = pipes.method.apply(pd.Series)[method_cols.keys()].rename(
        columns=method_cols
    )
    pipes = pd.concat([pipes, param, method], axis="columns")
    # pipes["start_point"] = pipes.geometry.apply(lambda x: Point(x.coords[0]))
    # pipes["end_point"] = pipes.geometry.apply(lambda x: Point(x.coords[-1]))
    pipes["start_country_id"] = pipes.country_code.apply(
        lambda x: country_translator[x[0]]
    )
    pipes["end_country_id"] = pipes.country_code.apply(
        lambda x: country_translator[x[-1]]
    )
    return pipes


def estimate_ch4_capacity(
    pipes: gpd.GeoDataFrame,
    inferred_mm: float | None = None,
    recalculate_below_mw: float | None = None,
    capacity_correction_threshold: float = 8,
) -> gpd.GeoDataFrame:
    """Estimate natural gas capacity for each pipeline segment.

    Adapted from PyPSA-Eur (https://github.com/pypsa/pypsa-eur).

    Args:
        pipes (gpd.GeoDataFrame):
            pipelines dataframe.
        inferred_mm (float | None, optional):
            replaces Median inferred diameters. Defaults to None.
        recalculate_below_mw (float | None, optional):
            capacities below this will use recalculated values. Defaults to None.
        capacity_correction_threshold (int, optional):
            Ratio threshold to trigger recalculation. Defaults to 8.

    Returns:
        gpd.GeoDataFrame: pipeline dataframe with CH4 capacity.
    """
    pipes = pipes.copy()
    conversion_factor = 1e6 * CH4_KG_M3 * CH4_HHV_MJ_PER_KG / (24 * 60 * 60)
    pipes["ch4_capacity_mw"] = pipes["max_cap_M_m3_per_d"] * conversion_factor

    # Handle inferred diameters
    inferred_mask = pipes["diameter_method"].eq("Median(max_cap_M_m3_per_d)")
    if inferred_mm is not None:
        pipes.loc[inferred_mask, "diameter_mm"] = inferred_mm
        pipes.loc[inferred_mask, "diameter_method"] = "Module recalculation"

    capacity_mw_diameter_based = pipes["diameter_mm"].apply(_diameter_to_capacity)
    # Nordstream pressure ranges from 170-220
    # https://en.wikipedia.org/wiki/Nord_Stream_1#Baltic_Sea_offshore_pipeline
    not_nordstream = pipes["max_pressure_bar"] < 170
    ratio = pipes["ch4_capacity_mw"] / capacity_mw_diameter_based
    discrepant_mask = not_nordstream & (
        (ratio > capacity_correction_threshold)
        | (ratio < 1 / capacity_correction_threshold)
    )

    low_mask = False
    if recalculate_below_mw is not None:
        low_mask = pipes["ch4_capacity_mw"] <= recalculate_below_mw

    # Replace with diameter based capacity
    correction_mask = discrepant_mask | low_mask
    pipes.loc[correction_mask, "ch4_capacity_mw"] = capacity_mw_diameter_based.loc[
        correction_mask
    ]
    pipes.loc[correction_mask, "max_cap_method"] = "Module recalculation"

    return pipes


def identify_offshore(
    pipes: gpd.GeoDataFrame,
    landmass_file: str,
    *,
    n_samples: int = 50,
    land_threshold: float = 0.5,
    projected_crs: str = "EPSG:3857",
) -> gpd.GeoDataFrame:
    """Add boolean column `is_offshore` to the dataset.

    Sample `n_samples` points along each (multi)line in normalized [0,1].
    Compute fraction of samples intersecting land. If land_fraction < land_threshold,
    label as offshore.
    """
    # Load land polygons and project to ensure distance calculations are correct.
    land = gpd.read_parquet(landmass_file)[["geometry"]]
    pipes_p = pipes.to_crs(projected_crs)
    land = land.to_crs(projected_crs)

    land_union = land.geometry.union_all()

    fracs = np.linspace(0.0, 1.0, n_samples)

    points = []
    seg_idx = []
    for idx, geom in pipes_p.geometry.items():
        if geom is None or geom.is_empty:
            continue
        g = geom
        if g.geom_type == "MultiLineString":
            g = max(g.geoms, key=lambda ls: ls.length)
        # If something weird sneaks in, fall back to using only the midpoint
        if getattr(g, "length", 0) == 0:
            p = _line_midpoint_safe(g)
            if p is not None:
                points.append(p)
                seg_idx.append(idx)
            continue

        for f in fracs:
            points.append(g.interpolate(float(f), normalized=True))
            seg_idx.append(idx)

    pts = gpd.GeoDataFrame({"seg_idx": seg_idx}, geometry=points, crs=pipes_p.crs)

    # Spatial join: mark whether each sampled point hits land
    on_land = pts.geometry.intersects(land_union)
    land_frac = on_land.groupby(pts["seg_idx"]).mean()
    land_frac = land_frac.reindex(pipes.index)

    pipes["is_offshore"] = land_frac < land_threshold
    return pipes


def prepare_pipelines(
    pipelines_file: str,
    landmass_file: str,
    projected_crs: str,
    impute_params: dict,
    output_file: str,
):
    """Clean and validate the pipelines dataset."""
    pipes = standardise_pipelines(pipelines_file)
    pipes = estimate_ch4_capacity(
        pipes,
        inferred_mm=impute_params.get("inferred_mm", None),
        recalculate_below_mw=impute_params.get("recalculate_below_mw", None),
    )
    pipes = identify_offshore(pipes, landmass_file, projected_crs=projected_crs)
    pipes = _schemas.PipelineSchema.validate(pipes)
    pipes.to_parquet(output_file)


def plot_offshore(pipes_file: str, output_file: str):
    """Plot pipelines, identifying 'unknown' segments."""
    pipes = gpd.read_parquet(pipes_file)
    # Unknown in either endpoint
    offshore = pipes["is_offshore"]

    gdf_off = pipes[offshore]
    gdf_land = pipes[~offshore]

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")

    # Land first, then 'unknown' on top
    if not gdf_land.empty:
        gdf_land.plot(ax=ax, color="tab:brown", linewidth=0.6)
    if not gdf_off.empty:
        gdf_off.plot(ax=ax, color="tab:blue", linewidth=1.0)

    ax.set_title("SciGrid gas pipelines")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.savefig(output_file, dpi=300)


def plot_ch4_capacity(pipes_file: str, output_file: str):
    """Plot estimated CH4 capacity."""
    pipes = gpd.read_parquet(pipes_file)
    fig, ax = plt.subplots(layout="constrained")
    cmap = Colormap("bids:fake_parula").to_mpl()
    ax = pipes.plot("ch4_capacity_mw", ax=ax, legend=True, cmap=cmap)
    ax.set_title("$CH_4$ pipeline capacity ($MW$)")
    fig.savefig(output_file, dpi=300)


if __name__ == "__main__":
    prepare_pipelines(
        pipelines_file=snakemake.input.raw_pipelines,
        landmass_file=snakemake.input.landmass,
        projected_crs=snakemake.params.projected_crs,
        impute_params=snakemake.params.imputation,
        output_file=snakemake.output.pipelines,
    )
    plot_offshore(snakemake.output.pipelines, snakemake.output.fig_offshore)
    plot_ch4_capacity(snakemake.output.pipelines, snakemake.output.fig_ch4_capacity)
