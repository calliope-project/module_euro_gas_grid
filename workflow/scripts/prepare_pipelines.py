"""Filter SciGrid pipelines to fit our data standards."""

import sys
from typing import TYPE_CHECKING, Any

import _schemas
import country_converter as coco
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Point

if TYPE_CHECKING:
    snakemake: Any
sys.stderr = open(snakemake.log[0], "w")


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
    pipes["start_point"] = pipes.geometry.apply(lambda x: Point(x.coords[0]))
    pipes["end_point"] = pipes.geometry.apply(lambda x: Point(x.coords[-1]))
    pipes["start_country_id"] = pipes.country_code.apply(
        lambda x: country_translator[x[0]]
    )
    pipes["end_country_id"] = pipes.country_code.apply(
        lambda x: country_translator[x[-1]]
    )
    return pipes


def identify_offshore(
    pipes: gpd.GeoDataFrame,
    landmass_file: str,
    *,
    n_samples: int = 100,
    land_threshold: float = 0.5,
    projected_crs: str = "EPSG:3857"
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


def prepare_pipelines(pipelines_file: str, landmass_file: str, output_file: str):
    """Clean and validate the pipelines dataset."""
    pipes = standardise_pipelines(pipelines_file)
    pipes = identify_offshore(pipes, landmass_file)

    pipes = _schemas.PipelineSchema.validate(pipes)
    pipes.to_parquet(output_file)


def plot(pipes_file: str, output_file: str):
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


if __name__ == "__main__":
    prepare_pipelines(
        pipelines_file=snakemake.input.raw_pipelines,
        landmass_file=snakemake.input.landmass,
        output_file=snakemake.output.pipelines,
    )
    plot(snakemake.output.pipelines, snakemake.output.fig)
