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


def plot(pipes_file: str, output_file: str):
    """Plot pipelines, identifying 'unknown' segments."""
    pipes = gpd.read_parquet(pipes_file)
    # Unknown in either endpoint
    offshore = pipes["start_country_id"].str.contains("XXX", na=False) | pipes[
        "end_country_id"
    ].str.contains("XXX", na=False)

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


def prepare_pipelines(raw_file: str, output_file: str):
    """Clean and validate the pipelines dataset."""
    pipes = gpd.read_file(raw_file)
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
    pipes = _schemas.PipelineSchema.validate(pipes)
    pipes.to_parquet(output_file)


if __name__ == "__main__":
    prepare_pipelines(
        raw_file=snakemake.input.raw, output_file=snakemake.output.pipelines
    )
    plot(snakemake.output.pipelines, snakemake.output.fig)
