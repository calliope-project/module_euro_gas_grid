"""Prepare Natural Earth landmass."""

import sys
from typing import TYPE_CHECKING, Any

import _plots
import _schemas
import geopandas as gpd
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    snakemake: Any
sys.stderr = open(snakemake.log[0], "w")


def plot(land_file: str, output_file: str):
    """Plot landmass."""
    landmass = gpd.read_parquet(land_file)
    fig, ax = plt.subplots(layout="constrained")

    landmass.plot(ax=ax, color="tab:blue")
    _plots.style_map_plot(ax, "Natural Earth landmass")
    fig.savefig(output_file, dpi=300)


def prepare_landmass(raw_file: str, output_file: str):
    """Prepare the landmass dataset."""
    land = gpd.read_file(raw_file)
    land = land[land["featurecla"] == "Land"]
    land = land.rename({"featurecla": "feature_class"}, axis="columns")
    land = land.reset_index(drop=True)
    _schemas.LandSchema.validate(land).to_parquet(output_file)


if __name__ == "__main__":
    prepare_landmass(
        raw_file=snakemake.input.raw_landmass, output_file=snakemake.output.landmass
    )
    plot(snakemake.output.landmass, snakemake.output.fig)
