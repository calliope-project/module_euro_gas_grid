"""Prepare Natural Earth Countries."""

import sys
from typing import TYPE_CHECKING, Any

import _plots
import _schemas
import country_converter as coco
import geopandas as gpd
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    snakemake: Any
sys.stderr = open(snakemake.log[0], "w")


def plot(land_file: str, output_file: str):
    """Plot countries."""
    landmass = gpd.read_parquet(land_file)
    fig, ax = plt.subplots(layout="compressed")

    landmass.plot(ax=ax, color="tab:purple")
    _plots.style_map_plot(ax, "Natural Earth countries")
    fig.savefig(output_file, dpi=300)


def prepare_countries(raw_file: str, output_file: str):
    """Prepare the countries dataset.

    Will only be used to assign import naming if necessary.
    """
    raw_countries = gpd.read_file(raw_file)
    countries = gpd.GeoDataFrame(
        {
            "sovereign_id": raw_countries["SOV_A3"],
            "sovereign_name": raw_countries["SOVEREIGNT"],
            "sovereign_type": raw_countries["TYPE"],
            "admin_name": raw_countries["ADMIN"],
            "geometry": raw_countries["geometry"],
        },
        crs=raw_countries.crs,
    )
    countries = countries.reset_index(drop=True)
    countries["admin_id"] = coco.convert(countries["admin_name"], to="iso3", not_found="XXX")
    _schemas.CountriesSchema.validate(countries).to_parquet(output_file)


if __name__ == "__main__":
    prepare_countries(
        raw_file=snakemake.input.raw_countries,
        output_file=snakemake.output.countries,
    )
    plot(snakemake.output.countries, snakemake.output.fig)
