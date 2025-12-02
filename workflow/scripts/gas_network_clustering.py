"""Cluster gas grid to shapes.

TODO: should keep incomming connections too!
"""

import sys
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

if TYPE_CHECKING:
    snakemake: Any
sys.stderr = open(snakemake.log[0], "w")

# Density at normal temperature and pressure (closer to operating conditions than STP)
# https://www.engineeringtoolbox.com/gas-density-d_158.html
CH4_KG_M3 = 0.668
# Typical values for natural gas (CH4)
# https://ocw.tudelft.nl/wp-content/uploads/Summary_table_with_heating_values_and_CO2_emissions.pdf
CH4_LHV_MJ_PER_KG = 50
MAXIMUM_THEORETICAL_H2_SHARE = 0.88  # 88% of original capacity



def connection_mapping(geometry, connection_regions):
    gdf = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

    connection_map = gpd.sjoin(gdf, connection_regions, how="left", predicate="within")

    if "id" in connection_map.columns:
        code_to_return = connection_map.id
    else:
        code_to_return = connection_map.country_code

    return connection_map.index_right, code_to_return


def read_and_concat_offshore_pipes(offshore_path, onshore_df):
    offshore_df = gpd.read_file(offshore_path).rename(
        columns=dict(StartBus="start_point", EndBus="end_point")
    )

    offshore_df["pipeline_type"] = "offshore"

    return pd.concat([onshore_df, offshore_df])


def cluster_onshore_pipes(pipelines_path, shapes_path):
    gas_pipelines = gpd.read_parquet(pipelines_path)
    connection_regions = gpd.read_file(shapes_path)

    # startpoint and endpoint mapping

    for point in ["start_point", "end_point"]:
        bus_map, country_code = connection_mapping(
            gas_pipelines[point], connection_regions
        )

        bus_point = point.replace("point", "bus")
        gas_pipelines[bus_point] = bus_map
        gas_pipelines[bus_point + "_country"] = country_code
        gas_pipelines[point] = gas_pipelines[bus_point].map(
            connection_regions.to_crs(3035).centroid.to_crs(4326)
        )

    # drop pipelines outside the regions
    gas_pipelines = gas_pipelines.loc[
        ~gas_pipelines.start_bus.isna() & ~gas_pipelines.end_bus.isna()
    ]

    # drop pipelines in the same region
    gas_pipelines = gas_pipelines.loc[gas_pipelines.start_bus != gas_pipelines.end_bus]

    # recaulcate the pipeline length
    gas_pipelines["geometry"] = gas_pipelines.apply(
        lambda x: LineString(
            [
                (x["start_point"].x, x["start_point"].y),
                (x["end_point"].x, x["end_point"].y),
            ]
        ),
        axis=1,
    )

    gas_pipelines["pipeline_type"] = "onshore"

    gas_pipelines.sort_index(axis=1, inplace=True)
    gas_pipelines.drop(["start_point", "end_point"], axis=1, inplace=True)

    # gas_pipelines = read_and_concat_offshore_pipes(offshore_path,gas_pipelines)
    gas_pipelines["length_km"] = gas_pipelines.to_crs(3035).geometry.length / 1000

    gas_pipelines["energy_cap"] = gas_pipelines.apply(estimate_capacity, axis=1)  # MW

    return gas_pipelines


def pipe_sectioning(gas_pipelines):
    large_pipes = gas_pipelines[
        (gas_pipelines.diameter_mm > 950) & (gas_pipelines.max_pressure_bar >= 70)
    ]

    medium_pipes = gas_pipelines[
        (gas_pipelines.diameter_mm > 700)
        & (gas_pipelines.diameter_mm <= 950)
        & (gas_pipelines.max_pressure_bar >= 50)
    ]

    small_pipes = gas_pipelines[
        (gas_pipelines.diameter_mm <= 700) & (gas_pipelines.max_pressure_bar >= 50)
    ]

    return {
        "large_pipes (>900 mm)": large_pipes,
        "medium_pipes (700 - 950 mm)": medium_pipes,
        "small_pipes (<700 mm)": small_pipes,
    }


def estimate_capacity(row):
    if row.max_cap_method in ["raw", "Median"]:
        capacity = (
            row.max_cap_M_m3_per_d
            * CH4_KG_M3
            * CH4_LHV_MJ_PER_KG
            * 24
            * 1000
            * MAXIMUM_THEORETICAL_H2_SHARE
            / 3600
        )

    else:
        #  Based on p.18 of https://ehb.eu/files/downloads/ehb-report-220428-17h00-interactive-1.pdf

        # slopes definitions
        m0 = (1400 - 0) / (500 - 0)
        m1 = (5500 - 1400) / (900 - 500)
        m2 = (15300 - 5500) / (1200 - 900)

        # intercept
        a0 = 0
        a1 = -3725
        a2 = -23900

        if row.diameter_mm < 501:
            capacity = a0 + m0 * row.diameter_mm

        elif row.diameter_mm < 901:
            capacity = a1 + m1 * row.diameter_mm

        else:
            capacity = a2 + m2 * row.diameter_mm

    return capacity


def set_index(pipe):
    return f"{pipe.start_bus_country}::{pipe.end_bus_country}"
    # if pipe.start_bus < pipe.end_bus:
    #     return f"{pipe.start_bus_country}::{pipe.end_bus_country}"

    # return f"{pipe.end_bus_country}::{pipe.start_bus_country}"


def set_pipe_directions(gas_pipelines):
    gas_pipelines.index = gas_pipelines.apply(set_index, axis=1)
    gas_pipelines["one_way"] = gas_pipelines.is_bothDirection.apply(
        lambda bi_direction: 0 if bi_direction else 1
    )

    return gas_pipelines.sort_index(axis=1).drop("is_bothDirection", axis=1)


def aggregate_parallel_pipes(gas_pipelines):
    how_to_aggregate = dict(
        end_bus="first",
        start_bus="first",
        pipeline_type="first",
        diameter_method="first",
        diameter_mm="mean",
        name="".join,
        length_km="mean",
        max_cap_M_m3_per_d="max",
        max_cap_method="first",
        max_pressure_bar="max",
        one_way="min",
        energy_cap="sum",
        geometry="first",
    )

    return gas_pipelines.groupby(gas_pipelines.index).agg(how_to_aggregate)[
        [*how_to_aggregate]
    ]


def create_sections_clustered_gas_pipes(pipes_file, shapes_file, output_file):
    gas_pipelines = cluster_onshore_pipes(pipes_file, shapes_file)

    pipes = pipe_sectioning(gas_pipelines)

    for pipe, df in pipes.items():
        df = set_pipe_directions(df)
        pipes[pipe] = aggregate_parallel_pipes(df)

    df = pd.concat(pipes)
    df.index.names = ["type", "locs"]
    columns = [
        "type",
        "locs",
        "diameter_mm",
        "length_km",
        "max_pressure_bar",
        "energy_cap",
        "one_way",
        "geometry",
    ]

    gdf = gpd.GeoDataFrame(df.reset_index()[columns])
    gdf["energy_cap_unit"] = "MW"
    gdf.to_file(output_file)


if __name__ == "__main__":
    create_sections_clustered_gas_pipes(
        pipes_file=snakemake.input.scigrid,
        shapes_file=snakemake.input.regions,
        output_file=snakemake.output.clusters,
    )
