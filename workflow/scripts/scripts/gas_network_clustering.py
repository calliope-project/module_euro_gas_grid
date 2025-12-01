
breakpoint()
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point,LineString
import json
import zipfile
import io

NG_DENSITY = 0.68 # kg/Sm3
NG_ENERGY_CONTENT = 55 # MJ/kg
MAXIMUM_THEORETICAL_H2_SHARE = 0.88 # 88% of original capacity


def read_clean_scigrid(
        file_path
):

    # File inside the zip that you want to extract and read
    geojson_file_path = 'gas_grid/data/IGGINL_PipeSegments.geojson'

    # Open the zip file
    with zipfile.ZipFile(file_path, 'r') as z:
        # Extract the specific file
        with z.open(geojson_file_path) as geojson_file:
            # Read the geojson file into a GeoDataFrame
            gas_pipelines = gpd.read_file(io.BytesIO(geojson_file.read()))


    param_cols = ["diameter_mm", "max_cap_M_m3_per_d","is_bothDirection","length_km","max_pressure_bar"]
    param = gas_pipelines.param.apply(json.loads).apply(pd.Series)[param_cols]

    method_cols = {
        "diameter_mm": "diameter_method",
        "max_cap_M_m3_per_d": "max_cap_method",
        }

    method = gas_pipelines.method.apply(json.loads).apply(
        pd.Series
    )[[*method_cols]].rename(
        columns= method_cols
    )

    gas_pipelines = pd.concat([gas_pipelines,param,method],axis=1)



    to_drop = ["param", "method", "tags"]
    to_drop = gas_pipelines.columns.intersection(to_drop)
    gas_pipelines.drop(to_drop, axis=1,inplace=True)

    gas_pipelines["start_point"]= gas_pipelines.geometry.apply(lambda x: Point(x.coords[0]))
    gas_pipelines["end_point"]= gas_pipelines.geometry.apply(lambda x: Point(x.coords[-1]))

    return gas_pipelines


def connection_mapping(
        geometry,
        connection_regions,
):

    gdf = gpd.GeoDataFrame(
        geometry = geometry,
        crs = "EPSG:4326",
    )

    connection_map = gpd.sjoin(
        gdf,
        connection_regions,
        how="left",
        predicate="within",
        )

    if "id" in connection_map.columns:
        code_to_return =  connection_map.id
    else:
        code_to_return = connection_map.country_code

    return connection_map.index_right,code_to_return

def read_and_concat_offshore_pipes(
        offshore_path,
        onshore_df,
):
    offshore_df = gpd.read_file(offshore_path).rename(
        columns= dict(StartBus = "start_point",EndBus="end_point")
    )

    offshore_df["pipeline_type"] = "offshore"



    return pd.concat([onshore_df,offshore_df])




def cluster_onshore_pipes(
        grid_path,
        connection_path,
        offshore_path,
):

    gas_pipelines = read_clean_scigrid(grid_path)
    connection_regions = gpd.read_file(connection_path)

    # startpoint and endpoint mapping

    for point in ["start_point","end_point"]:

        bus_map,country_code = connection_mapping(
            gas_pipelines[point],
            connection_regions
        )

        bus_point = point.replace("point","bus")
        gas_pipelines[bus_point] = bus_map
        gas_pipelines[bus_point+"_country"] = country_code
        gas_pipelines[point] = gas_pipelines[bus_point].map(connection_regions.to_crs(3035).centroid.to_crs(4326))


    # drop pipelines outside the regions
    gas_pipelines = gas_pipelines.loc[~gas_pipelines.start_bus.isna() & ~gas_pipelines.end_bus.isna()]


    # drop pipelines in the same region
    gas_pipelines = gas_pipelines.loc[gas_pipelines.start_bus != gas_pipelines.end_bus]


    # recaulcate the pipeline length
    gas_pipelines["geometry"] = gas_pipelines.apply(
    lambda x: LineString([(x["start_point"].x, x["start_point"].y),(x["end_point"].x, x["end_point"].y)]), axis=1
    )

    gas_pipelines["pipeline_type"] = "onshore"

    gas_pipelines.sort_index(axis=1,inplace=True)
    gas_pipelines.drop(["start_point","end_point"],axis=1,inplace=True)


    # gas_pipelines = read_and_concat_offshore_pipes(offshore_path,gas_pipelines)
    gas_pipelines["length_km"] = gas_pipelines.to_crs(3035).geometry.length/1000

    gas_pipelines["energy_cap"] = gas_pipelines.apply(estimate_capacity,axis=1) # MW



    return gas_pipelines


def pipe_sectioning(gas_pipelines):

    large_pipes = gas_pipelines[
        (gas_pipelines.diameter_mm > 950) & (gas_pipelines.max_pressure_bar >= 70)
    ]

    medium_pipes = gas_pipelines[
        (gas_pipelines.diameter_mm > 700) & (gas_pipelines.diameter_mm <= 950) & (gas_pipelines.max_pressure_bar >= 50)
    ]

    small_pipes = gas_pipelines[
        (gas_pipelines.diameter_mm <= 700) & (gas_pipelines.max_pressure_bar >= 50)
    ]

    return {
         "large_pipes (>900 mm)" : large_pipes,
         "medium_pipes (700 - 950 mm)": medium_pipes,
         "small_pipes (<700 mm)": small_pipes,
        }


def estimate_capacity(row):

    if row.max_cap_method in ["raw","Median"]:
        capacity = row.max_cap_M_m3_per_d * NG_DENSITY * NG_ENERGY_CONTENT * 24 * 1000 * MAXIMUM_THEORETICAL_H2_SHARE / 3600

    else:
       #  Based on p.18 of https://ehb.eu/files/downloads/ehb-report-220428-17h00-interactive-1.pdf

        # slopes definitions
        m0 = (1400 - 0) / (500 - 0)
        m1 = (5500 - 1400) / (900 - 500)
        m2 = (15300 - 5500) / (1200- 900)

        # intercept
        a0 = 0
        a1 = -3725
        a2 = -23900

        if row.diameter_mm < 501:
            capacity= a0 + m0 * row.diameter_mm

        elif row.diameter_mm < 901:
            capacity= a1 + m1 * row.diameter_mm

        else:
            capacity= a2 + m2 * row.diameter_mm


    return capacity



def set_index(pipe):

    return f"{pipe.start_bus_country}::{pipe.end_bus_country}"
    # if pipe.start_bus < pipe.end_bus:
    #     return f"{pipe.start_bus_country}::{pipe.end_bus_country}"

    # return f"{pipe.end_bus_country}::{pipe.start_bus_country}"

def set_pipe_directions(gas_pipelines):

    gas_pipelines.index = gas_pipelines.apply(set_index,axis=1)
    gas_pipelines["one_way"] = gas_pipelines.is_bothDirection.apply(
        lambda bi_direction: 0 if bi_direction else 1
    )

    return gas_pipelines.sort_index(axis=1).drop("is_bothDirection",axis=1)


def aggregate_parallel_pipes(gas_pipelines):

    how_to_aggregate = dict(
        end_bus = "first",
        start_bus = "first",
        pipeline_type = "first",
        diameter_method = "first",
        diameter_mm = "mean",
        name = "".join,
        length_km = "mean",
        max_cap_M_m3_per_d= 'max',
        max_cap_method= 'first',
        max_pressure_bar= 'max',
        one_way= 'min',
        energy_cap= 'sum',
        geometry= 'first'
    )

    return gas_pipelines.groupby(gas_pipelines.index).agg(how_to_aggregate)[[*how_to_aggregate]]


def create_sections_clustered_gas_pipes(
        path_to_scigrid,
        path_to_regions,
        path_to_offshore,
        path_to_output
):

    gas_pipelines = cluster_onshore_pipes(
        path_to_scigrid,path_to_regions,path_to_offshore)

    pipes = pipe_sectioning(gas_pipelines)

    for pipe,df in pipes.items():
        df = set_pipe_directions(df)
        pipes[pipe] = aggregate_parallel_pipes(df)


    df = pd.concat(pipes)
    df.index.names = ["type","locs"]
    columns = ["type","locs","diameter_mm","length_km","max_pressure_bar","energy_cap","one_way","geometry"]

    gdf = gpd.GeoDataFrame(df.reset_index()[columns])
    gdf["energy_cap_unit"] = "MW"
    gdf.to_file(path_to_output)



#%%
if __name__ == "__main__":

    create_sections_clustered_gas_pipes(
            path_to_scigrid=snakemake.input.scigrid,
            path_to_regions=snakemake.input.regions,
            path_to_offshore=snakemake.input.offshore_grid,
            path_to_output=snakemake.output[0],
    )
