
import geopandas as gpd
import pandas as pd

STORAGE_CAP = "storage_cap"

def estimate_capacity(capacity_per_area, share, area_caverns):

    return capacity_per_area * share * area_caverns / 1000


def get_closest_region(offshore_df,regions):


    offshore_df = offshore_df.geometry.centroid.to_crs(epsg=4326).to_frame()
    regions = regions.to_crs(epsg=4326)

    # to find the nearest region for each offshore point
    nearest = gpd.sjoin_nearest(offshore_df, regions, how="left", distance_col="distance")

    return nearest["id"].tolist()

def aggregate_regions(df,how):

    return df.groupby(["id"]).agg(how)

def calculate_salt_cavern_potential(
        potentials_path,
        regions_path,
        output_path,

):

    potentials = gpd.read_file(potentials_path).to_crs(epsg=3035)
    regions = gpd.read_file(regions_path).to_crs(epsg=3035)

    if "id" not in regions:
        regions["id"] = regions.country_code

    potentials["area_caverns"]= potentials.area.div(1e6) 

    # onshore potentail is filtered by taking the intersection of regions and potentials
    onshore_potential = gpd.overlay(
        regions.reset_index(), 
        potentials, 
        keep_geom_type=True, 
        how='intersection'
        )

    onshore_potential["area_intersect"] = onshore_potential.area.div(1e6) 

    # offshore potentail is filtered by taking the differences of regions and potential
    offshore_potential = gpd.overlay(
        potentials,
        regions.reset_index(), 
        keep_geom_type=True, 
        how='difference'    
    )

    offshore_potential["id"] = get_closest_region(offshore_potential,regions)


    onshore_potential["share"] = onshore_potential["area_intersect"]/ onshore_potential["area_caverns"]
    

    onshore_potential[STORAGE_CAP] = estimate_capacity(
        capacity_per_area = onshore_potential.capacity_per_area,
        share = onshore_potential.share,
        area_caverns=onshore_potential.area_caverns,
    )

    offshore_potential[STORAGE_CAP] = estimate_capacity(
        capacity_per_area = offshore_potential.capacity_per_area,
        share = 1,
        area_caverns=offshore_potential.area_caverns,
    )

    onshore_potential = aggregate_regions(onshore_potential,{"area_intersect": 'sum',"share": 'sum',STORAGE_CAP: 'sum',})
    offshore_potential = aggregate_regions(offshore_potential,{STORAGE_CAP: 'sum',})
   
    df_onshore = regions.merge(onshore_potential,on="id")
    df_offshore = regions.merge(offshore_potential,on="id")

    gdf = gpd.GeoDataFrame(pd.concat({"onshore":df_onshore,"offshore":df_offshore},names=["storage_cluster_type"]))
    gdf[STORAGE_CAP+"_unit"] = "TWh"
    gdf.to_file(output_path)





if __name__ == "__main__":

    calculate_salt_cavern_potential(
        potentials_path = snakemake.input.salt_cavern_potentials,
        regions_path = snakemake.input.regions,
        output_path = snakemake.output[0],
    )
# %%
