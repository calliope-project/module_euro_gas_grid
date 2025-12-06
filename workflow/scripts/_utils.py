"""General utility functions."""
import geopandas as gpd
from pyproj import CRS


def to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Quick CRS conversion."""
    return gdf.to_crs(crs) if gdf.crs != crs else gdf

def check_projected_crs(crs) -> None:
    if not CRS(crs).is_projected:
        raise ValueError(f"Requested crs must be projected. Got {crs!r}.")
