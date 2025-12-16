"""General utility functions."""
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS


def to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Quick CRS conversion."""
    return gdf.to_crs(crs) if gdf.crs != crs else gdf

def check_projected_crs(crs) -> None:
    if not CRS(crs).is_projected:
        raise ValueError(f"Requested crs must be projected. Got {crs!r}.")


# TODO: improve connection/juntion logic
# connections with two bi-lateral lines are labeled as junctions
def compute_node_attributes(
    pipes: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Identify graph charactersitics (both directed and undirected) per node."""
    u = pipes["start_node_id"]
    v = pipes["end_node_id"]
    deg = pd.concat([u, v]).value_counts()

    both = pipes["is_bidirectional"].astype(bool)
    arcs = pd.concat(
        [
            pd.DataFrame({"src": u, "dst": v}),
            pd.DataFrame({"src": v.loc[both], "dst": u.loc[both]}),
        ],
        ignore_index=True,
    )
    out_deg = arcs["src"].value_counts()
    in_deg = arcs["dst"].value_counts()

    nodes["degree"] = nodes["node_id"].map(deg).fillna(0).astype(int)
    nodes["out_degree"] = nodes["node_id"].map(out_deg).fillna(0).astype(int)
    nodes["in_degree"] = nodes["node_id"].map(in_deg).fillna(0).astype(int)

    if (nodes["degree"] == 0).any():
        raise RuntimeError(
            f"Isolated node(s): {nodes.loc[nodes['degree'] == 0, 'node_id'].tolist()}"
        )

    d = nodes["degree"]
    i = nodes["in_degree"]
    o = nodes["out_degree"]

    nodes["etype"] = np.select(
        [
            (d == 1) & (i == 0) & (o > 0),  # pure directed terminal source
            (d == 1) & (o == 0) & (i > 0),  # pure directed terminal sink
            (d == 1),  # terminal (incl. bidir)
            (d == 2),  # connection (pass-through), regardless of i/o
            (d >= 3),  # junction
        ],
        ["source", "sink", "terminal", "connection", "junction"],
        default="__error__",
    )
    return nodes
