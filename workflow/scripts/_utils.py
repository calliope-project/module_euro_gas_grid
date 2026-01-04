"""General utility functions."""

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import Point


def to_crs(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Quick CRS conversion."""
    target = CRS.from_user_input(crs)
    current = CRS.from_user_input(gdf.crs) if gdf.crs is not None else None
    return (
        gdf if (current is not None and current.equals(target)) else gdf.to_crs(target)
    )


def check_projected_crs(crs) -> None:
    if not CRS(crs).is_projected:
        raise ValueError(f"Requested crs must be projected. Got {crs!r}.")

def compute_node_graph_attributes(
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


def match_points_to_polygons(
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    columns: str | Sequence[str],
    *,
    predicate: str = "intersects",
) -> pd.DataFrame:
    """Match each point to polygon attributes, resolving overlaps by smallest polygon area.

    Uses a spatial join to generate candidate (point, polygon) pairs, then if multiple
    polygons match a point (e.g., overlapping polygons), chooses the polygon with the
    smallest area.

    Args:
        points: GeoDataFrame of Point geometries. Index must be unique.
        polygons: GeoDataFrame of polygon geometries.
        columns: Column name(s) from `polygons` to match.
        predicate: Spatial predicate for matching:
            - "intersects": includes points on polygon boundaries
            - "within": point strictly inside polygon (boundary -> no match)

    Returns:
        DataFrame with index matching `points.index` and columns `polygon_columns`.
        Non-matching points receive NA values.
    """
    if not points.index.is_unique:
        raise ValueError("points.index must be unique.")
    check_projected_crs(points.crs)
    if points.crs != polygons.crs:
        raise ValueError("points and polygons must share a CRS.")

    poly_cols = (
        [columns] if isinstance(columns, str) else list(columns)
    )

    output = pd.DataFrame({c: pd.NA for c in poly_cols}, index=points.index)

    # relevant polygon attributes
    polys = polygons[poly_cols + ["geometry"]].copy()
    polys["_poly_area"] = polys.geometry.area

    candidates = gpd.sjoin(
        points[["geometry"]], polys, how="inner", predicate=predicate
    )
    if not candidates.empty:
        # smallest area wins: sort then keep first candidate per point index
        candidates = candidates.sort_values("_poly_area", kind="mergesort")
        best = candidates[~candidates.index.duplicated(keep="first")]

        output.loc[best.index, poly_cols] = best[poly_cols].to_numpy()

    return output


def build_nodes_from_pipelines(pipelines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Construct a unique nodes GeoDataFrame from pipeline endpoints.

    Returns one row per unique `node_id`, with point geometry at the line endpoint.
    If the same `node_id` appears with different coordinates across pipelines,
    raises a ValueError.

    Output columns: ["node_id", "geometry"] with CRS copied from `pipelines`.
    """
    geom_types = pipelines.geometry.geom_type.unique()
    if len(geom_types) != 1 or geom_types[0] != "LineString":
        raise ValueError(f"Only LineStrings are valid. Found {geom_types!r}")

    # extract endpoint geometries
    start_pts = pipelines.geometry.apply(lambda g: Point(g.coords[0]))
    end_pts = pipelines.geometry.apply(lambda g: Point(g.coords[-1]))

    start = gpd.GeoDataFrame(
        {"node_id": pipelines["start_node_id"], "geometry": start_pts},
        crs=pipelines.crs,
    )
    end = gpd.GeoDataFrame(
        {"node_id": pipelines["end_node_id"], "geometry": end_pts}, crs=pipelines.crs
    )

    nodes = pd.concat([start, end], ignore_index=True)
    nodes = nodes.dropna(subset=["node_id"]).copy()
    nodes["node_id"] = nodes["node_id"].astype(int)

    # ensure a node_id never maps to multiple distinct coordinates
    wkb = nodes.geometry.to_wkb()
    conflicts = (
        pd.DataFrame({"node_id": nodes["node_id"].to_numpy(), "wkb": wkb})
        .groupby("node_id")["wkb"]
        .nunique()
    )
    bad = conflicts[conflicts > 1].index.to_list()
    if bad:
        raise ValueError(f"Conflicting endpoint coordinates for node_id(s): {bad}")

    nodes = (
        nodes.sort_values("node_id", kind="mergesort")
        .drop_duplicates("node_id")
        .reset_index(drop=True)
    )
    return gpd.GeoDataFrame(nodes[["node_id", "geometry"]], crs=pipelines.crs)
