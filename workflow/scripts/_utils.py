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


def match_lines_to_polygons_by_length_share(
    lines: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    *,
    polygon_value_col: str = "shape_id",
    threshold: float = 0.5,
    candidate_predicate: str = "intersects",
    keep: str = "max_share",  # "max_share" | "first"
) -> pd.Series:
    """Assign each line a polygon value if enough of the line lies within that polygon.

    Uses a spatial join to generate candidate (line, polygon) pairs, then computes:

        share = length(line ∩ polygon) / length(line)

    and keeps matches where share >= threshold.

    Args:
        lines: GeoDataFrame of LineString geometries. Index must be unique.
        polygons: GeoDataFrame of polygon geometries.
        polygon_value_col: Column on `polygons` to return (e.g., "shape_id").
        threshold: Minimum fraction of line length that must lie within a polygon
            to be considered a match. Must be in [0, 1].
        candidate_predicate: Predicate for the initial candidate search (spatial index),
            usually "intersects".
        keep: How to resolve multiple qualifying polygons for the same line:
            - "max_share": choose the polygon with the largest covered share (default)
            - "first": choose the first qualifying polygon encountered

    Returns:
        Series indexed like `lines`, containing the matched polygon value from
        `polygon_value_col`, or NaN if no polygon reaches the threshold.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1].")

    if not lines.index.is_unique:
        raise ValueError("lines.index must be unique (required for stable assignment).")

    if polygon_value_col not in polygons.columns:
        raise KeyError(f"{polygon_value_col!r} not found in polygons columns.")

    if lines.crs is None or polygons.crs is None:
        raise ValueError("Both GeoDataFrames must have a CRS set.")
    if lines.crs != polygons.crs:
        raise ValueError("CRS must match between lines and polygons.")
    if not lines.crs.is_projected:
        raise ValueError("CRS must be projected for length-based matching.")

    left = lines[["geometry"]]
    right = polygons[[polygon_value_col, "geometry"]]

    # Candidate pairs via spatial index
    cand = gpd.sjoin(
        left,
        right,
        how="inner",
        predicate=candidate_predicate,
    )
    if cand.empty:
        return pd.Series(np.nan, index=lines.index, name=polygon_value_col)

    # Attach polygon geometry (sjoin keeps left geometry; add right geometry for intersections)
    cand = cand.join(right.geometry.rename("_poly_geom"), on="index_right")

    # Lengths
    cand["_line_len"] = left.geometry.length.reindex(cand.index).to_numpy()
    cand["_inter_len"] = cand.geometry.intersection(cand["_poly_geom"]).length
    cand["_share"] = np.where(cand["_line_len"] > 0, cand["_inter_len"] / cand["_line_len"], 0.0)

    # Apply threshold
    cand = cand[cand["_share"] >= threshold]
    if cand.empty:
        return pd.Series(np.nan, index=lines.index, name=polygon_value_col)

    # Resolve multiple matches per line (cand index == left index)
    if keep == "max_share":
        best_idx = cand.groupby(level=0)["_share"].idxmax()
        best = cand.loc[best_idx]
    elif keep == "first":
        best = cand[~cand.index.duplicated(keep="first")]
    else:
        raise ValueError("keep must be 'max_share' or 'first'.")

    out = pd.Series(np.nan, index=lines.index, name=polygon_value_col)
    out.loc[best.index] = best[polygon_value_col].to_numpy()
    return out
