"""Identify pipeline connections by running a simple greedy algorithm."""

import sys
from typing import TYPE_CHECKING, Any

import _plots
import _schemas
import geopandas as gpd
import numpy as np
import pandas as pd
from cmap import Colormap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

if TYPE_CHECKING:
    snakemake: Any


def _replace_endpnts(
    ls: LineString, start_pnt: Point | None, end_pnt: Point | None
) -> LineString:
    """Replace the first/last vertex of a LineString (optionally).

    Args:
        ls: Input LineString.
        start_pnt: Replacement for the first vertex (or None to keep original).
        end_pnt: Replacement for the last vertex (or None to keep original).

    Returns:
        A new LineString with updated endpoints.
    """
    coords = list(ls.coords)
    if not coords:
        return ls
    if start_pnt is not None:
        coords[0] = (start_pnt.x, start_pnt.y)
    if end_pnt is not None:
        coords[-1] = (end_pnt.x, end_pnt.y)
    return LineString(coords)


def _greedy_xy_clustering(xy: np.ndarray, buffer_distance: float) -> np.ndarray:
    """Greedy heuristical clustering using a KD-tree.

    Clusters points only if their `buffer_distance` is within *every* point in a cluster.
    This avoids "clumping" dense areas into one large point.

    Example:
        If A-B are clustered, and B-C are also clustered, it does
        NOT imply A-C are clustered too.

    Args:
        xy: Array of shape (n, 2) containing point coordinates in a metric CRS (meters).
        buffer_distance: Distance threshold (e.g., metres).

    Returns:
        An integer array of shape (n,) with compact cluster ids 0..K-1.
    """
    n = int(xy.shape[0])
    if n == 0:
        return np.array([], dtype=np.int64)
    if xy.shape[1] != 2:
        raise ValueError("xy must have shape (n, 2)")
    if buffer_distance <= 0:
        raise ValueError("buffer_distance must be > 0")

    # Extreme-case guard: buffer >= bounding-box diagonal
    # every pair of points is within buffer
    dx = float(xy[:, 0].max() - xy[:, 0].min())
    dy = float(xy[:, 1].max() - xy[:, 1].min())
    if buffer_distance >= (dx * dx + dy * dy) ** 0.5:
        return np.zeros(n, dtype=np.int64)

    tree = cKDTree(xy)
    unassigned = np.ones(n, dtype=bool)
    cluster_id = np.full(n, -1, dtype=np.int64)
    cid = 0

    while unassigned.any():
        # Start a new cluster at the first unassigned point.
        seed = int(np.flatnonzero(unassigned)[0])
        unassigned[seed] = False
        members = [seed]

        # Initial candidates: neighbors of the seed
        cand = tree.query_ball_point(xy[seed], r=float(buffer_distance))
        cand = [j for j in cand if j != seed and unassigned[j]]

        sx, sy = xy[seed, 0], xy[seed, 1]
        # Deterministic ordering: (distance^2 to seed, index) for tie-breaks
        cand.sort(key=lambda j: ((xy[j, 0] - sx) ** 2 + (xy[j, 1] - sy) ** 2, int(j)))

        # track points that are still within buffer of *all* chosen members.
        allowed = set(cand)

        for j in cand:
            if not unassigned[j]:
                continue
            if j not in allowed:
                continue

            members.append(j)
            unassigned[j] = False

            # Tighten allowed set: future members must also be within buffer of j
            neigh_j = tree.query_ball_point(xy[j], r=float(buffer_distance))
            allowed.intersection_update(neigh_j)

        cluster_id[members] = cid
        cid += 1

    return cluster_id


def cluster_and_snap_pipelines(
    pipeline_file: str,
    *,
    buffer_distance: float = 100.0,
    projected_crs: str = "EPSG:3035",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Cluster pipeline endpoints into nodes, drop collapsed lines, and snap joint endpoints.

    Algorithm:
      1) Harmonise inputs.
      2) Extract start/end coordinates of each LineString.
      3) Greedily cluster all endpoints.
      4) Each line becomes an edge between its start/end clusters (node ids).
      5) Remove edges where both ends are in the same node (collapsed edges).
      6) Build a representative point per node (mean x/y of member endpoints).
      7) Compute degrees from the remaining edges:
           - joint node: degree >= 2
           - terminal node: degree == 1
      8) Snap ONLY joint endpoints of each edge to the node representative point.
         Terminal endpoints are not moved.

    Args:
        pipeline_file: pipeline file (must fit schema).
        buffer_distance: Clustering distance threshold.
        projected_crs: Projected CRS used for distance computations and snapping.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: pipelines (clustered), nodes
    """
    # -------------------------------------------------------------------------
    # 1) Validate and harmonise input data
    # -------------------------------------------------------------------------
    if buffer_distance <= 0:
        raise ValueError("buffer_m must be > 0")

    # Create stable edge identifiers
    pipes = _schemas.PipelineSchema.validate(gpd.read_parquet(pipeline_file))
    pipes = pipes.set_index("pipeline_id", drop=False)

    # Projected CRS (so buffer is correct)
    original_crs = pipes.crs
    pipes = pipes.to_crs(projected_crs)
    if not pipes.crs.is_projected:
        raise ValueError(f"Provided CRS '{projected_crs}' must be projected.")

    # -------------------------------------------------------------------------
    # 2) Extract endpoints as raw XY arrays
    # -------------------------------------------------------------------------
    start_xy = np.array(
        [(float(ls.coords[0][0]), float(ls.coords[0][1])) for ls in pipes.geometry],
        dtype=float,
    )
    end_xy = np.array(
        [(float(ls.coords[-1][0]), float(ls.coords[-1][1])) for ls in pipes.geometry],
        dtype=float,
    )

    n = len(pipes)
    xy = np.vstack([start_xy, end_xy])  # shape (2N, 2)
    endpts = pd.DataFrame(
        {
            "pipeline_id": np.concatenate(
                [pipes.index.to_numpy(), pipes.index.to_numpy()]
            ),
            "endpoint_type": np.array(["start"] * n + ["end"] * n),
            "x": xy[:, 0],
            "y": xy[:, 1],
        }
    )

    # -------------------------------------------------------------------------
    # 3) Cluster endpoints -> node_id
    # -------------------------------------------------------------------------
    endpts["node_id"] = _greedy_xy_clustering(xy, buffer_distance=buffer_distance)

    # Map each pipeline_id to (start_node_id, end_node_id)
    start_node = endpts.loc[
        endpts["endpoint_type"] == "start", ["pipeline_id", "node_id"]
    ].set_index("pipeline_id")["node_id"]
    end_node = endpts.loc[
        endpts["endpoint_type"] == "end", ["pipeline_id", "node_id"]
    ].set_index("pipeline_id")["node_id"]
    pipe_nodes = pd.DataFrame(
        {"start_node_id": start_node, "end_node_id": end_node}
    ).loc[pipes.index]

    # -------------------------------------------------------------------------
    # 4) Remove collapsed pipes: both endpoints in the same node (start == end)
    # -------------------------------------------------------------------------
    loop_mask = pipe_nodes["start_node_id"] == pipe_nodes["end_node_id"]

    pipe_nodes = pipe_nodes.loc[~loop_mask].copy()
    pipes = pipes.loc[pipe_nodes.index].copy()
    endpts = endpts.loc[endpts["pipeline_id"].isin(pipe_nodes.index)].copy()

    if pipes.empty:
        raise ValueError(
            f"Buffer {buffer_distance!r} collapsed away all pipelines in {pipeline_file!r}."
        )

    # -------------------------------------------------------------------------
    # 5) Build node points (representative point = mean of member endpoint coords)
    # -------------------------------------------------------------------------
    node_xy = endpts.groupby("node_id", as_index=False).agg(
        x=("x", "mean"), y=("y", "mean")
    )
    nodes_m = gpd.GeoDataFrame(
        node_xy,
        geometry=gpd.points_from_xy(node_xy["x"], node_xy["y"]),
        crs=projected_crs,
    )
    nodes_m = nodes_m.drop(columns=["x", "y"])
    nodes_m["node_id"] = nodes_m["node_id"].astype(np.int64)
    nodes_m = nodes_m.set_index("node_id", drop=False)

    # -------------------------------------------------------------------------
    # 6) Degrees per node from remaining edges/pipes
    # -------------------------------------------------------------------------
    # Undirected graph
    deg = pd.concat(
        [pipe_nodes["start_node_id"], pipe_nodes["end_node_id"]]
    ).value_counts()
    nodes_m["degree"] = nodes_m["node_id"].map(deg).fillna(0).astype(int)

    # Directed graph (start -> end for every pipe)
    s = pipe_nodes["start_node_id"].astype("int64")
    t = pipe_nodes["end_node_id"].astype("int64")
    # Bidirectional pipes contribute an extra reverse arc: end -> start
    bi = pipes.loc[pipe_nodes.index, "is_bothDirection"].astype(bool)

    out_deg = s.value_counts()
    out_deg = out_deg.add(t[bi].value_counts(), fill_value=0).astype(int)

    in_deg = t.value_counts()
    in_deg = in_deg.add(s[bi].value_counts(), fill_value=0).astype(int)

    nodes_m["out_degree"] = nodes_m["node_id"].map(out_deg).fillna(0).astype(int)
    nodes_m["in_degree"] = nodes_m["node_id"].map(in_deg).fillna(0).astype(int)

    # Classify nodes (failing if isolated nodes are found)
    in_d = nodes_m["in_degree"]
    out_d = nodes_m["out_degree"]
    nodes_m["node_type"] = np.select(
        [
            (in_d == 0) & (out_d > 0),  # source
            (out_d == 0) & (in_d > 0),  # sink
            (in_d == 1) & (out_d == 1),  # connector
            (in_d > 0) & (out_d > 0) & ((in_d > 1) | (out_d > 1)),  # junction
        ],
        ["source", "sink", "connector", "junction"],
        default="isolated",
    )
    if (nodes_m["node_type"] == "isolated").any():
        bad = nodes_m.loc[
            nodes_m["node_type"] == "isolated",
            ["node_id", "degree", "in_degree", "out_degree"],
        ].head()
        raise RuntimeError(f"Found unexpected isolated nodes:\n{bad}")

    # -------------------------------------------------------------------------
    # 7) Snap ONLY joint endpoints (degree >= 2) to the node representative point
    # -------------------------------------------------------------------------
    node_geom = nodes_m.geometry
    node_deg = nodes_m["degree"]

    new_geoms = []
    for pipeline_id, row in pipe_nodes.iterrows():
        ls = pipes.loc[pipeline_id, "geometry"]
        s = int(row.start_node_id)
        t = int(row.end_node_id)

        # Snap endpoint only if its node is a joint (degree >= 2).
        p0_new = node_geom.loc[s] if int(node_deg.loc[s]) >= 2 else None
        p1_new = node_geom.loc[t] if int(node_deg.loc[t]) >= 2 else None

        new_geoms.append(_replace_endpnts(ls, p0_new, p1_new))

    pipes["geometry"] = new_geoms
    pipes["start_node_id"] = pipe_nodes["start_node_id"].astype(np.int64)
    pipes["end_node_id"] = pipe_nodes["end_node_id"].astype(np.int64)

    # -------------------------------------------------------------------------
    # 8) Reproject back to original CRS
    # -------------------------------------------------------------------------
    edges_out = gpd.GeoDataFrame(
        pipes.to_crs(original_crs), geometry="geometry", crs=original_crs
    )
    nodes_out = nodes_m.to_crs(original_crs)

    return edges_out, nodes_out


def plot(
    pipes_file: str,
    nodes_file: str,
    countries_file: str,
    *,
    projected_crs: str = "EPSG:3857",
) -> tuple[Figure, Axes]:
    """Plot the clustered network using saved node degrees."""
    pipes = gpd.read_parquet(pipes_file).to_crs(projected_crs)
    nodes = gpd.read_parquet(nodes_file).to_crs(projected_crs)
    countries = gpd.read_parquet(countries_file).to_crs(projected_crs)

    xlim, ylim = _plots.get_padded_bounds(pipes, pad_frac=0.05)
    countries = countries.cx[xlim[0] : xlim[1], ylim[0] : ylim[1]]

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")

    # Background
    countries.plot(ax=ax, color="black", alpha=0.2, zorder=-2)
    countries.boundary.plot(ax=ax, color="black", lw=0.5, zorder=-1)

    # Pipes & Nodes
    pipes.plot(ax=ax, linewidth=1, color="tab:blue", zorder=0)
    if not nodes.empty:
        cmap = Colormap("colorbrewer:Accent").to_mpl()
        nodes.plot(
            ax=ax,
            column="node_type",
            cmap=cmap,
            categorical=True,
            legend=True,
            markersize=4,
            zorder=1,
        )

    _plots.style_map_plot(ax, "Summary of snapped pipelines.", xlim, ylim)
    return fig, ax


def main():
    """Main snakemake function."""
    sys.stderr = open(snakemake.log[0], "w")

    pipes_out_file = snakemake.output.pipelines
    nodes_out_file = snakemake.output.nodes
    crs = snakemake.params.projected_crs

    clustered_pipes, clustered_nodes = cluster_and_snap_pipelines(
        snakemake.input.pipelines,
        buffer_distance=snakemake.params.buffer,
        projected_crs=crs,
    )
    _schemas.NodeSchema.validate(clustered_nodes).to_parquet(nodes_out_file)
    _schemas.PipelineSchema.validate(clustered_pipes).to_parquet(pipes_out_file)
    fig, _ = plot(
        pipes_file=pipes_out_file,
        nodes_file=nodes_out_file,
        countries_file=snakemake.input.countries,
        projected_crs=crs,
    )
    fig.savefig(snakemake.output.fig, dpi=300)


if __name__ == "__main__":
    main()
