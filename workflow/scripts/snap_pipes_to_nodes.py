"""Snap pipeline endpoints to nearest nodes and derive node graph metrics."""

import _schemas
import _utils
import country_converter as coco
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


def match_pipes_and_nodes(
    pipes: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    buffer_dist: float = 100.0,
    crs: str = "EPSG:3035",
):
    nodes = nodes.reset_index(drop=True)

    pipes = _utils.to_crs(pipes, crs)
    nodes = _utils.to_crs(nodes, crs)
    if not pipes.crs.is_projected:
        raise ValueError(f"Requested crs must be projected. Got {crs!r}.")

    nodes["node_id"] = np.arange(len(nodes), dtype=int)

    # ---- endpoints (2 rows per line)
    start_pts = pipes.geometry.map(lambda ls: Point(ls.coords[0]))
    end_pts = pipes.geometry.map(lambda ls: Point(ls.coords[-1]))

    endpoints = gpd.GeoDataFrame(
        {
            "pipeline_id": pd.Index(pipes["pipeline_id"]).repeat(2).to_numpy(),
            "endpoint": np.tile(["start", "end"], len(pipes)),
        },
        geometry=pd.concat([start_pts, end_pts], ignore_index=True),
        crs=pipes.crs,
    )

    matched = gpd.sjoin_nearest(
        endpoints,
        nodes[["node_id", "geometry"]],
        how="left",
        max_distance=buffer_dist,
        distance_col="snap_dist",
    )

    if matched["node_id"].isna().any():
        bad = matched.loc[matched["node_id"].isna(), ["pipeline_id", "endpoint"]]
        raise RuntimeError(
            f"Unmatched line endpoints within {buffer_dist} units:\n{bad.to_string(index=False)}"
        )

    # Pipeline -> (start_node_id, end_node_id) in one shot
    node_map = matched.pivot(index="pipeline_id", columns="endpoint", values="node_id")
    pipes["start_node_id"] = pipes["pipeline_id"].map(node_map["start"]).astype(int)
    pipes["end_node_id"] = pipes["pipeline_id"].map(node_map["end"]).astype(int)

    # ---- snap geometries (preserve direction: replace only first/last coord)
    node_xy = nodes.set_index("node_id").geometry.map(lambda p: p.coords[0])

    def _snap(ls: LineString, s_id: int, e_id: int) -> LineString:
        coords = list(ls.coords)
        coords[0] = tuple(node_xy.loc[s_id])
        coords[-1] = tuple(node_xy.loc[e_id])
        return LineString(coords)

    pipes["geometry"] = [
        _snap(ls, s, e)
        for ls, s, e in zip(pipes.geometry, pipes.start_node_id, pipes.end_node_id)
    ]

    # ---- graph metrics
    u = pipes["start_node_id"]
    v = pipes["end_node_id"]
    deg = pd.concat([u, v]).value_counts()

    both = pipes["is_bothDirection"].astype(bool)
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

    nodes["e_type"] = np.select(
        [
            (i == 0) & (o > 0),  # source
            (o == 0) & (i > 0),  # sink
            (d == 1) & (i == 1) & (o == 1),  # terminal (single bidirectional pipe)
            (d == 2) & (i == 1) & (o == 1),  # connection (pass-through)
            (i > 0) & (o > 0),  # junction
        ],
        ["source", "sink", "terminal", "connection", "junction"],
        default="__error__",
    )

    return pipes, nodes


def fix_node_country_ids(
    nodes: gpd.GeoDataFrame,
    countries: gpd.GeoDataFrame,
    *,
    country_code_col: str = "country_code",
    country_id_col: str = "country_id",
    id_col: str = "sovereign_id",
    missing: str = "XXX",
):
    nodes = nodes.copy()
    countries = countries[[id_col, "geometry"]].copy()

    if nodes.crs != countries.crs:
        countries = countries.to_crs(nodes.crs)

    uniq = [c for c in nodes[country_code_col].dropna().unique().tolist() if c != "XX"]
    converted = coco.convert(uniq, to="iso3", not_found=np.nan)
    if isinstance(converted, str):  # coco returns scalar for single input sometimes
        converted = [converted]
    tr = dict(zip(uniq, converted))
    tr["XX"] = missing

    nodes[country_id_col] = nodes[country_code_col].map(tr).fillna(missing)

    m = nodes[country_id_col].eq(missing)
    if m.any():
        joined = gpd.sjoin(
            nodes.loc[m, ["geometry"]], countries, predicate="within", how="inner"
        )

        if joined.index.duplicated(keep=False).any():
            bad = joined.index[joined.index.duplicated(keep=False)].unique().tolist()
            raise RuntimeError(f"Ambiguous country match for node index(es): {bad}")

        nodes.loc[joined.index, country_id_col] = joined[id_col].astype(str)

    return nodes
