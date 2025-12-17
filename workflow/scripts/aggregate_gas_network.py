"""Aggregate gas network to shapes."""

from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import _utils
import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point


@dataclass(frozen=True, order=True)
class Region:
    """Terminal region identifier."""

    kind: Literal["shape", "country"]
    id: str


def _assign_edge_region(
    shape_id, country_id, internal_countries: set[str]
) -> Region | None:
    """Obtain the region of an edge.

    - shape_id -> Region("shape", shape_id)
    - else country_id -> Region("country", ISO3), if external
    - else -> None (pass-through)
    """
    region = None
    if shape_id is not None:
        region = Region("shape", str(shape_id))

    elif country_id is not None:
        cid = str(country_id)
        if cid not in internal_countries:
            region = Region("country", cid)

    return region


def _snap_relevant_nodes(
    pipelines: gpd.GeoDataFrame, relevant_nodes: gpd.GeoDataFrame
) -> set[int]:
    """Snap relevant points to the nearest pipeline connection (endpoint node_id)."""
    start_pts = pipelines.geometry.apply(lambda g: Point(g.coords[0]))
    end_pts = pipelines.geometry.apply(lambda g: Point(g.coords[-1]))

    endpoints = pd.concat(
        [
            gpd.GeoDataFrame(
                {
                    "node_id": pipelines["start_node_id"].values,
                    "geometry": start_pts.values,
                },
                crs=pipelines.crs,
            ),
            gpd.GeoDataFrame(
                {
                    "node_id": pipelines["end_node_id"].values,
                    "geometry": end_pts.values,
                },
                crs=pipelines.crs,
            ),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["node_id"])

    nodes = _utils.to_crs(relevant_nodes, pipelines.crs)
    joined = gpd.sjoin_nearest(
        nodes[["geometry"]], endpoints[["node_id", "geometry"]], how="inner"
    )
    return set(joined["node_id"])


def _build_split_digraph(
    pipelines: gpd.GeoDataFrame, internal_countries: set[str]
) -> nx.DiGraph:
    """Split each pipeline row into u -> mid -> v (and reverse if bidirectional).

    storing 'capacity' and 'region' on edges.

    The intermediate node makes parallel edges safe in DiGraph.
    """
    G = nx.DiGraph()

    # Explicitly track which nodes are "physical" (as opposed to intermediate "pipe" nodes)
    physical_nodes = set(pipelines["start_node_id"]).union(
        set(pipelines["end_node_id"])
    )
    G.graph["physical_nodes"] = physical_nodes

    for row in pipelines.itertuples(index=False):
        pid = row.pipeline_id
        u = row.start_node_id
        v = row.end_node_id
        cap = row.ch4_capacity_mw

        reg = _assign_edge_region(row.shape_id, row.country_id, internal_countries)

        # Create a node in the middle of a pipeline
        # (allows parallel pipelines in the directed graph)
        mid = ("pipe", pid, +1)
        G.add_edge(u, mid, capacity=cap, region=reg)
        G.add_edge(mid, v, capacity=cap, region=reg)

        # Reverse if bidirectional
        # NOTE: this simplification ONLY works when calculating shape-to-shape max flows.
        # Multi-region flow calculations will likely lead to duplicated capacity!
        if bool(row.is_bidirectional):
            mid_r = ("pipe", pid, -1)
            G.add_edge(v, mid_r, capacity=cap, region=reg)
            G.add_edge(mid_r, u, capacity=cap, region=reg)

    return G


def _incident_regions_at_nodes(G: nx.DiGraph) -> dict[int, set[Region]]:
    """Map each physical node to the terminal Regions incident to it."""
    physical_nodes: set[int] = G.graph["physical_nodes"]

    inc: dict[int, set[Region]] = {}
    for u, v, d in G.edges(data=True):
        reg = d.get("region")
        if reg is None:
            continue
        if u in physical_nodes:
            inc.setdefault(u, set()).add(reg)
        if v in physical_nodes:
            inc.setdefault(v, set()).add(reg)
    return inc


def _discover_adjacent_pairs(G: nx.DiGraph) -> set[tuple[Region, Region]]:
    """Find which region pairs to run max-flow for (speedup).

    Candidate pairs:
      direct: a physical node touches >=2 terminal regions
      corridor: a connected component of pass-through edges (region=None) touches >=2 terminal regions
    """
    physical_nodes: set[int] = G.graph["physical_nodes"]

    inc = _incident_regions_at_nodes(G)
    pairs: set[tuple[Region, Region]] = set()

    # direct at node
    for regs in inc.values():
        if len(regs) >= 2:
            for a, b in combinations(sorted(regs), 2):
                pairs.add((a, b))

    # pass-through components (region=None edges)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if d.get("region") is None:
            H.add_edge(u, v)
    for comp in nx.connected_components(H):
        # Find which regions touch unconnected offshore 'blobs'
        touching: set[Region] = set()
        for n in comp:
            if n in physical_nodes and n in inc:
                touching |= inc[n]
        if len(touching) >= 2:
            # sorting prevents duplicates (DEU, NOR) <-> (NOR, DEU)
            for a, b in combinations(sorted(touching), 2):
                pairs.add((a, b))

    return pairs


def _get_region_node_caps(
    G: nx.DiGraph,
) -> tuple[dict[tuple[int, Region], float], dict[tuple[int, Region], float]]:
    """Out/in 'ports' per physical node and terminal region."""
    physical_nodes: set[int] = G.graph["physical_nodes"]

    out_cap: dict[tuple[int, Region], float] = {}
    in_cap: dict[tuple[int, Region], float] = {}

    for u, v, d in G.edges(data=True):
        reg = d.get("region")
        if reg is None:
            continue
        cap = float(d["capacity"])
        # u(out) ---edge--> v(in)
        if u in physical_nodes:
            out_cap[(u, reg)] = out_cap.get((u, reg), 0.0) + cap
        if v in physical_nodes:
            in_cap[(v, reg)] = in_cap.get((v, reg), 0.0) + cap

    return out_cap, in_cap


def _estimate_max_flow(
    G: nx.DiGraph,
    out_cap,
    in_cap,
    ra: Region,
    rb: Region,
    *,
    relevant_nodes: set[int] | None = None,
) -> float:
    """Max flow ra -> rb allowing transit only through {ra, rb, None}-edges."""
    allowed = {None, ra, rb}

    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        if d.get("region") in allowed:
            H.add_edge(u, v, **d)

    s = "__super_source__"
    t = "__super_sink__"
    H.add_node(s)
    H.add_node(t)

    for (n, r), cap in out_cap.items():
        if r == ra and n in H and (relevant_nodes is None or n in relevant_nodes):
            H.add_edge(s, n, capacity=float(cap))
    for (n, r), cap in in_cap.items():
        if r == rb and n in H and (relevant_nodes is None or n in relevant_nodes):
            H.add_edge(n, t, capacity=float(cap))

    flow, _ = nx.maximum_flow(H, s, t, capacity="capacity")
    return float(flow)


def estimate_trade(
    pipelines: gpd.GeoDataFrame,
    internal_countries: set[str],
    *,
    relevant_nodes: gpd.GeoDataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (trade_shapes, trade_external).

    trade_shapes columns:
      shape_id_a, shape_id_b, cap_mw_a_to_b, cap_mw_b_to_a, cap_mw_bidirectional

    trade_external columns (from shape perspective):
      shape_id, country_id, cap_mw_shape_to_country, cap_mw_country_to_shape, cap_mw_bidirectional

    Notes:
      - internal countries are pass-through (never endpoints)
      - country-country pairs are excluded
      - if relevant_nodes is provided, super source/sink attach only at snapped endpoints
    """
    G = _build_split_digraph(pipelines, internal_countries)
    out_cap, in_cap = _get_region_node_caps(G)
    pairs = sorted(_discover_adjacent_pairs(G))

    node_ids = None
    if relevant_nodes is not None:
        node_ids = _snap_relevant_nodes(pipelines, relevant_nodes)

    shape_rows: list[dict] = []
    ext_rows: list[dict] = []

    for ra, rb in pairs:
        # drop country-country
        if ra.kind == "country" and rb.kind == "country":
            continue

        cap_ab = _estimate_max_flow(G, out_cap, in_cap, ra, rb, relevant_nodes=node_ids)
        cap_ba = _estimate_max_flow(G, out_cap, in_cap, rb, ra, relevant_nodes=node_ids)
        cap_bi = min(cap_ab, cap_ba)

        # shape-shape
        if ra.kind == "shape" and rb.kind == "shape":
            shape_rows.append(
                dict(
                    shape_id_a=ra.id,
                    shape_id_b=rb.id,
                    cap_mw_a_to_b=cap_ab,
                    cap_mw_b_to_a=cap_ba,
                    cap_mw_bidirectional=cap_bi,
                )
            )
            continue

        # shape-country, oriented from shape perspective
        if ra.kind == "shape" and rb.kind == "country":
            ext_rows.append(
                dict(
                    shape_id=ra.id,
                    country_id=rb.id,
                    cap_mw_shape_to_country=cap_ab,
                    cap_mw_country_to_shape=cap_ba,
                    cap_mw_bidirectional=cap_bi,
                )
            )
        # Inverted case
        elif ra.kind == "country" and rb.kind == "shape":
            ext_rows.append(
                dict(
                    shape_id=rb.id,
                    country_id=ra.id,
                    cap_mw_shape_to_country=cap_ba,
                    cap_mw_country_to_shape=cap_ab,
                    cap_mw_bidirectional=cap_bi,
                )
            )

    trade_shapes = pd.DataFrame(shape_rows)
    trade_external = pd.DataFrame(ext_rows)

    if not trade_shapes.empty:
        trade_shapes = trade_shapes.sort_values(
            ["cap_mw_bidirectional", "shape_id_a", "shape_id_b"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    if not trade_external.empty:
        trade_external = trade_external.sort_values(
            ["cap_mw_bidirectional", "shape_id", "country_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    return trade_shapes, trade_external


def trade_to_centroid_lines(
    trade_shapes: pd.DataFrame,
    shapes: gpd.GeoDataFrame,
    *,
    shape_id_col: str = "shape_id",
) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of LineStrings between shape centroids, carrying trade capacities."""
    # Map shape_id -> centroid point
    shape_geom = shapes.set_index(shape_id_col)["geometry"]
    centroids = shape_geom.centroid

    a_pts = trade_shapes["shape_id_a"].map(centroids)
    b_pts = trade_shapes["shape_id_b"].map(centroids)

    lines = [LineString([a, b]) for a, b in zip(a_pts, b_pts)]

    out = trade_shapes.copy()
    return gpd.GeoDataFrame(out, geometry=lines, crs=shapes.crs)
