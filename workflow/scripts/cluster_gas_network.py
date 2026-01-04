import itertools
import re
import sys
from typing import TYPE_CHECKING, Any

import _plots
import _schemas
import _utils
import geopandas as gpd
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Point

if TYPE_CHECKING:
    snakemake: Any

HUB_RE = re.compile(r"^hub_(\d+)$")

KIND_RANK = {"shape": 0, "hub": 1, "outside": 2}
UNKNOWN_KIND = "other"
UNKNOWN_RANK = 99

INF_CAPACITY = 1e18  # used for auxiliary source/sink construction


def _centroid_of_points(points: list[Point], crs) -> Point:
    """Centroid of a list of points."""
    return gpd.GeoSeries(points, crs=crs).union_all().centroid


def assign_terminals_to_nodes(
    nodes: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    *,
    shape_id_col: str = "shape_id",
    shape_country_col: str = "country_id",
    sovereign_col: str = "sovereign_id",
) -> pd.DataFrame:
    """Classify each node_id as a terminal.

    In order of priority:
      - shape terminal   -> shape_id exists
      - outside terminal -> sovereign_id exists and not in shapes.country_id
      - intermediary     -> neither of the above

    Returns a DataFrame indexed by node_id with columns:
      - term_kind: {"shape","outside"} or NA
      - term_id: identifier value (shape_id or sovereign_id) or NA
    """
    shape_country_ids = set(shapes[shape_country_col].dropna().astype(str).unique())

    terms = nodes[["node_id", shape_id_col, sovereign_col]].copy()
    terms["term_kind"] = pd.NA
    terms["term_id"] = pd.NA

    in_shape = terms[shape_id_col].notna()
    terms.loc[in_shape, "term_kind"] = "shape"
    terms.loc[in_shape, "term_id"] = terms.loc[in_shape, shape_id_col]

    outside = (
        terms[shape_id_col].isna()
        & terms[sovereign_col].notna()
        & ~terms[sovereign_col].astype(str).isin(shape_country_ids)
    )
    terms.loc[outside, "term_kind"] = "outside"
    terms.loc[outside, "term_id"] = terms.loc[outside, sovereign_col]

    return terms.set_index("node_id")


def aggregate_terminals_to_points(
    nodes: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    term: pd.DataFrame,
    *,
    shape_id_col: str = "shape_id",
) -> dict[tuple[str, str], Point]:
    """Build a Point per terminal key.

    Two approaches:
    - shapes: polygon centroid
    - outside: centroid of node points assigned to that outside terminal
    """
    pts: dict[tuple[str, str], Point] = {}

    # shapes: always polygon centroids
    for shape_id, geom in shapes.set_index(shape_id_col).geometry.items():
        pts[("shape", str(shape_id))] = geom.centroid

    # outside: centroid of assigned node points
    out = term.loc[term["term_kind"] == "outside", "term_id"].dropna().astype(str)
    if not out.empty:
        tmp = nodes.set_index("node_id").geometry.loc[out.index]
        df = pd.DataFrame({"tid": out.to_numpy(), "geom": tmp.to_list()})
        for tid, g in df.groupby("tid")["geom"]:
            pts[("outside", str(tid))] = _centroid_of_points(g.to_list(), nodes.crs)

    return pts


def build_capacity_digraph(
    pipelines: pd.DataFrame,
    *,
    cap_col: str = "capacity_mw",
    bidir_col: str = "is_bidirectional",
) -> nx.DiGraph:
    """Directed capacity graph on node_id integers.

    - start_node_id -> end_node_id always.
    - if is_bidirectional==True, also include reverse arc with same capacity.
    - parallel arcs are summed.
    """
    df = pipelines.dropna(subset=["start_node_id", "end_node_id", cap_col]).copy()
    df["u"] = df["start_node_id"].astype(int)
    df["v"] = df["end_node_id"].astype(int)
    df["cap"] = df[cap_col].astype(float)
    df["bidir"] = df[bidir_col].astype(bool)

    fwd = df[["u", "v", "cap"]]
    rev = df.loc[df["bidir"], ["v", "u", "cap"]].rename(columns={"v": "u", "u": "v"})
    arcs = (
        pd.concat([fwd, rev], ignore_index=True)
        .groupby(["u", "v"], as_index=False)["cap"]
        .sum()
        .query("u != v and cap > 0")
        .rename(columns={"cap": "capacity"})
    )

    return nx.from_pandas_edgelist(
        arcs, "u", "v", edge_attr="capacity", create_using=nx.DiGraph()
    )


def find_intermediary_components(
    G: nx.DiGraph, inter_nodes: set[int]
) -> list[set[int]]:
    """Undirected connected components of intermediary-only nodes.

    These are 'corridors' between terminals (i.e., nodes not in shapes or countries).
    Returns sets of nodes that form part of an intermediary component.
    """
    U = nx.Graph()
    U.add_nodes_from(inter_nodes)
    for u, v in G.edges:
        if u in inter_nodes and v in inter_nodes:
            U.add_edge(u, v)
    return [set(c) for c in nx.connected_components(U)]


def build_corridor_subgraph(
    G: nx.DiGraph, components: set[int], boundary_nodes: set[int]
) -> nx.DiGraph:
    """Subgraph induced by (union of component and boundary terminal-nodes).

    Removes terminal<->terminal arcs to prevent double counting with direct terminal links.
    """
    H = G.subgraph(set(components) | set(boundary_nodes)).copy()
    for u, v in list(H.edges):
        if u in boundary_nodes and v in boundary_nodes:
            H.remove_edge(u, v)
    return H


def max_transfer(G: nx.DiGraph, source, sink) -> float:
    """Max transferable capacity s->t (max-flow == min-cut)."""
    if source not in G or sink not in G or source == sink:
        return 0.0
    return float(
        nx.maximum_flow_value(
            G,
            source,
            sink,
            capacity="capacity",
            flow_func=nx.algorithms.flow.preflow_push,
        )
    )


def max_transfer_sets(G: nx.DiGraph, sources: set[int], sinks: set[int]) -> float:
    """Max flow from a set of source nodes S to a *set* of sink nodes T.

    Implemented by adding a super-source -> S and T -> super-sink with INF capacity.
    """
    if not sources or not sinks or sources == sinks:
        return 0.0
    source, sink = object(), object()
    H = G.copy()
    H.add_node(source)
    H.add_node(sink)
    for s in sources:
        H.add_edge(source, s, capacity=INF_CAPACITY)
    for s in sinks:
        H.add_edge(s, sink, capacity=INF_CAPACITY)
    return max_transfer(H, source, sink)


def build_trade_network_with_hubs(
    nodes: gpd.GeoDataFrame, pipelines: pd.DataFrame, shapes: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """Aggregate pipelines into a shape-centric trade network with corridor hubs.

    Builds a directed capacity graph from `pipelines`, assigns pipeline endpoints to:
    shapes (by shape_id), outside terminals (by sovereign_id), or intermediary nodes.
    Intermediary components connecting 3+ terminals are collapsed into hub nodes.

    Args:
        nodes: Point GeoDataFrame with at least ["node_id","geometry","shape_id","sovereign_id"].
        pipelines: Table with at least ["start_node_id","end_node_id","capacity_mw","is_bidirectional"].
        shapes: Polygon GeoDataFrame with at least ["shape_id","country_id","geometry"].

    Returns:
        tuple: aggregation summary
        - agg_nodes: aggregated nodes w/ ["loc_id","kind","geometry"].
        - agg_pipelines: directed pipelines w/ ["src","dst","capacity_mw","link_type","corridor_id","geometry"].
        - hubs: Point GeoDataFrame with columns
            ["hub_id","corridor_id","n_inter_nodes","n_terminals","throughput_mw","geometry"].
        - hub_membership: DataFrame with columns ["hub_id","corridor_id","terminal_id","terminal_kind"].

    Raises:
        ValueError: If terminal identifiers would collide or required inputs are invalid.
    """
    nodes_t = assign_terminals_to_nodes(nodes, shapes)
    term_pts = aggregate_terminals_to_points(nodes, shapes, nodes_t)

    # Terminal keys are explicit (kind, id_as_str). Graph nodes remain node_id:int.
    def term_key(n: int) -> tuple[str, str] | None:
        k = nodes_t.at[n, "term_kind"]
        if pd.isna(k):
            return None
        return (str(k), str(nodes_t.at[n, "term_id"]))

    shapes_set = {("shape", sid) for (k, sid) in term_pts if k == "shape"}
    outside_set = {("outside", oid) for (k, oid) in term_pts if k == "outside"}

    # Guard against shape_id == outside sovereign_id collisions in loc_id strings.
    shape_ids = {sid for (_k, sid) in shapes_set}
    outside_ids = {oid for (_k, oid) in outside_set}
    collide = shape_ids & outside_ids
    if collide:
        raise ValueError(
            f"Terminal id collision between shape_id and sovereign_id: {sorted(collide)[:10]} "
            "(would create duplicate loc_id strings). Consider disambiguating identifiers."
        )

    G = build_capacity_digraph(pipelines)
    nodes_idx = nodes.set_index("node_id")

    link_rows: list[tuple[str, str, float, str, object]] = []
    hub_rows: list[tuple[str, int, int, int, float, Point]] = []
    hub_membership_rows: list[tuple[str, int, str, str]] = []

    # Direct terminal->terminal links (no flow needed)
    for u, v, data in G.edges(data=True):
        tu, tv = term_key(u), term_key(v)
        if tu is None or tv is None or tu == tv:
            continue
        cap = float(data["capacity"])
        if cap > 0:
            link_rows.append((tu[1], tv[1], cap, "direct", pd.NA))

    # Corridor components (intermediary-only)
    inter_nodes = set(nodes_t.index[nodes_t["term_kind"].isna()].astype(int))
    for corridor_id, comp in enumerate(find_intermediary_components(G, inter_nodes)):
        # touched terminals as keys + boundary node_ids per terminal
        touched: set[tuple[str, str]] = set()
        boundary: dict[tuple[str, str], set[int]] = {}

        for n in comp:
            for m in itertools.chain(G.predecessors(n), G.successors(n)):
                tk = term_key(m)
                if tk is None:
                    continue
                touched.add(tk)
                boundary.setdefault(tk, set()).add(int(m))

        if len(touched) < 2:
            continue  # skip cases with less than two touched terminals
        touched_shapes = [t for t in touched if t[0] == "shape"]
        if not touched_shapes:
            continue  # skip components unrelated to requested shapes
        touched_outside = [t for t in touched if t[0] == "outside"]
        if len(touched_shapes) == 1 and not touched_outside:
            continue  # skip dead-end offshore plumbing

        boundary_nodes = set().union(*boundary.values())
        H = build_corridor_subgraph(G, comp, boundary_nodes)
        touched_list = sorted(touched)

        # Two terminals: corridor pair links via set-to-set max flow.
        if len(touched_list) == 2:
            a, b = touched_list
            cap_ab = max_transfer_sets(H, boundary[a], boundary[b])
            cap_ba = max_transfer_sets(H, boundary[b], boundary[a])
            if cap_ab > 0:
                link_rows.append((a[1], b[1], cap_ab, "corridor_pair", corridor_id))
            if cap_ba > 0:
                link_rows.append((b[1], a[1], cap_ba, "corridor_pair", corridor_id))
            continue

        # 3+ terminals: hub collapse.
        hub_id = f"hub_{corridor_id}"
        hub_pt = _centroid_of_points(
            nodes_idx.loc[list(comp), "geometry"].to_list(), nodes.crs
        )

        # Interface caps: terminal -> hub (send to rest), hub -> terminal (receive from rest).
        throughput_best = 0.0
        for t in touched_list:
            others = [o for o in touched_list if o != t]
            rest = set().union(*(boundary[o] for o in others))

            out_cap = max_transfer_sets(H, boundary[t], rest)
            in_cap = max_transfer_sets(H, rest, boundary[t])

            if out_cap > 0:
                link_rows.append((t[1], hub_id, out_cap, "hub_interface", corridor_id))
            if in_cap > 0:
                link_rows.append((hub_id, t[1], in_cap, "hub_interface", corridor_id))

            throughput_best = max(throughput_best, out_cap)
            hub_membership_rows.append((hub_id, corridor_id, t[1], t[0]))

        hub_rows.append(
            (hub_id, corridor_id, len(comp), len(touched_list), throughput_best, hub_pt)
        )

    # Aggregate links (keeping NA corridor_id for direct links)
    links = pd.DataFrame(
        link_rows, columns=["src", "dst", "capacity_mw", "link_type", "corridor_id"]
    )
    links = links.groupby(
        ["src", "dst", "link_type", "corridor_id"], as_index=False, dropna=False
    )["capacity_mw"].sum()

    # Shape-centric filtering
    hub_ids = {hid for hid, *_ in hub_rows}
    hubs_connected_to_shapes = set()
    for s, d in links[["src", "dst"]].itertuples(index=False):
        if s in hub_ids and d in shape_ids:
            hubs_connected_to_shapes.add(s)
        if d in hub_ids and s in shape_ids:
            hubs_connected_to_shapes.add(d)

    keep = (
        links["src"].isin(shape_ids)
        | links["dst"].isin(shape_ids)
        | links["src"].isin(hubs_connected_to_shapes)
        | links["dst"].isin(hubs_connected_to_shapes)
    )
    links = links.loc[keep].copy()
    links = links.loc[
        ~(links["src"].isin(outside_ids) & links["dst"].isin(outside_ids))
    ].copy()

    outside_keep = (set(links["src"]) | set(links["dst"])) & outside_ids

    # Aggregated nodes
    agg_node_rows = [
        (sid, "shape", term_pts[("shape", sid)]) for sid in sorted(shape_ids)
    ]
    agg_node_rows += [
        (oid, "outside", term_pts[("outside", oid)]) for oid in sorted(outside_keep)
    ]
    agg_node_rows += [
        (hid, "hub", geom)
        for hid, *_rest, geom in hub_rows
        if hid in hubs_connected_to_shapes
    ]

    agg_nodes = gpd.GeoDataFrame(
        agg_node_rows,
        columns=["loc_id", "kind", "geometry"],
        geometry="geometry",
        crs=nodes.crs,
    )

    # Link geometry (LineStrings)
    pt_map = agg_nodes.set_index("loc_id").geometry.to_dict()
    geoms, ok = [], []
    for s, d in links[["src", "dst"]].itertuples(index=False):
        ps, pd_ = pt_map.get(s), pt_map.get(d)
        ok.append(ps is not None and pd_ is not None)
        geoms.append(LineString([ps, pd_]) if ok[-1] else None)

    agg_pipelines = gpd.GeoDataFrame(
        links.loc[ok].copy(),
        geometry=[g for g, keep in zip(geoms, ok) if keep],
        crs=nodes.crs,
    )

    # Hub tables
    hubs = (
        gpd.GeoDataFrame(
            hub_rows,
            columns=[
                "hub_id",
                "corridor_id",
                "n_inter_nodes",
                "n_terminals",
                "throughput_best_mw",
                "geometry",
            ],
            geometry="geometry",
            crs=nodes.crs,
        )
        .loc[lambda df: df["hub_id"].isin(hubs_connected_to_shapes)]
        .copy()
    )

    hub_membership = (
        pd.DataFrame(
            hub_membership_rows,
            columns=["hub_id", "corridor_id", "terminal_id", "terminal_kind"],
        )
        .loc[lambda df: df["hub_id"].isin(hubs_connected_to_shapes)]
        .copy()
    )

    # Stable ordering
    agg_nodes = agg_nodes.sort_values(["kind", "loc_id"], kind="mergesort").reset_index(
        drop=True
    )
    agg_pipelines = agg_pipelines.sort_values(
        ["link_type", "src", "dst"], kind="mergesort"
    ).reset_index(drop=True)
    hubs = hubs.sort_values(["corridor_id"], kind="mergesort").reset_index(drop=True)
    hub_membership = hub_membership.sort_values(
        ["corridor_id", "terminal_id"], kind="mergesort"
    ).reset_index(drop=True)

    return agg_nodes, agg_pipelines, hubs, hub_membership


def condense_agg_pipeline_pairs(
    agg_nodes: gpd.GeoDataFrame, agg_pipelines: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Create an unordered-pair view of directed pipelines.

    - one row per unordered pair (a,b)
    - directional capacities side-by-side (capacity_a_to_b_mw, capacity_b_to_a_mw)
    - includes uni-directional and bi-directional cases
    """
    kind_map = agg_nodes.set_index("loc_id")["kind"].to_dict()
    pt_map = agg_nodes.set_index("loc_id").geometry.to_dict()

    def sort_key(x: str):
        k = kind_map.get(x, UNKNOWN_KIND)
        rank = KIND_RANK.get(k, UNKNOWN_RANK)
        if k == "hub":
            m = HUB_RE.match(x)
            return (rank, int(m.group(1)) if m else x)
        return (rank, x)

    d = (
        agg_pipelines.groupby(["src", "dst"], as_index=False, dropna=False)[
            "capacity_mw"
        ]
        .sum()
        .rename(columns={"capacity_mw": "cap"})
    )

    rows: dict[tuple[str, str], tuple[float, float]] = {}
    for src, dst, cap in d.itertuples(index=False):
        a, b = sorted((src, dst), key=sort_key)
        ab, ba = rows.get((a, b), (0.0, 0.0))

        if src == a and dst == b:
            ab += float(cap)
        else:
            ba += float(cap)

        rows[(a, b)] = (ab, ba)

    out = pd.DataFrame(
        [
            {"a": a, "b": b, "capacity_a_to_b_mw": ab, "capacity_b_to_a_mw": ba}
            for (a, b), (ab, ba) in rows.items()
        ]
    )

    out["a_kind"] = out["a"].map(kind_map)
    out["b_kind"] = out["b"].map(kind_map)

    out["has_a_to_b"] = out["capacity_a_to_b_mw"] > 0
    out["has_b_to_a"] = out["capacity_b_to_a_mw"] > 0
    out["is_bidirectional"] = out["has_a_to_b"] & out["has_b_to_a"]

    out["min_capacity_mw"] = out[["capacity_a_to_b_mw", "capacity_b_to_a_mw"]].min(axis="columns")
    out["max_capacity_mw"] = out[["capacity_a_to_b_mw", "capacity_b_to_a_mw"]].max(axis="columns")

    geoms = []
    for a, b in out[["a", "b"]].itertuples(index=False):
        pa, pb = pt_map.get(a), pt_map.get(b)
        geoms.append(
            LineString([pa, pb]) if (pa is not None and pb is not None) else None
        )

    pairs = gpd.GeoDataFrame(out, geometry=geoms, crs=agg_nodes.crs).dropna(
        subset=["geometry"]
    )

    return pairs.sort_values(
        ["a_kind", "a", "b_kind", "b"], kind="mergesort"
    ).reset_index(drop=True)


def plot(
    agg_nodes: gpd.GeoDataFrame, agg_pipes: gpd.GeoDataFrame, shapes: gpd.GeoDataFrame
):
    """Simple plot of resulting aggregated system."""
    fig, ax = plt.subplots(figsize=(6, 6), layout="compressed")
    agg_pipes.plot("max_capacity_mw", lw=1.5, legend=True, ax=ax)
    agg_nodes.plot("kind", ax=ax, markersize=5, zorder=2, legend=True)
    shapes.boundary.plot(ax=ax, color="black", lw=0.5)
    _plots.style_map_plot(ax, "Aggregated max directed capacity ($MW$)")
    return fig, ax


def main():
    """Main snakemake process."""
    proj_crs = snakemake.params.projected_crs
    _utils.check_projected_crs(proj_crs)

    # Prepare inputs
    nodes = _utils.to_crs(gpd.read_parquet(snakemake.input.nodes), proj_crs)
    pipelines = _utils.to_crs(gpd.read_parquet(snakemake.input.pipelines), proj_crs)
    shapes = _utils.to_crs(gpd.read_parquet(snakemake.input.shapes), proj_crs)
    shapes = _schemas.ShapesSchema.validate(shapes)

    # Prepare nodes dataset
    nodes = nodes.join(_utils.match_points_to_polygons(nodes, shapes, "shape_id"))
    replace_sovereign = snakemake.params.replace_sovereign
    if replace_sovereign:  # Swap sovereign IDs if requested
        nodes["sovereign_id"] = nodes["sovereign_id"].replace(replace_sovereign)

    # Produce aggregated dataset
    agg_nodes, agg_pipes, hubs, _ = build_trade_network_with_hubs(
        nodes, pipelines, shapes
    )
    agg_pipes = condense_agg_pipeline_pairs(agg_nodes, agg_pipes)

    # Plotting
    fig, _ = plot(agg_nodes, agg_pipes, shapes)

    # Output saving
    agg_nodes.to_parquet(snakemake.output.nodes)
    agg_pipes.to_parquet(snakemake.output.pipelines)
    hubs.to_parquet(snakemake.output.hubs)
    fig.savefig(snakemake.output.fig, dpi=300)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")
    main()
