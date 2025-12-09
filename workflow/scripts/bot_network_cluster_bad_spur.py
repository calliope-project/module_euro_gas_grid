from __future__ import annotations

from itertools import combinations
from collections import deque

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx


OUT_LABEL = "OUT"
OFFSHORE_CODE = "XXX"  # kept for your wider codebase; not used for is_out anymore
ANCHOR_ETYPES = {"source", "sink", "terminal", "junction"}

MAX_BFS_NODES = 50_000  # cap on BFS expansion per (entry_node, shape) when looking for "anchor"/"interior" evidence


def _label_nodes_to_shapes(
    nodes: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    metric_crs=3035,
    include_boundary: bool = True,
) -> gpd.GeoDataFrame:
    if nodes.crs != shapes.crs:
        shapes = shapes.to_crs(nodes.crs)

    shapes = shapes.copy()
    shapes["_area"] = shapes.to_crs(metric_crs).geometry.area  # overlap tie-break (enclaves win)

    pred = "intersects" if include_boundary else "within"
    j = (
        gpd.sjoin(
            nodes[["node_id", "geometry"]],
            shapes[["shape_id", "_area", "geometry"]],
            how="left",
            predicate=pred,
        )
        .sort_values("_area", kind="stable")
    )

    best = j.drop_duplicates("node_id", keep="first").set_index("node_id")["shape_id"]

    out = nodes.copy()
    out["shape_id"] = out["node_id"].map(best).fillna(OUT_LABEL)
    return out


def _build_graph(nodes_labeled: pd.DataFrame, pipelines: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for r in nodes_labeled[["node_id", "shape_id", "etype", "country_id"]].itertuples(index=False):
        G.add_node(int(r.node_id), shape_id=str(r.shape_id), etype=r.etype, country_id=r.country_id)

    for r in pipelines[["start_node_id", "end_node_id"]].dropna().itertuples(index=False):
        G.add_edge(int(r.start_node_id), int(r.end_node_id))

    return G


def _absorb_single_touch_out_components(G: nx.Graph, absorb_single_touch_out: bool = True):
    # IMPORTANT: OUT is defined ONLY by shape assignment (shape file is the authority)
    def is_out(n: int) -> bool:
        return G.nodes[n]["shape_id"] == OUT_LABEL

    out_nodes = [n for n in G.nodes if is_out(n)]
    out_sub = G.subgraph(out_nodes)

    out_info = {}
    node_to_comp = {}

    for cid, comp in enumerate(nx.connected_components(out_sub)):
        comp = set(comp)
        for n in comp:
            node_to_comp[n] = cid

        touched = set()
        for n in comp:
            for nbr in G.neighbors(n):
                sid = G.nodes[nbr]["shape_id"]
                if sid != OUT_LABEL:
                    touched.add(str(sid))
        touched = sorted(touched)

        info = {"kind": "isolated_out", "touched_shapes": touched, "absorbed_to": None}

        if len(touched) >= 2:
            info["kind"] = "corridor"
        elif len(touched) == 1:
            if absorb_single_touch_out:
                info["kind"] = "absorbed_spur"
                info["absorbed_to"] = touched[0]
                for n in comp:
                    G.nodes[n]["shape_id"] = touched[0]
            else:
                info["kind"] = "single_touch_out"

        out_info[cid] = info

    return G, out_info, node_to_comp


def _shape_sets(G: nx.Graph):
    sets = {}
    for n, d in G.nodes(data=True):
        sid = str(d["shape_id"])
        S = sets.setdefault(sid, {"nodes": set(), "anchors": set(), "boundary": set(), "interior": set()})
        S["nodes"].add(n)
        if d["etype"] in ANCHOR_ETYPES:
            S["anchors"].add(n)

    for sid, S in sets.items():
        for n in S["nodes"]:
            if any(str(G.nodes[v]["shape_id"]) != sid for v in G.neighbors(n)):
                S["boundary"].add(n)
        S["interior"] = S["nodes"] - S["boundary"]

    return sets


def _evidence(G: nx.Graph, entry: int, sid: str, sets):
    """Return 'anchor' | 'interior' | 'boundary_only' | None, staying within sid."""
    S = sets.get(str(sid))
    if entry in S["anchors"]:
        return "anchor"
    if entry in S["interior"]:
        return "interior"

    q = deque([entry])
    seen = {entry}
    moved = False

    while q and len(seen) < MAX_BFS_NODES:
        u = q.popleft()
        if u in S["anchors"]:
            return "anchor"
        if u in S["interior"]:
            return "interior"

        for v in G.neighbors(u):
            if v in seen:
                continue
            if str(G.nodes[v]["shape_id"]) != str(sid):
                continue
            seen.add(v)
            q.append(v)
            moved = True

    return "boundary_only" if moved else None


def classify_pipeline_trade(
    shapes: gpd.GeoDataFrame,
    pipelines: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    metric_crs=3035,
    include_boundary: bool = True,
    absorb_single_touch_out: bool = True,
    min_land_inshape_m: float = 0.0,
    min_marine_inshape_m: float = 0.0,
    marine_col: str = "is_marine",
):
    """
    Adds:
      - start_shape, end_shape
      - trade_class, trade_reason
      - out_component_id, out_component_kind

    Spur suppression (optional):
      If min_*_inshape_m > 0, crossings into a shape require
      len( line ∩ that_shape ) >= threshold (land vs marine).
      Shapes must provide boolean `marine_col` (e.g. is_marine).
    """
    # 1) label nodes to user shapes
    nodes_labeled = _label_nodes_to_shapes(nodes, shapes, metric_crs=metric_crs, include_boundary=include_boundary)

    # 2) topology graph + OUT component handling
    G = _build_graph(nodes_labeled, pipelines)
    G, out_info, node_to_comp = _absorb_single_touch_out_components(G, absorb_single_touch_out=absorb_single_touch_out)

    nodes_labeled = nodes_labeled.copy()
    nodes_labeled["shape_id"] = nodes_labeled["node_id"].map(lambda n: G.nodes[int(n)]["shape_id"])

    # 3) evidence sets
    sets = _shape_sets(G)
    ev_cache = {}

    def ev(n, sid):
        key = (int(n), str(sid))
        if key not in ev_cache:
            ev_cache[key] = _evidence(G, int(n), str(sid), sets)
        return ev_cache[key]

    # 4) pipeline endpoint shapes
    pipes_out = pipelines.copy()
    pipes_out["start_shape"] = pipes_out["start_node_id"].map(lambda n: G.nodes[int(n)]["shape_id"])
    pipes_out["end_shape"]   = pipes_out["end_node_id"].map(lambda n: G.nodes[int(n)]["shape_id"])

    # OUT component metadata (FIXED: take component from whichever endpoint is OUT)
    out_start = pipes_out["start_node_id"].map(lambda n: node_to_comp.get(int(n), np.nan))
    out_end   = pipes_out["end_node_id"].map(lambda n: node_to_comp.get(int(n), np.nan))
    is_start_out = pipes_out["start_shape"] == OUT_LABEL
    is_end_out   = pipes_out["end_shape"] == OUT_LABEL

    pipes_out["out_component_id"] = np.where(
        is_start_out, out_start,
        np.where(is_end_out, out_end, np.nan)
    )
    kind_map = {cid: info["kind"] for cid, info in out_info.items()}
    pipes_out["out_component_kind"] = pipes_out["out_component_id"].map(
        lambda cid: None if pd.isna(cid) else kind_map[int(cid)]
    )

    # 5) optional: min in-shape length thresholds (land vs maritime)
    use_len_thresholds = (min_land_inshape_m > 0) or (min_marine_inshape_m > 0)
    if use_len_thresholds:
        shapes_m = shapes.to_crs(metric_crs).set_index("shape_id")
        shape_geom = shapes_m.geometry.to_dict()
        shape_class = shapes.set_index("shape_id")["shape_class"].to_dict()  # "land" or "maritime"

        pipes_m = pipes_out.to_crs(metric_crs)
        len_cache = {}  # (pipe_index, shape_id) -> length_m

        def thr(sid: str) -> float:
            return min_marine_inshape_m if shape_class.get(sid) == "maritime" else min_land_inshape_m

        def inshape_len(idx, sid: str) -> float:
            key = (idx, sid)
            if key not in len_cache:
                len_cache[key] = pipes_m.loc[idx, "geometry"].intersection(shape_geom[sid]).length
            return len_cache[key]

    strong = {"anchor", "interior"}
    cls, why = [], []

    for idx, r in pipes_out[["start_node_id","end_node_id","start_shape","end_shape","out_component_kind"]].iterrows():
        u, v = int(r.start_node_id), int(r.end_node_id)
        su, sv = str(r.start_shape), str(r.end_shape)

        if su == sv:
            cls.append("internal"); why.append("same_shape"); continue

        if (su == OUT_LABEL) or (sv == OUT_LABEL):
            if r.out_component_kind == "corridor":
                cls.append("out_corridor_link"); why.append("touches_out_corridor")
            else:
                cls.append("out"); why.append("touches_out")
            continue

        eu, evv = ev(u, su), ev(v, sv)

        # base classification from graph evidence
        if (eu in strong) and (evv in strong):
            base = "trade"
            base_why = f"evidence:{eu}+{evv}"
        elif (eu is not None) or (evv is not None):
            base = "transit"
            base_why = f"weak:{eu}+{evv}"
        else:
            base = "border_artifact"
            base_why = "no_in_shape_evidence"

        # spur suppression by in-shape intersection length (optional)
        if use_len_thresholds and base in {"trade", "transit"}:
            tsu, tsv = thr(su), thr(sv)
            ok = True
            parts = []

            if tsu > 0:
                L = inshape_len(idx, su)
                ok &= (L >= tsu)
                parts.append(f"{su}:{L:.0f}/{tsu:.0f}m")
            if tsv > 0:
                L = inshape_len(idx, sv)
                ok &= (L >= tsv)
                parts.append(f"{sv}:{L:.0f}/{tsv:.0f}m")

            if not ok:
                cls.append("spur")
                why.append("min_inshape_len_fail:" + ",".join(parts))
                continue

        cls.append(base)
        why.append(base_why)

    pipes_out["trade_class"] = cls
    pipes_out["trade_reason"] = why

    # 6) trade pairs (direct + corridor)
    rows = []

    direct = pipes_out[pipes_out["trade_class"] == "trade"]
    if len(direct):
        a = np.minimum(direct["start_shape"], direct["end_shape"])
        b = np.maximum(direct["start_shape"], direct["end_shape"])
        trade_pairs = (
            direct.assign(shape_a=a, shape_b=b)
                  .groupby(["shape_a", "shape_b"], as_index=False)
                  .agg(n_pipelines=("pipeline_id", "count"),
                       sum_capacity_mw=("ch4_capacity_mw", "sum"))
        )
        trade_pairs["kind"] = "direct"
        rows.append(trade_pairs)

    corridor_ids = [cid for cid, info in out_info.items() if info["kind"] == "corridor"]
    if corridor_ids:
        comp_nodes = {}
        for n, cid in node_to_comp.items():
            comp_nodes.setdefault(cid, set()).add(n)

        corr_rows = []
        for cid in corridor_ids:
            touched = out_info[cid]["touched_shapes"]
            outs = comp_nodes.get(cid, set())

            valid = []
            for s in touched:
                entry_nodes = {nbr for outn in outs for nbr in G.neighbors(outn) if G.nodes[nbr]["shape_id"] == s}
                if any(ev(en, s) in strong for en in entry_nodes):
                    valid.append(s)

            for a, b in combinations(sorted(valid), 2):
                corr_rows.append({"shape_a": a, "shape_b": b, "kind": f"corridor:{cid}",
                                  "n_pipelines": np.nan, "sum_capacity_mw": np.nan})

        if corr_rows:
            rows.append(pd.DataFrame(corr_rows))

    trade_pairs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["shape_a", "shape_b", "kind", "n_pipelines", "sum_capacity_mw"]
    )

    return pipes_out, trade_pairs, nodes_labeled
