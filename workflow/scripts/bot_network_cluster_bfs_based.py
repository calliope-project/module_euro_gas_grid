from collections import deque
from itertools import combinations

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

OUT_LABEL = "OUT"
OFFSHORE_CODE = "XXX"
ANCHOR_ETYPES = {"source", "sink", "terminal", "junction"}
MAX_BFS_NODES = 50_000


# =========================
# Stage 1 — Label nodes to shapes (authoritative)
# =========================


def label_nodes_to_shapes(
    nodes: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    *,
    metric_crs: int = 3035,
    include_boundary: bool = True,
) -> gpd.GeoDataFrame:
    """
    Authoritative spatial labeling: node.shape_id is either a provided shapes.shape_id or OUT_LABEL.
    Overlaps are resolved by choosing the smallest-area shape (enclaves win).
    """
    if nodes.crs != shapes.crs:
        shapes = shapes.to_crs(nodes.crs)

    shp = shapes.copy()
    shp["_area"] = shp.to_crs(metric_crs).geometry.area  # tie-break: smallest wins

    pred = "intersects" if include_boundary else "within"
    j = gpd.sjoin(
        nodes[["node_id", "geometry"]],
        shp[["shape_id", "_area", "geometry"]],
        how="left",
        predicate=pred,
    ).sort_values("_area", kind="stable")

    best_shape = j.drop_duplicates("node_id", keep="first").set_index("node_id")[
        "shape_id"
    ]

    out = nodes.copy()
    out["shape_id"] = out["node_id"].map(best_shape).fillna(OUT_LABEL)
    return out


# =========================
# Stage 2 — Build topology graph
# =========================


def build_topology_graph(
    nodes_labeled: gpd.GeoDataFrame, pipelines: gpd.GeoDataFrame
) -> nx.Graph:
    """Undirected graph of the pipeline network with node attributes attached."""
    G = nx.Graph()

    for r in nodes_labeled[["node_id", "shape_id", "country_id", "etype"]].itertuples(
        index=False
    ):
        G.add_node(
            r.node_id, shape_id=r.shape_id, country_id=r.country_id, etype=r.etype
        )

    for r in pipelines[["start_node_id", "end_node_id"]].itertuples(index=False):
        G.add_edge(r.start_node_id, r.end_node_id)

    return G


# =========================
# Stage 3 — OUT component analysis
# =========================


def analyze_out_components(
    G: nx.Graph,
    *,
    internal_shape_ids: set[str],
    internal_countries: set[str],
    shape_country: dict[str, str],
) -> tuple[dict[int, dict], dict[int, int]]:
    """OUT components are connected components induced by nodes with shape_id == OUT_LABEL.

    out_info[cid] contains:
      kind ∈ {corridor_internal, corridor_external, single_touch_out, external_only}
      touched_internal: list[str]
      external_countries: list[str]
      absorbed_candidate: bool
      absorbed_mode: 'offshore_xxx' | 'same_country_out' | None

    node_to_outcomp maps OUT node_id -> component id.
    """

    def is_external_country(cid: str) -> bool:
        return (cid != OFFSHORE_CODE) and (cid not in internal_countries)

    out_nodes = [n for n in G.nodes if G.nodes[n]["shape_id"] == OUT_LABEL]
    out_sub = G.subgraph(out_nodes)

    out_info: dict[int, dict] = {}
    node_to_outcomp: dict[int, int] = {}

    for cid, comp in enumerate(nx.connected_components(out_sub)):
        comp = set(comp)
        for n in comp:
            node_to_outcomp[n] = cid

        touched_internal: set[str] = set()
        comp_countries: set[str] = set()
        external_countries: set[str] = set()

        for n in comp:
            c = G.nodes[n]["country_id"]
            comp_countries.add(c)
            if is_external_country(c):
                external_countries.add(c)

            for nbr in G.neighbors(n):
                sid = G.nodes[nbr]["shape_id"]
                if sid in internal_shape_ids:
                    touched_internal.add(sid)

        if len(touched_internal) >= 2 and len(external_countries) == 0:
            kind = "corridor_internal"
        elif len(touched_internal) >= 1 and len(external_countries) >= 1:
            kind = "corridor_external"
        elif len(touched_internal) == 1:
            kind = "single_touch_out"
        else:
            kind = "external_only"

        absorbed_candidate = False
        absorbed_mode = None
        if kind == "single_touch_out" and len(external_countries) == 0:
            only_shape = next(iter(touched_internal))
            internal_country = shape_country[only_shape]

            # A) offshore-only
            if comp_countries == {OFFSHORE_CODE}:
                absorbed_candidate = True
                absorbed_mode = "offshore_xxx"

            # B) OUT nodes labelled with same country_id as the touched internal shape (plus maybe XXX)
            elif comp_countries.issubset({OFFSHORE_CODE, internal_country}):
                absorbed_candidate = True
                absorbed_mode = "same_country_out"

        out_info[cid] = {
            "kind": kind,
            "touched_internal": sorted(touched_internal),
            "external_countries": sorted(external_countries),
            "absorbed_candidate": absorbed_candidate,
            "absorbed_mode": absorbed_mode,
        }

    return out_info, node_to_outcomp


# =========================
# Stage 4 — Evidence sets + BFS evidence
# =========================


def build_shape_sets(
    G: nx.Graph, internal_shape_ids: set[str]
) -> dict[str, dict[str, set[int]]]:
    """Precompute nodes/anchors/boundary/interior per internal shape (for BFS evidence)."""
    sets: dict[str, dict[str, set[int]]] = {}

    for n, d in G.nodes(data=True):
        sid = d["shape_id"]
        if sid not in internal_shape_ids:
            continue
        S = sets.setdefault(
            sid,
            {"nodes": set(), "anchors": set(), "boundary": set(), "interior": set()},
        )
        S["nodes"].add(n)
        if d["etype"] in ANCHOR_ETYPES:
            S["anchors"].add(n)

    for sid, S in sets.items():
        for n in S["nodes"]:
            if any(G.nodes[v]["shape_id"] != sid for v in G.neighbors(n)):
                S["boundary"].add(n)
        S["interior"] = S["nodes"] - S["boundary"]

    return sets


def bfs_evidence(
    G: nx.Graph, *, entry: int, sid: str, shape_sets: dict[str, dict[str, set[int]]]
) -> str | None:
    """Return 'anchor'|'interior'|'boundary_only'|None while staying within sid."""
    S = shape_sets.get(sid)
    if S is None:
        return None

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
            if G.nodes[v]["shape_id"] != sid:
                continue
            seen.add(v)
            q.append(v)
            moved = True

    return "boundary_only" if moved else None


def evidence_cached(
    G: nx.Graph,
    *,
    entry: int,
    sid: str,
    shape_sets: dict[str, dict[str, set[int]]],
    cache: dict[tuple[int, str], str | None],
) -> str | None:
    """Cached wrapper for bfs_evidence()."""
    key = (entry, sid)
    if key not in cache:
        cache[key] = bfs_evidence(G, entry=entry, sid=sid, shape_sets=shape_sets)
    return cache[key]


# =========================
# Stage 5 — Spur thresholds (optional)
# =========================


def thresholds_by_shape(
    shapes: gpd.GeoDataFrame, *, min_land_inshape_m: float, min_marine_inshape_m: float
) -> dict[str, float]:
    """Threshold (meters) for each provided shape based on shape_class."""
    cls = shapes.set_index("shape_id")["shape_class"].to_dict()
    return {
        sid: (min_marine_inshape_m if cls[sid] == "maritime" else min_land_inshape_m)
        for sid in cls
    }


def prepare_length_context(
    pipes_out: gpd.GeoDataFrame, shapes: gpd.GeoDataFrame, *, metric_crs: int
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    """
    Prepare projected geometries needed for in-shape length calculations.
    Returns:
      pipes_m   : pipes_out in metric_crs
      shape_geom: dict[shape_id -> polygon] in metric_crs
    """
    shapes_m = shapes.to_crs(metric_crs).set_index("shape_id")
    shape_geom = shapes_m.geometry.to_dict()
    pipes_m = pipes_out.to_crs(metric_crs)
    return pipes_m, shape_geom


def inshape_len_cached(
    pipes_m: gpd.GeoDataFrame,
    shape_geom: dict[str, object],
    *,
    pipe_idx: int,
    shape_id: str,
    cache: dict[tuple[int, str], float],
) -> float:
    """Cached length of (pipe geometry ∩ shape geometry) in meters."""
    key = (pipe_idx, shape_id)
    if key not in cache:
        cache[key] = (
            pipes_m.loc[pipe_idx, "geometry"].intersection(shape_geom[shape_id]).length
        )
    return cache[key]


def compute_component_len_by_node(
    pipes_out: gpd.GeoDataFrame,
    *,
    pipes_m: gpd.GeoDataFrame,
    shape_geom: dict[str, object],
    thr_by_shape: dict[str, float],
) -> dict[int, float]:
    """
    For each internal shape, compute total in-shape length per connected component
    (restricted to edges fully inside that shape). Map nodes in that component to that total.
    """
    active_shapes = [sid for sid, t in thr_by_shape.items() if t > 0]
    if not active_shapes:
        return {}

    comp_len_by_node: dict[int, float] = {}

    for sid in active_shapes:
        mask = (pipes_out["start_shape"] == sid) & (pipes_out["end_shape"] == sid)
        if not mask.any():
            continue

        poly = shape_geom[sid]
        lens = pipes_m.loc[mask, "geometry"].intersection(poly).length.to_numpy()
        uv = pipes_out.loc[mask, ["start_node_id", "end_node_id"]]

        Gi = nx.Graph()
        for (u, v), L in zip(uv.itertuples(index=False, name=None), lens):
            Gi.add_edge(u, v, w=float(L))

        for comp in nx.connected_components(Gi):
            total = sum(d["w"] for _, _, d in Gi.subgraph(comp).edges(data=True))
            for n in comp:
                prev = comp_len_by_node.get(n, 0.0)
                if total > prev:
                    comp_len_by_node[n] = total

    return comp_len_by_node


# =========================
# Stage 6 — Pipeline labeling + classification
# =========================


def attach_endpoint_labels(
    pipelines: gpd.GeoDataFrame, nodes_labeled: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Attach start/end shapes (authoritative) and start/end country awareness."""
    node_shape = nodes_labeled.set_index("node_id")["shape_id"].to_dict()
    node_country = nodes_labeled.set_index("node_id")["country_id"].to_dict()

    out = pipelines.copy()
    out["start_shape"] = out["start_node_id"].map(node_shape.__getitem__)
    out["end_shape"] = out["end_node_id"].map(node_shape.__getitem__)
    out["start_country_id"] = out["start_node_id"].map(node_country.__getitem__)
    out["end_country_id"] = out["end_node_id"].map(node_country.__getitem__)
    return out


def attach_out_component_metadata(
    pipes_out: gpd.GeoDataFrame,
    *,
    node_to_outcomp: dict[int, int],
    out_info: dict[int, dict],
) -> gpd.GeoDataFrame:
    """Attach OUT component id + kind for edges that touch OUT."""
    out_ids: list[int | None] = []

    cols = ["start_node_id", "end_node_id", "start_shape", "end_shape"]
    for r in pipes_out[cols].itertuples(index=False):
        if r.start_shape == OUT_LABEL:
            out_ids.append(node_to_outcomp[r.start_node_id])
        elif r.end_shape == OUT_LABEL:
            out_ids.append(node_to_outcomp[r.end_node_id])
        else:
            out_ids.append(None)

    out = pipes_out.copy()
    out["out_component_id"] = pd.array(out_ids, dtype="Int64")  # nullable integer array
    out["out_component_kind"] = out["out_component_id"].map(
        lambda cid: None if pd.isna(cid) else out_info[int(cid)]["kind"]
    )
    return out


def classify_edges(
    pipes_out: gpd.GeoDataFrame,
    *,
    G: nx.Graph,
    internal_countries: set[str],
    out_info: dict[int, dict],
    shape_sets: dict[str, dict[str, set[int]]],
    thr_by_shape: dict[str, float],
    pipes_m: gpd.GeoDataFrame | None,
    shape_geom: dict[str, object] | None,
    inshape_cache: dict[tuple[int, str], float] | None,
    comp_len_by_node: dict[int, float],
) -> gpd.GeoDataFrame:
    """
    Classify each pipeline into:
      internal | trade | transit | spur | border_artifact |
      out | out_corridor_link | absorbed | external_ignored
    """
    strong = {"anchor", "interior"}
    ev_cache: dict[tuple[int, str], str | None] = {}

    use_thr = (
        pipes_m is not None
        and shape_geom is not None
        and inshape_cache is not None
        and any(t > 0 for t in thr_by_shape.values())
    )

    classes: list[str] = []
    reasons: list[str] = []

    cols = [
        "start_node_id",
        "end_node_id",
        "start_shape",
        "end_shape",
        "start_country_id",
        "end_country_id",
        "out_component_id",
    ]

    for idx, u, v, su, sv, cu, cv, ocid in pipes_out[cols].itertuples(
        index=True, name=None
    ):
        # --- OUT involvement ---
        if su == OUT_LABEL or sv == OUT_LABEL:
            # stop-at-first-contact: OUT/OUT deep inside the same external country
            if su == OUT_LABEL and sv == OUT_LABEL:
                if (
                    (cu != OFFSHORE_CODE)
                    and (cu not in internal_countries)
                    and (cv == cu)
                ):
                    classes.append("external_ignored")
                    reasons.append(f"external_interior:{cu}")
                    continue

            if ocid is None:
                raise RuntimeError(
                    "Edge touches OUT but has no out_component_id (bug in preprocessing)."
                )

            cid = int(ocid)
            info = out_info[cid]

            if info["kind"] == "external_only":
                classes.append("external_ignored")
                reasons.append("out_component_external_only")
                continue

            if info["absorbed_candidate"]:
                classes.append("absorbed")
                reasons.append(info["absorbed_mode"])
                continue

            if info["kind"] in {"corridor_internal", "corridor_external"}:
                classes.append("out_corridor_link")
                reasons.append(f"out_component:{info['kind']}")
                continue

            classes.append("out")
            reasons.append("single_touch_out_or_mixed")
            continue

        # --- internal/internal ---
        if su == sv:
            classes.append("internal")
            reasons.append("same_shape")
            continue

        # evidence cache
        k1 = (u, su)
        if k1 not in ev_cache:
            ev_cache[k1] = bfs_evidence(G, entry=u, sid=su, shape_sets=shape_sets)
        eu = ev_cache[k1]

        k2 = (v, sv)
        if k2 not in ev_cache:
            ev_cache[k2] = bfs_evidence(G, entry=v, sid=sv, shape_sets=shape_sets)
        evv = ev_cache[k2]

        if (eu in strong) and (evv in strong):
            base, reason = "trade", f"evidence:{eu}+{evv}"
        elif (eu is not None) or (evv is not None):
            base, reason = "transit", f"weak:{eu}+{evv}"
        else:
            base, reason = "border_artifact", "no_in_shape_evidence"

        # --- spur suppression ---
        if use_thr and base in {"trade", "transit"}:
            ok = True
            parts = []

            for node_id, sid in ((u, su), (v, sv)):
                t = thr_by_shape.get(sid, 0.0)
                if t <= 0:
                    continue

                # cached in-shape intersection length
                lk = (idx, sid)
                if lk not in inshape_cache:
                    inshape_cache[lk] = (
                        pipes_m.loc[idx, "geometry"]
                        .intersection(shape_geom[sid])
                        .length
                    )
                Ledge = inshape_cache[lk]

                Lcomp = comp_len_by_node.get(node_id, 0.0)
                ok = ok and ((Ledge >= t) or (Lcomp >= t))
                parts.append(f"{sid}:edge{Ledge:.0f}/thr{t:.0f},comp{Lcomp:.0f}")

            if not ok:
                classes.append("spur")
                reasons.append("min_inshape_len_or_component_fail:" + ",".join(parts))
                continue

        classes.append(base)
        reasons.append(reason)

    out = pipes_out.copy()
    out["trade_class"] = classes
    out["trade_reason"] = reasons
    return out


# =========================
# Stage 7 — Trade pairs summary
# =========================


def summarize_trade_pairs(
    pipes_out: gpd.GeoDataFrame, out_info: dict[int, dict]
) -> pd.DataFrame:
    """Summarize internal trade links and corridor-derived relations."""
    rows = []

    direct = pipes_out[pipes_out["trade_class"] == "trade"]
    if len(direct):
        a = np.minimum(direct["start_shape"], direct["end_shape"])
        b = np.maximum(direct["start_shape"], direct["end_shape"])
        tp = (
            direct.assign(shape_a=a, shape_b=b)
            .groupby(["shape_a", "shape_b"], as_index=False)
            .agg(
                n_pipelines=("pipeline_id", "count"),
                sum_capacity_mw=("ch4_capacity_mw", "sum"),
            )
        )
        tp["kind"] = "direct"
        rows.append(tp)

    corr = []
    for cid, info in out_info.items():
        if info["kind"] == "corridor_internal":
            for a, b in combinations(sorted(info["touched_internal"]), 2):
                corr.append(
                    {
                        "shape_a": a,
                        "shape_b": b,
                        "kind": f"corridor_internal:{cid}",
                        "n_pipelines": np.nan,
                        "sum_capacity_mw": np.nan,
                    }
                )
        elif info["kind"] == "corridor_external":
            for a in info["touched_internal"]:
                for c in info["external_countries"]:
                    corr.append(
                        {
                            "shape_a": a,
                            "shape_b": c,
                            "kind": f"corridor_external:{cid}",
                            "n_pipelines": np.nan,
                            "sum_capacity_mw": np.nan,
                        }
                    )

    if corr:
        rows.append(pd.DataFrame(corr))

    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=["shape_a", "shape_b", "kind", "n_pipelines", "sum_capacity_mw"]
        )
    )


# =========================
# Main orchestration
# =========================


def classify_pipeline_trade(
    shapes: gpd.GeoDataFrame,
    pipelines: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    metric_crs: int = 3035,
    include_boundary: bool = True,
    min_land_inshape_m: float = 0.0,
    min_marine_inshape_m: float = 0.0,
):
    """
    Pipeline trade classification in explicit stages (easy to debug stage-by-stage).
    Assumes schemas validated inputs (ids exist, types are correct, uniqueness holds).
    """
    # Normalize IDs once (avoid int(...) chaff everywhere else)
    nodes = nodes.copy()
    pipelines = pipelines.copy()
    nodes["node_id"] = nodes["node_id"].astype(int)
    pipelines["start_node_id"] = pipelines["start_node_id"].astype(int)
    pipelines["end_node_id"] = pipelines["end_node_id"].astype(int)

    internal_shape_ids = set(shapes["shape_id"])
    internal_countries = set(shapes["country_id"])
    shape_country = shapes.set_index("shape_id")["country_id"].to_dict()

    # Stage 1
    nodes_labeled = label_nodes_to_shapes(
        nodes, shapes, metric_crs=metric_crs, include_boundary=include_boundary
    )

    # Stage 2
    G = build_topology_graph(nodes_labeled, pipelines)

    # Stage 3
    out_info, node_to_outcomp = analyze_out_components(
        G,
        internal_shape_ids=internal_shape_ids,
        internal_countries=internal_countries,
        shape_country=shape_country,
    )

    # Stage 4
    shape_sets = build_shape_sets(G, internal_shape_ids)

    # Stage 5
    thr_by_shape = thresholds_by_shape(
        shapes,
        min_land_inshape_m=min_land_inshape_m,
        min_marine_inshape_m=min_marine_inshape_m,
    )
    use_thr = any(t > 0 for t in thr_by_shape.values())

    # Stage 6 (label pipelines then classify)
    pipes_out = attach_endpoint_labels(pipelines, nodes_labeled)
    pipes_out = attach_out_component_metadata(
        pipes_out, node_to_outcomp=node_to_outcomp, out_info=out_info
    )

    if use_thr:
        pipes_m, shape_geom = prepare_length_context(
            pipes_out, shapes, metric_crs=metric_crs
        )
        inshape_cache: dict[tuple[int, str], float] = {}
        comp_len_by_node = compute_component_len_by_node(
            pipes_out, pipes_m=pipes_m, shape_geom=shape_geom, thr_by_shape=thr_by_shape
        )
    else:
        pipes_m = None
        shape_geom = None
        inshape_cache = None
        comp_len_by_node = {}

    pipes_out = classify_edges(
        pipes_out,
        G=G,
        internal_countries=internal_countries,
        out_info=out_info,
        shape_sets=shape_sets,
        thr_by_shape=thr_by_shape,
        pipes_m=pipes_m,
        shape_geom=shape_geom,
        inshape_cache=inshape_cache,
        comp_len_by_node=comp_len_by_node,
    )

    # Stage 7
    trade_pairs = summarize_trade_pairs(pipes_out, out_info)

    return pipes_out, trade_pairs, nodes_labeled
