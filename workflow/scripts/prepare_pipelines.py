"""Filter SciGrid pipelines to fit our data standards.

Some portions of this code were adapted from PyPSA-Eur (https://github.com/pypsa/pypsa-eur)
Copyright (c) 2017-2024 The PyPSA-Eur Authors
Licensed under the MIT License
Commit: 822a92729e6973aa3aff741d6c94f1da2c75e8b2
"""

import sys
import warnings
from typing import TYPE_CHECKING, Any

import _plots
import _schemas
import _utils
import country_converter as coco
import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
from cmap import Colormap
from matplotlib import pyplot as plt
from shapely.geometry import Point

if TYPE_CHECKING:
    snakemake: Any

# Density at normal temperature and pressure (closer to operating conditions than STP)
# https://www.engineeringtoolbox.com/gas-density-d_158.html
CH4_KG_M3 = 0.668
# Typical values for natural gas (CH4)
# https://ocw.tudelft.nl/wp-content/uploads/Summary_table_with_heating_values_and_CO2_emissions.pdf
CH4_HHV_MJ_PER_KG = 55
# Destroyed underwater pipelines.
# https://en.wikipedia.org/wiki/Nord_Stream_pipelines_sabotage
NORDSTREAM_IDS = (6055, 6364)


def _line_midpoint_safe(geom):
    """Fallback function to get the midpoint in the line."""
    if geom is None or geom.is_empty:
        return None
    g = geom
    if g.geom_type == "MultiLineString":
        g = max(g.geoms, key=lambda ls: ls.length)
    if getattr(g, "length", 0) == 0:
        return g.centroid
    return g.interpolate(0.5, normalized=True)


def _diameter_to_capacity(pipe_diameter_mm: float) -> float:
    """Estimate natural gas (CH4) pipeline capacity from diameter.

    These values have been back-converted from the European Hydrogen Backbone report,
    using the stated  0.8*MW_CH4 = MW_H2 conversion assumption in p.15.
    https://ehb.eu/files/downloads/2020_European-Hydrogen-Backbone_Report.pdf

    Adapted from PyPSA-Eur (https://github.com/pypsa/pypsa-eur).

    Args:
        pipe_diameter_mm (float): Pipe diameter [mm]

    Returns:
        float: Pipeline capacity [MW]
    """
    # Anchor points: (diameter_mm, capacity_MW)
    p0 = (0.0, 0.0)
    p1 = (500.0, 1500.0)
    p2 = (600.0, 5000.0)
    p3 = (900.0, 11250.0)
    p4 = (1200.0, 21700.0)

    def line_through(pa, pb):
        """Return (m, a) for y = a + m*x through points pa,pb."""
        (x1, y1), (x2, y2) = pa, pb
        m = (y2 - y1) / (x2 - x1)  # [MW/mm]
        a = y1 - m * x1  # [MW]
        return m, a

    d = pipe_diameter_mm
    if d < p1[0]:
        m0, a0 = line_through(p0, p1)
        return a0 + m0 * d
    elif d < p2[0]:
        m1, a1 = line_through(p1, p2)
        return a1 + m1 * d
    elif d < p3[0]:
        m2, a2 = line_through(p2, p3)
        return a2 + m2 * d
    else:
        m3, a3 = line_through(p3, p4)
        return a3 + m3 * d


def _fill_country_ids(
    nodes: gpd.GeoDataFrame,
    countries_file: str,
    *,
    country_code_col: str = "country_code",
    id_col: str = "sovereign_id",
    missing: str = "XXX",
) -> pd.Series:
    ids = nodes[country_code_col]

    # start with all missing, overwrite if valid
    out = pd.Series(missing, index=nodes.index)
    # convert only codes that are present and not explicitly "unknown"
    mask = ids.notna() & ~ids.isin({"XX", missing})
    if mask.any():
        unique = pd.unique(ids.loc[mask])
        converted = coco.convert(unique.tolist(), to="iso3", not_found=np.nan)
        translator = dict(zip(unique, np.atleast_1d(converted)))
        out.loc[mask] = ids.loc[mask].map(translator).fillna(missing)

    # spatial fill only remaining unknowns
    mask = out.eq(missing) & nodes.geometry.notna()
    if mask.any():
        countries = gpd.read_parquet(countries_file)[[id_col, "geometry"]]  # <- fix
        countries = _utils.to_crs(countries, nodes.crs)

        joined = gpd.sjoin(
            nodes.loc[mask, ["geometry"]], countries, how="inner", predicate="within"
        )

        if joined.index.duplicated(keep=False).any():
            bad = joined.index[joined.index.duplicated(keep=False)].unique().tolist()
            raise RuntimeError(f"Ambiguous country match for node index(es): {bad}")
        out.loc[joined.index] = joined[id_col].astype(str).values

    return out


def match_pipes_to_nodes(
    pipes: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    buffer_dist: float = 100.0,
    crs: str = "EPSG:3035",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Link assign pipelines to node IDs."""
    _utils.check_projected_crs(crs)
    pipes = _utils.to_crs(pipes, crs)
    nodes = _utils.to_crs(nodes, crs)

    n = len(pipes)

    # build aligned endpoints (start,end,start,end,...) w/ geometry
    start_pts = pipes.geometry.map(lambda ls: Point(ls.coords[0])).to_numpy()
    end_pts = pipes.geometry.map(lambda ls: Point(ls.coords[-1])).to_numpy()

    geom = np.empty(2 * n, dtype=object)
    geom[0::2] = start_pts
    geom[1::2] = end_pts

    endpoints = gpd.GeoDataFrame(
        {
            "pipeline_id": pd.Index(pipes["pipeline_id"]).repeat(2).to_numpy(),
            "endpoint": np.tile(["start", "end"], n),
        },
        geometry=geom,
        crs=crs,
    )

    matched = gpd.sjoin_nearest(
        endpoints, nodes[["node_id", "geometry"]], how="left", max_distance=buffer_dist
    )

    bad_pipe_ids = set()

    # Drop: any pipeline with an unmatched endpoint
    unmatched = matched["node_id"].isna()
    if unmatched.any():
        bad_pipe_ids |= matched.loc[unmatched, "pipeline_id"].unique().tolist()

    # Drop: any pipeline endpoint with multiple nearest matches (ties)
    multi = matched.duplicated(subset=["pipeline_id", "endpoint"], keep=False)
    if multi.any():
        bad_pipe_ids |= matched.loc[multi, "pipeline_id"].unique().tolist()

    if bad_pipe_ids:
        pipes = pipes.loc[~pipes["pipeline_id"].isin(bad_pipe_ids)]
        matched = matched.loc[~matched["pipeline_id"].isin(bad_pipe_ids)]
        warnings.warn(
            f"Dropped {len(bad_pipe_ids)} pipeline(s) due to endpoint matching issues."
        )

    if pipes.empty:
        return pipes, nodes

    # assign node IDs to remaining pipes
    node_map = matched.pivot(index="pipeline_id", columns="endpoint", values="node_id")
    pipes["start_node_id"] = pipes["pipeline_id"].map(node_map["start"])
    pipes["end_node_id"] = pipes["pipeline_id"].map(node_map["end"])

    # Drop: anything still unmapped (defensive)
    bad = pipes["start_node_id"].isna() | pipes["end_node_id"].isna()
    if bad.any():
        drop_ids = pipes.loc[bad, "pipeline_id"].unique().tolist()
        pipes = pipes.loc[~bad]
        warnings.warn(
            f"Dropped {len(drop_ids)} pipeline(s) due to missing start/end node after pivot."
        )

    pipes["start_node_id"] = pipes["start_node_id"].astype(int)
    pipes["end_node_id"] = pipes["end_node_id"].astype(int)

    # Drop: self-loops
    loops = pipes["start_node_id"].eq(pipes["end_node_id"])
    if loops.any():
        drop_ids = pipes.loc[loops, "pipeline_id"].unique().tolist()
        pipes = pipes.loc[~loops]
        warnings.warn(f"Dropped {len(drop_ids)} self-loop pipeline(s).")

    # Remove nodes with issues
    used = pd.unique(
        np.concatenate(
            [pipes["start_node_id"].to_numpy(), pipes["end_node_id"].to_numpy()]
        )
    )
    nodes = nodes.loc[nodes["node_id"].isin(used)]

    return pipes, nodes


def compute_node_attributes(
    pipes: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # 2. identify node graph metrics
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
            (i == 0) & (o > 0),  # source
            (o == 0) & (i > 0),  # sink
            (d == 1) & (i == 1) & (o == 1),  # terminal (single bidirectional pipe)
            (d == 2) & (i == 1) & (o == 1),  # connection (pass-through)
            (i > 0) & (o > 0),  # junction
        ],
        ["source", "sink", "terminal", "connection", "junction"],
        default="__error__",
    )
    return nodes


def initialise_nodes(nodes_file: str, countries_file: str) -> gpd.GeoDataFrame:
    """Fit SciGrid Nodes to our schema."""
    raw = gpd.read_file(nodes_file).reset_index(drop=True)
    nodes = gpd.GeoDataFrame(
        {
            "node_id": raw.index.to_numpy(dtype=int),
            "country_id": _fill_country_ids(raw, countries_file),
            "geometry": raw["geometry"],
        },
        geometry="geometry",
        crs=raw.crs,
    )
    return nodes


def initialise_pipelines(pipelines_file: str) -> gpd.GeoDataFrame:
    """Fit SciGrid PipeSegments to our schema."""
    raw = gpd.read_file(pipelines_file).reset_index(drop=True)

    param_cols = [
        "diameter_mm",
        "max_cap_M_m3_per_d",
        "max_pressure_bar",
        "is_bidirectional",
    ]
    param = (
        pd.json_normalize(raw["param"])
        .rename(columns={"is_bothDirection": "is_bidirectional"})
        .reindex(columns=param_cols)
    )
    method_map = {
        "diameter_mm": "diameter_method",
        "max_cap_M_m3_per_d": "max_cap_method",
    }
    method = (
        pd.json_normalize(raw["method"])
        .reindex(columns=method_map)
        .rename(columns=method_map)
    )

    pipes = (
        gpd.GeoDataFrame(
            {
                "pipeline_id": raw.index.to_numpy(np.int64),
                "name": raw["name"],
                "etype": "pipeline"
            },
            geometry=raw.geometry,
            crs=raw.crs,
        )
        .join(param)
        .join(method)
    )

    return pipes


def estimate_ch4_capacity(
    pipes: gpd.GeoDataFrame,
    inferred_mm: float | None = None,
    *,
    recalculate_below_mw: float | None = None,
    capacity_correction_threshold: float = 8,
    remove_nordstream: bool = True,
) -> gpd.GeoDataFrame:
    """Estimate natural gas capacity for each pipeline segment.

    Adapted from PyPSA-Eur (https://github.com/pypsa/pypsa-eur).

    Args:
        pipes (gpd.GeoDataFrame):
            pipelines dataframe.
        inferred_mm (float | None, optional):
            replaces Median inferred diameters. Defaults to None.
        recalculate_below_mw (float | None, optional):
            capacities below this will use recalculated values. Defaults to None.
        capacity_correction_threshold (int, optional):
            Ratio threshold to trigger recalculation. Defaults to 8.
        remove_nordstream (bool, optional):
            Drop nordstream pipelines.

    Returns:
        gpd.GeoDataFrame: pipeline dataframe with CH4 capacity.
    """
    pipes = pipes.copy()

    conversion_factor = 1e6 * CH4_KG_M3 * CH4_HHV_MJ_PER_KG / (24 * 60 * 60)

    # Base estimate: convert reported capacity (million m3 / day) to MW
    pipes["ch4_capacity_mw"] = pipes["max_cap_M_m3_per_d"] * conversion_factor

    # Optionally override inferred diameters
    inferred_mask = pipes["diameter_method"].ne("raw")
    if inferred_mm is not None:
        pipes.loc[inferred_mask, "diameter_mm"] = inferred_mm
        pipes.loc[inferred_mask, "diameter_method"] = "inferred"

    nordstream = pipes["pipeline_id"].isin(NORDSTREAM_IDS)
    if remove_nordstream:
        # Recommended option. See:
        # https://en.wikipedia.org/wiki/Nord_Stream_pipelines_sabotage
        pipes = pipes.loc[~nordstream]

    # cap_diam_mw should match the *current* pipes
    cap_diam_mw = pipes["diameter_mm"].apply(_diameter_to_capacity)

    ratio = pipes["ch4_capacity_mw"] / cap_diam_mw
    thr = capacity_correction_threshold
    discrepant_mask = ~pipes["pipeline_id"].isin(NORDSTREAM_IDS) & (
        (ratio > thr) | (ratio < 1 / thr)
    )

    # Optional: force recalculation below a threshold
    low_mask = pd.Series(False, index=pipes.index)
    if recalculate_below_mw is not None:
        low_mask = pipes["ch4_capacity_mw"] <= recalculate_below_mw

    # Apply corrections
    correction_mask = discrepant_mask | low_mask
    pipes.loc[correction_mask, "ch4_capacity_mw"] = cap_diam_mw.loc[correction_mask]

    # Track method provenance
    pipes["ch4_capacity_method"] = "conversion factor based"
    pipes.loc[correction_mask, "ch4_capacity_method"] = "diameter based"
    pipes.loc[discrepant_mask & ~low_mask, "ch4_capacity_method"] = (
        "diameter based (ratio discrepancy)"
    )
    pipes.loc[low_mask & ~discrepant_mask, "ch4_capacity_method"] = (
        "diameter based (below threshold)"
    )
    pipes.loc[low_mask & discrepant_mask, "ch4_capacity_method"] = (
        "diameter based (ratio discrepancy+below threshold)"
    )
    return pipes


def identify_offshore_pipelines(
    pipes: gpd.GeoDataFrame,
    landmass_file: str,
    *,
    n_samples: int = 50,
    land_threshold: float = 0.5,
    crs: str = "EPSG:3035",
) -> gpd.GeoDataFrame:
    """Add boolean column `is_offshore` to the dataset.

    1. Sample `n_samples` points along each line in normalized [0,1].
    2. Compute fraction of samples intersecting land.
    3. If land_fraction < land_threshold, label as offshore.
    """
    # Initial checks
    bad = pipes.geometry.isna() | pipes.geometry.is_empty
    if bad.any():
        preview = pipes.loc[bad, "pipeline_id"].head(5)
        raise RuntimeError(
            f"Pipe geometry is missing/empty for {int(bad.sum())} row(s):\n"
            f"{preview.to_string(index=True)}"
        )
    _utils.check_projected_crs(crs)

    land = _schemas.LandSchema.validate(gpd.read_parquet(landmass_file))
    pipes_p = pipes.to_crs(crs)
    land = land.to_crs(crs)

    land_union = land.geometry.union_all()
    fracs = np.linspace(0.0, 1.0, n_samples)

    points, seg_idx = [], []
    for idx, geom in pipes_p.geometry.items():
        g = geom
        # If something weird sneaks in, fall back to using only the midpoint
        if getattr(g, "length", 0) == 0:
            p = _line_midpoint_safe(g)
            if p is not None:
                points.append(p)
                seg_idx.append(idx)
            continue

        for f in fracs:
            points.append(g.interpolate(float(f), normalized=True))
            seg_idx.append(idx)

    pts = gpd.GeoDataFrame({"seg_idx": seg_idx}, geometry=points, crs=pipes_p.crs)

    on_land = pts.geometry.intersects(land_union)
    land_frac = on_land.groupby(pts["seg_idx"]).mean().reindex(pipes.index)
    if land_frac.isna().any():
        bad_idx = land_frac.index[land_frac.isna()].tolist()
        raise RuntimeError(
            f"Failed to compute land fraction for pipe index(es): {bad_idx[:20]}"
        )

    pipes["is_offshore"] = land_frac <= land_threshold
    return pipes


# def prepare_pipelines(
#     pipelines_file: str,
#     landmass_file: str,
#     projected_crs: str,
#     impute_params: dict,
#     output_file: str,
# ):
#     """Clean and validate the pipelines dataset."""
#     pipes = initialise_pipelines(pipelines_file)
#     pipes = estimate_ch4_capacity(
#         pipes,
#         inferred_mm=impute_params.get("inferred_mm", None),
#         recalculate_below_mw=impute_params.get("recalculate_below_mw", None),
#         remove_nordstream=impute_params["remove_nordstream"],
#     )
#     pipes = identify_offshore_pipelines(pipes, landmass_file, crs=projected_crs)
#     pipes = _schemas.PipelineSchema.validate(pipes)
#     pipes.to_parquet(output_file)


def plot(
    pipes_file: str, countries_file: str, output_file: str, *, crs: str = "EPSG:3035"
):
    """Plot general information of the harmonised pipelines dataset.

    - pipelines offshore/onshore map
    - pipeline capacity map
    - density kernels:
        - pipeline diameter
        - pipeline capacity
    """
    pipes = gpd.read_parquet(pipes_file).to_crs(crs)
    countries = gpd.read_parquet(countries_file).to_crs(crs)

    fig, axes = plt.subplot_mosaic(
        [["ul", "ur", "cb"], ["bl", "br", "."]],
        figsize=(10, 10),
        gridspec_kw={"width_ratios": [1, 1, 0.06], "height_ratios": [7, 3]},
        layout="constrained",
    )

    xlim, ylim = _plots.get_padded_bounds(pipes, pad_frac=0.05)
    countries_view = countries.cx[xlim[0] : xlim[1], ylim[0] : ylim[1]]

    # land/offshore map
    ul = axes["ul"]
    countries_view.boundary.plot(ax=ul, color="black", lw=0.5, zorder=-1)
    offshore = pipes["is_offshore"]
    pipes.loc[~offshore].plot(ax=ul, color="tab:brown", lw=0.6, label="onshore")
    pipes.loc[offshore].plot(ax=ul, color="tab:blue", lw=1.0, label="offshore")
    _plots.style_map_plot(ul, xlim, ylim, "SciGrid gas pipelines")
    ul.legend()

    # capacity map
    ur = axes["ur"]
    title = r"$CH_4$ pipeline capacity ($MW$)"
    cmap = Colormap("bids:fake_parula").to_mpl()

    countries_view.boundary.plot(ax=ur, color="black", lw=0.5, zorder=-1)
    v = pipes["ch4_capacity_mw"]
    norm = mpl.colors.Normalize(vmin=float(v.min()), vmax=float(v.max()))
    pipes.plot("ch4_capacity_mw", ax=ur, cmap=cmap, norm=norm, lw=0.8, legend=False)
    _plots.style_map_plot(ur, xlim, ylim, title)

    # colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=axes["cb"])
    cbar.set_label(title)

    # density kernels
    _plots.plot_density(axes["bl"], pipes["diameter_mm"], r"Pipeline diameter ($mm$)")
    _plots.plot_density(axes["br"], pipes["ch4_capacity_mw"], title)

    fig.savefig(output_file, dpi=300)


def main():
    """Main process when calling from snakemake."""
    crs = "EPSG:3035"  # ETRS89-extended / LAEA Europe
    smk_params = snakemake.params

    countries_file = snakemake.input.countries
    nodes = initialise_nodes(snakemake.input.raw_nodes, countries_file)
    pipes = initialise_pipelines(snakemake.input.raw_pipelines)
    pipes, nodes = match_pipes_to_nodes(pipes, nodes, crs=crs)
    nodes = compute_node_attributes(pipes, nodes)
    pipes = estimate_ch4_capacity(
        pipes,
        inferred_mm=smk_params["imputation"].get("inferred_mm", None),
        recalculate_below_mw=smk_params["imputation"].get("recalculate_below_mw", None),
        remove_nordstream=smk_params["imputation"]["remove_nordstream"],
    )
    pipes = identify_offshore_pipelines(pipes, snakemake.input.landmass, crs=crs)

    pipes_out_file = snakemake.output.pipelines
    nodes_out_file = snakemake.output.nodes
    _schemas.PipelineSchema.validate(pipes).to_parquet(pipes_out_file)
    _schemas.NodeSchema.validate(nodes).to_parquet(nodes_out_file)
    plot(pipes_out_file, countries_file, snakemake.output.fig, crs=crs)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")
    main()
