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

NG_LHV_KWH_PER_M3 = 10.5
NG_LHV_MJ_PER_M3  = NG_LHV_KWH_PER_M3 * 3.6  # 37.8 MJ/m3


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


def match_pipes_to_nodes(
    pipes: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    *,
    buffer_dist: float = 100.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Link assign pipelines to node IDs."""
    _utils.check_projected_crs(pipes.crs)
    crs = pipes.crs
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
        bad_pipe_ids.update(matched.loc[unmatched, "pipeline_id"].unique())

    # Drop: any pipeline endpoint with multiple nearest matches (ties)
    multi = matched.duplicated(subset=["pipeline_id", "endpoint"], keep=False)
    if multi.any():
        bad_pipe_ids.update(matched.loc[multi, "pipeline_id"].unique())

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


def initialise_nodes(nodes_file: str, countries_file: str, proj_crs) -> gpd.GeoDataFrame:
    """Fit SciGrid Nodes to our schema."""
    raw = _utils.to_crs(gpd.read_file(nodes_file).reset_index(drop=True), proj_crs)
    countries = _utils.to_crs(gpd.read_parquet(countries_file), proj_crs)
    nodes = gpd.GeoDataFrame(
        {
            "node_id": raw.index.to_numpy(dtype=int),
            "geometry": raw["geometry"],
        },
        geometry="geometry",
        crs=raw.crs,
    )
    nodes = nodes.join(
        _utils.match_points_to_polygons(nodes, countries, "sovereign_id")
    )
    return nodes


def initialise_pipelines(pipelines_file: str, crs: str) -> gpd.GeoDataFrame:
    """Fit SciGrid PipeSegments to our schema."""
    raw = _utils.to_crs(gpd.read_file(pipelines_file).reset_index(drop=True), crs)

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
                "etype": "pipeline",
            },
            geometry=raw.geometry,
            crs=raw.crs,
        )
        .join(param)
        .join(method)
    )

    return pipes


def estimate_capacity(
    pipes: gpd.GeoDataFrame,
    inferred_mm: float | None = None,
    *,
    recalculate_below_mw: float | None = None,
    capacity_correction_threshold: float | None = None,
    excluded_pipeline_ids: list[int] | None = None,
    bidirectional_below_distance: float | None = None,
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
        excluded_pipeline_ids (list[int], optional):
            List of unique pipeline IDs to drop.
        bidirectional_below_distance (int):
            Pipelines below this length will be assumed to be bidirectonal.

    Returns:
        gpd.GeoDataFrame: pipeline dataframe with CH4 capacity.
    """
    pipes = pipes.copy()

    conversion_factor = 1e6 * NG_LHV_MJ_PER_M3 / (24 * 60 * 60)

    # Base estimate: convert reported capacity (million m3 / day) to MW
    pipes["capacity_mw"] = pipes["max_cap_M_m3_per_d"] * conversion_factor

    # Optionally override inferred diameters
    inferred_mask = pipes["diameter_method"].ne("raw")
    if inferred_mm is not None:
        pipes.loc[inferred_mask, "diameter_mm"] = inferred_mm
        pipes.loc[inferred_mask, "diameter_method"] = "inferred"

    # Optionally mark short lines as bidirectional
    if bidirectional_below_distance is not None:
        length = pipes.geometry.length
        short_lines = length < float(bidirectional_below_distance)
        pipes.loc[short_lines, "is_bidirectional"] = True

    # Optional exclusion list (configurable). NordStream can be added to it.
    exclude_ids: set[int] = set(excluded_pipeline_ids or [])
    if exclude_ids:
        pipes = pipes.loc[~pipes["pipeline_id"].isin(exclude_ids)]

    # cap_diam_mw should match the *current* pipes
    cap_diam_mw = pipes["diameter_mm"].apply(_diameter_to_capacity)

    discrepant_mask = pd.Series(False, index=pipes.index)
    if capacity_correction_threshold is not None:
        ratio = pipes["capacity_mw"] / cap_diam_mw
        thr = float(capacity_correction_threshold)

        # exclude high pressure pipelines from ratio-based corrections
        below_max_press = pipes["max_pressure_bar"] < 220
        discrepant_mask = ((ratio > thr) | (ratio < 1 / thr)) & below_max_press

    # Optional: force recalculation below a threshold
    low_mask = pd.Series(False, index=pipes.index)
    if recalculate_below_mw is not None:
        low_mask = pipes["capacity_mw"] <= recalculate_below_mw

    # Apply corrections
    correction_mask = discrepant_mask | low_mask
    pipes.loc[correction_mask, "capacity_mw"] = cap_diam_mw.loc[correction_mask]

    # Track method provenance
    pipes["capacity_mw_method"] = "conversion factor based"
    pipes.loc[correction_mask, "capacity_mw_method"] = "diameter based"
    pipes.loc[discrepant_mask & ~low_mask, "capacity_mw_method"] = (
        "diameter based (ratio discrepancy)"
    )
    pipes.loc[low_mask & ~discrepant_mask, "capacity_mw_method"] = (
        "diameter based (below threshold)"
    )
    pipes.loc[low_mask & discrepant_mask, "capacity_mw_method"] = (
        "diameter based (ratio discrepancy+below threshold)"
    )
    return pipes


def plot(
    pipes_file: str, nodes_file: str, countries_file: str, *, crs: str = "EPSG:3035"
):
    """Plot general information of the harmonised datasets."""
    pipes = gpd.read_parquet(pipes_file).to_crs(crs)
    nodes = gpd.read_parquet(nodes_file).to_crs(crs)
    countries = gpd.read_parquet(countries_file).to_crs(crs)

    fig = plt.figure(figsize=(10, 10), layout="compressed")

    # 2 rows × 3 cols, last col is the colorbar gutter.
    gs = fig.add_gridspec(2, 4, width_ratios=(1, 0.045, 1, 0.045), height_ratios=(1, 1))

    # Main panels
    ax_ul = fig.add_subplot(gs[0, 0])  # onshore/offshore
    ax_cb1 = fig.add_subplot(gs[0, 1])  # colorbar (only for top row)
    ax_uc = fig.add_subplot(gs[0, 2])  # capacity
    ax_cb2 = fig.add_subplot(gs[0, 3])  # colorbar (only for top row)

    ax_bl = fig.add_subplot(gs[1, 0])  # network properties

    # Bottom-right: 3 stacked densities
    dens_gs = gs[1, 2].subgridspec(3, 1)
    ax_k1 = fig.add_subplot(dens_gs[0, 0])
    ax_k2 = fig.add_subplot(dens_gs[1, 0])
    ax_k3 = fig.add_subplot(dens_gs[2, 0])

    # Keep the bottom-right gutter empty/invisible
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.axis("off")
    ax_empty = fig.add_subplot(gs[1, 3])
    ax_empty.axis("off")

    axes = {
        "ul": ax_ul,
        "uc": ax_uc,
        "bl": ax_bl,
        "k1": ax_k1,
        "k2": ax_k2,
        "k3": ax_k3,
        "cb1": ax_cb1,
        "cb2": ax_cb2,
    }

    # ---- shared view window ----
    xlim, ylim = _plots.get_padded_bounds(pipes, pad_frac=0.02)
    countries_view = countries.cx[xlim[0] : xlim[1], ylim[0] : ylim[1]]

    # ---- UL: diameter ----
    title = r"Pipeline diameter ($mm$)"
    cmap = Colormap("bids:magma").to_mpl()
    v = pipes["diameter_mm"]
    norm = mpl.colors.Normalize(vmin=float(v.min()), vmax=float(v.max()))
    countries_view.plot(ax=ax_ul, color="black", alpha=0.05, zorder=-2)
    countries_view.boundary.plot(ax=ax_ul, color="black", lw=0.5, zorder=-1)
    pipes.plot("diameter_mm", ax=ax_ul, cmap=cmap, norm=norm, lw=0.8)
    _plots.style_map_plot(ax_ul, title, xlim, ylim)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    _ = fig.colorbar(sm, cax=ax_cb1)

    # ---- UC: capacity + dedicated colorbar axis ----
    title = r"Pipeline capacity ($MW$)"
    cmap = Colormap("bids:fake_parula").to_mpl()
    v = pipes["capacity_mw"]
    norm = mpl.colors.Normalize(vmin=float(v.min()), vmax=float(v.max()))
    countries_view.plot(ax=ax_uc, color="black", alpha=0.05, zorder=-2)
    countries_view.boundary.plot(ax=ax_uc, color="black", lw=0.5, zorder=-1)
    pipes.plot("capacity_mw", ax=ax_uc, cmap=cmap, norm=norm, lw=0.8)
    _plots.style_map_plot(ax_uc, title, xlim, ylim)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    _ = fig.colorbar(sm, cax=ax_cb2)

    # ---- BL: network properties ----
    title = "Gas network properties"
    countries_view.plot(ax=ax_bl, color="black", alpha=0.05, zorder=-2)
    countries_view.boundary.plot(ax=ax_bl, color="black", lw=0.5, zorder=-1)
    etypes = gpd.GeoDataFrame(
        pd.concat([i[["etype", "geometry"]] for i in [nodes, pipes]]), crs=crs
    )
    cmap2 = Colormap("colorbrewer:Accent_r").to_mpl()
    etypes.plot(
        ax=ax_bl,
        column="etype",
        cmap=cmap2,
        categorical=True,
        markersize=3,
        lw=0.5,
        legend=True,
    )
    _plots.style_map_plot(ax_bl, title, xlim, ylim)

    # ---- densities ----
    _plots.plot_density(ax_k1, pipes["diameter_mm"], r"Pipeline diameter ($mm$)")
    _plots.plot_density(ax_k2, pipes["capacity_mw"], r"Pipeline capacity ($MW$)")
    unit_name = pipes.crs.axis_info[0].unit_name
    _plots.plot_density(
        ax_k3, pipes.geometry.length, rf"Pipeline length (${unit_name}$)"
    )

    return fig, axes


def main():
    """Main process when calling from snakemake."""
    proj_crs = snakemake.params.projected_crs
    _utils.check_projected_crs(proj_crs)
    imputation = snakemake.params.imputation

    # Transformations
    countries_file = snakemake.input.countries
    nodes = initialise_nodes(snakemake.input.raw_nodes, countries_file, proj_crs)
    pipes = initialise_pipelines(snakemake.input.raw_pipelines, proj_crs)
    pipes, nodes = match_pipes_to_nodes(pipes, nodes)
    nodes = _utils.compute_node_graph_attributes(pipes, nodes)
    pipes = estimate_capacity(
        pipes,
        inferred_mm=imputation.get("inferred_mm", None),
        recalculate_below_mw=imputation.get("recalculate_below_mw", None),
        capacity_correction_threshold=imputation.get("capacity_correction_threshold", None),
        excluded_pipeline_ids=imputation.get("excluded_pipeline_ids", None),
        bidirectional_below_distance=imputation.get("bidirectional_below_distance", None),
    )

    # Validation
    pipes_out_file = snakemake.output.pipelines
    nodes_out_file = snakemake.output.nodes
    _schemas.PipelineSchema.validate(pipes).to_parquet(pipes_out_file)
    _schemas.NodeSchema.validate(nodes).to_parquet(nodes_out_file)

    # Analysis
    fig, _ = plot(pipes_out_file, nodes_out_file, countries_file, crs=proj_crs)
    fig.savefig(snakemake.output.fig, dpi=300)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")
    main()
