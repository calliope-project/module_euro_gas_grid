"""Aggregate gas network to the provided shapes.

Steps:
1. Split pipelines to ensure a node is located at boundary crossings.
2. Assign a `shape_id` to each edge (pipeline) if >50% of its length is within it.
3. As a fallback, a `country_id` is assigned to lines outside shapes.
4. For the rest, it is assumed to be an offshore pipeline.


# Important notes:
- It is assumed that the shapefile provided has no or little overlaps.
- Requires a projected CRS.
"""

import sys
from typing import TYPE_CHECKING, Any

import _line_splitter
import _plots
import _schemas
import _utils
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import LineString, Point

if TYPE_CHECKING:
    snakemake: Any


def split_pipeline_network_on_shapes(
    pipelines: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    *,
    crs: str = "EPSG:3035",
    snap_tol_m: float = 1.0,
    min_segment_len_m: float = 0.0,
    next_pipe_id: int | None = None,
    next_node_id: int | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split pipelines at shape boundaries.

    Returns:
      - pipelines_split: segmented pipelines with rewired start/end nodes
      - new_nodes: newly created cut nodes

    Debug columns on pipelines_split:
      - parent_pipeline_id
      - segment_index
      - start_measure_m, end_measure_m
      - n_cuts, n_segments, is_split
    """
    _utils.check_projected_crs(crs)
    pipes_m = _utils.to_crs(pipelines, crs)
    nodes_m = _utils.to_crs(nodes, crs)
    shapes_m = _utils.to_crs(shapes, crs)

    if next_pipe_id is None:
        next_pipe_id = int(pipelines["pipeline_id"].max()) + 1
    if next_node_id is None:
        next_node_id = int(nodes["node_id"].max()) + 1

    boundary = _line_splitter.build_boundary(shapes_m.geometry)
    node_geom_m = nodes_m.set_index("node_id")["geometry"].to_dict()

    seg_rows: list[dict] = []
    new_nodes_rows_m: list[dict] = []

    for r in pipes_m.itertuples(index=False):
        parent_pid = r.pipeline_id
        start_id = r.start_node_id
        end_id = r.end_node_id

        line: LineString = r.geometry

        # Ensure geometry direction matches start_node_id -> end_node_id
        spt = node_geom_m[start_id]
        if Point(line.coords[0]).distance(spt) > Point(line.coords[-1]).distance(spt):
            line = LineString(list(line.coords)[::-1])

        segments, cuts = _line_splitter.cut_line_by_boundary_points(
            line,
            boundary,
            snap_tol_m=snap_tol_m,
            min_segment_len_m=min_segment_len_m,
            endpoint_exclusion_m=snap_tol_m,
        )

        n_cuts = len(cuts)
        n_segments = len(segments)
        is_split = n_segments > 1

        # Allocate fresh node_id per cut (no cross-pipeline sharing)
        cut_node_ids: list[int] = []
        for cut in cuts:
            nid = next_node_id
            next_node_id += 1
            cut_node_ids.append(nid)

            new_nodes_rows_m.append(
                {
                    "node_id": nid,
                    "parent_pipeline_id": parent_pid,
                    "measure_m": float(cut.measure),
                    "geometry": cut.point,
                }
            )

        # Emit segments
        for i, seg in enumerate(segments):
            u = start_id if i == 0 else cut_node_ids[i - 1]
            v = end_id if i == n_segments - 1 else cut_node_ids[i]

            row = r._asdict()
            row["pipeline_id"] = next_pipe_id
            next_pipe_id += 1

            row["geometry"] = seg.geometry
            row["start_node_id"] = u
            row["end_node_id"] = v

            # mapping + debug
            row["parent_pipeline_id"] = parent_pid
            row["segment_index"] = i
            row["start_measure_m"] = float(seg.start)
            row["end_measure_m"] = float(seg.end)
            row["n_cuts"] = n_cuts
            row["n_segments"] = n_segments
            row["is_split"] = is_split

            seg_rows.append(row)

    pipelines_split_m = gpd.GeoDataFrame(seg_rows, geometry="geometry", crs=crs)
    pipelines_split = _utils.to_crs(pipelines_split_m, pipelines.crs)

    new_nodes_m = gpd.GeoDataFrame(new_nodes_rows_m, geometry="geometry", crs=crs)
    new_nodes_m = _utils.compute_node_graph_attributes(pipelines_split_m, new_nodes_m)
    nodes_split = pd.concat(
        [nodes, _utils.to_crs(new_nodes_m, nodes.crs)], ignore_index=True
    )

    return pipelines_split, nodes_split


def plot(
    pipelines_file: str,
    nodes_file: str,
    shapes_file: str,
    output_file: str,
    *,
    crs: str = "EPSG:3035",
):
    """Simple plot showing assigned areas."""
    nodes = _utils.to_crs(gpd.read_parquet(nodes_file), crs)
    pipes = _utils.to_crs(gpd.read_parquet(pipelines_file), crs)
    shapes = _utils.to_crs(gpd.read_parquet(shapes_file), crs)

    inside = pipes["shape_id"].notna()
    external = pipes["shape_id"].isna() & pipes["country_id"].notna()

    pipes = pipes.copy()
    pipes["assignment"] = np.select(
        [inside, external], ["inside shapes", "external country"], default="offshore"
    )

    fig, ax = plt.subplots(layout="constrained")
    shapes.boundary.plot(ax=ax, lw=0.5, color="black")
    pipes.plot("assignment", ax=ax, legend=True, lw=1)
    nodes.plot(ax=ax, color="grey", markersize=2)

    _plots.style_map_plot(ax, "Clustered pipelines")
    fig.savefig(output_file, dpi=300)


def main():
    """Main clustering process."""
    proj_crs = snakemake.params.projected_crs
    min_segment_length = snakemake.params.min_segment_length
    shapes_input = snakemake.input.shapes

    # read + validate
    shapes = _schemas.ShapesSchema.validate(gpd.read_parquet(shapes_input))
    countries = _schemas.CountriesSchema.validate(
        gpd.read_parquet(snakemake.input.countries)
    )
    nodes = _schemas.NodeSchema.validate(gpd.read_parquet(snakemake.input.nodes))
    pipes = _schemas.PipelineSchema.validate(
        gpd.read_parquet(snakemake.input.pipelines)
    )

    # Work in projected CRS throughout
    out_crs = shapes.crs
    shapes = _utils.to_crs(shapes, proj_crs)
    countries = _utils.to_crs(countries, proj_crs)
    nodes = _utils.to_crs(nodes, proj_crs)
    pipes = _utils.to_crs(pipes, proj_crs)

    # split on boundaries (shapes first, then countries)
    pipes, nodes = split_pipeline_network_on_shapes(
        pipes, nodes, shapes, min_segment_len_m=min_segment_length, crs=proj_crs
    )
    pipes, nodes = split_pipeline_network_on_shapes(
        pipes, nodes, countries, min_segment_len_m=min_segment_length, crs=proj_crs
    )

    # assign polygon metadata
    pipes = pipes.join(
        _utils.match_lines_to_polygons(
            pipes, shapes, polygon_value_cols=["shape_id", "country_id"]
        )
    )
    mask = pipes["country_id"].isna()
    if mask.any():
        # Assign country_id to countries outside of shapes
        missing = _utils.match_lines_to_polygons(
            pipes.loc[mask], countries, polygon_value_cols=["sovereign_id"]
        )
        replace_sovereign = snakemake.params.replace_sovereign
        if replace_sovereign:
            missing["sovereign_id"] = missing["sovereign_id"].replace(replace_sovereign)
        pipes.loc[mask, "country_id"] = missing["sovereign_id"]

    # reproject for output + save + plot
    pipes_out = _utils.to_crs(pipes, out_crs)
    nodes_out = _utils.to_crs(nodes, out_crs)

    _schemas.PipelineSchema.validate(pipes_out).to_parquet(snakemake.output.pipelines)
    _schemas.NodeSchema.validate(nodes_out).to_parquet(snakemake.output.nodes)

    plot(
        snakemake.output.pipelines,
        snakemake.output.nodes,
        shapes_input,
        snakemake.output.fig,
        crs=proj_crs,
    )


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w")
    main()
