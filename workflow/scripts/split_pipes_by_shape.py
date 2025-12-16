"""code to aggregate pipelines to shapes."""

import _line_splitter
import _utils
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point


def split_pipeline_network_on_shapes(
    pipelines: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    *,
    metric_crs: int = 3035,
    snap_tol_m: float = 1.0,
    min_segment_len_m: float = 0.0,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split pipelines at shape boundaries.

    Returns:
      - pipelines_split: segmented pipelines with rewired start/end node ids and debug flags
      - new_nodes: newly created cut nodes (node_id, parent_pipeline_id, measure_m, geometry)

    Debug columns on pipelines_split:
      - parent_pipeline_id
      - segment_index
      - start_measure_m, end_measure_m
      - n_cuts, n_segments, is_split
    """
    pipes_m = pipelines.to_crs(metric_crs)
    nodes_m = nodes.to_crs(metric_crs)
    shapes_m = shapes.to_crs(metric_crs)

    boundary = _line_splitter.build_boundary(shapes_m.geometry)
    node_geom_m = nodes_m.set_index("node_id")["geometry"].to_dict()

    next_pipe_id = int(pipelines["pipeline_id"].max()) + 1
    next_node_id = int(nodes["node_id"].max()) + 1

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
            endpoint_exclusion_m=snap_tol_m
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

    pipelines_split_m = gpd.GeoDataFrame(seg_rows, geometry="geometry", crs=metric_crs)
    pipelines_split = pipelines_split_m.to_crs(pipelines.crs)

    new_nodes_m = gpd.GeoDataFrame(new_nodes_rows_m, geometry="geometry", crs=metric_crs)
    new_nodes_m = _utils.compute_node_graph_attributes(pipelines_split, new_nodes_m)
    nodes_split = pd.concat([nodes, new_nodes_m.to_crs(nodes.crs)], ignore_index=True)

    return pipelines_split, nodes_split
