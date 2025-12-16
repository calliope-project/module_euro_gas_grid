"""Split lines operations.

LineString and polygon geometries MUST be in a projected CRS (units = meters/feet).

Typical usage:
    boundary = build_boundary(polygons.geometry)

    # 1) cut at boundary crossings
    segs1, boundary_pts = cut_line_by_boundary_points(line, boundary)

    # 2) cut at interior points between crossings
    segs2, interior_pts = cut_line_by_interior_points(line, boundary, interior_ratio=0.5)
"""

from collections.abc import Iterable
from dataclasses import dataclass

from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring, unary_union


@dataclass(frozen=True)
class BoundaryPoint:
    """A point located at a boundary crossing measure along a line."""

    measure: float
    "Distance along the line (in the line's CRS units, typically meters)."
    point: Point
    "Shapely geometry."


@dataclass(frozen=True)
class InteriorPoint(BoundaryPoint):
    """A representative joint strictly between two boundary crossings.

    These are derived from consecutive boundary crossings and are meant to avoid
    ambiguity compared to using points that lie exactly on the boundary.
    """

    start: float
    "Boundary crossing measure for the previous crossing."
    end: float
    "Boundary crossing measure for the next crossing."

@dataclass(frozen=True)
class Segment:
    """A LineString segment that is an exact substring of the original line."""

    start: float
    "Distance of the original geometry where this segment starts."
    end: float
    "Distance of the original geometry where this segment ends."
    geometry: LineString


def _extract_intersection_points_and_overlap_lines(
    intersection_geom: BaseGeometry,
) -> tuple[list[Point], list[LineString]]:
    """From line ∩ boundary, extract intersections and overlaps.

    When the line overlaps the boundary, the start/end of that overlap are considered
    boundary crossings, so we can cut the line at those two locations.
    """
    intersection_points: list[Point] = []
    overlap_lines: list[LineString] = []
    stack: list[BaseGeometry] = [intersection_geom]

    while stack:
        g = stack.pop()
        if g.is_empty:
            continue

        t = g.geom_type
        if t == "Point":
            intersection_points.append(g)
        elif t == "MultiPoint":
            intersection_points.extend(list(g.geoms))
        elif t == "LineString":
            if g.length > 0:
                overlap_lines.append(g)
            stack.append(g.boundary)  # overlap endpoints become split points
        elif t == "MultiLineString":
            for ls in g.geoms:
                if ls.length > 0:
                    overlap_lines.append(ls)
                stack.append(ls.boundary)
        elif t == "GeometryCollection":
            stack.extend(list(g.geoms))
        # else: ignore polygons etc.

    return intersection_points, overlap_lines


def _dedupe_measures(measures: list[float], *, tol_m: float) -> list[float]:
    """Deduplicate measures along the line: keep one value per cluster within tol_m."""
    if not measures:
        return []
    measures = sorted(measures)
    out = [measures[0]]
    last = measures[0]
    for m in measures[1:]:
        if (m - last) > tol_m:
            out.append(m)
            last = m
    return out


def _intervals_overlap(
    a0: float, a1: float, b0: float, b1: float, *, tol_m: float = 0.0
) -> bool:
    """True if [a0,a1] and [b0,b1] overlap with positive length (> tol_m)."""
    lo = max(min(a0, a1), min(b0, b1))
    hi = min(max(a0, a1), max(b0, b1))
    return (hi - lo) > tol_m


def _sorted_by_measure[T: BoundaryPoint](points: list[T]) -> list[T]:
    return sorted(points, key=lambda p: p.measure)


def _keep_by_measure[T: BoundaryPoint](
    points_sorted: list[T], keep_measures: set[float]
) -> list[T]:
    return [p for p in points_sorted if p.measure in keep_measures]


def _filter_points_min_length[T: BoundaryPoint](
    points: list[T], *, line_length_m: float, min_segment_len_m: float
) -> list[T]:
    """Keep marks of least min_segment_len_m apart along the line.

    This does NOT guarantee that no short segment remains.
    """
    if min_segment_len_m <= 0 or not points:
        return points

    pts = _sorted_by_measure(points)
    kept: list[float] = []
    last = 0.0

    for p in pts:
        if (p.measure - last) >= min_segment_len_m:
            kept.append(p.measure)
            last = p.measure

    # Ensure the tail segment is long enough
    while kept and (line_length_m - kept[-1]) < min_segment_len_m:
        kept.pop()

    return _keep_by_measure(pts, set(kept))


def _filter_points_drop_short_segments[T: BoundaryPoint](
    boundary_points: list[T],
    *,
    line_length_m: float,
    min_segment_len_m: float,
) -> list[T]:
    """Ensure that cutting boundaries produces no short segment.

    This is boundary-specific: boundary crossings often come in close enter/exit pairs
    around narrow features (rivers). If a short segment would be created between two
    boundary marks, this removes BOTH marks (merging that narrow feature away).
    """
    if min_segment_len_m <= 0 or not boundary_points:
        return boundary_points

    pts = _sorted_by_measure(boundary_points)
    measures = [p.measure for p in pts]

    while measures:
        marks = [0.0] + measures + [line_length_m]
        seglens = [marks[i + 1] - marks[i] for i in range(len(marks) - 1)]

        idx = next((i for i, d in enumerate(seglens) if d < min_segment_len_m), None)
        if idx is None:
            break

        if idx == 0:
            # Short segment at the start: drop the first mark
            measures.pop(0)
        elif idx == len(seglens) - 1:
            # Short segment at the end: drop the last mark
            measures.pop(-1)
        else:
            # Short internal segment between measures[idx-1] and measures[idx]:
            # drop BOTH bounding marks
            measures.pop(idx)  # right bound first
            measures.pop(idx - 1)  # then left bound

    return _keep_by_measure(pts, set(measures))


def _find_boundary_points_and_overlaps(
    line: LineString,
    boundary: BaseGeometry,
    *,
    snap_tol_m: float = 1.0,
    endpoint_exclusion_m: float = 1.0,
) -> tuple[list[BoundaryPoint], list[tuple[float, float]]]:
    """Find boundary crossings and boundary-overlap intervals along the line.

    Args:
        line (LineString): _description_
        boundary (BaseGeometry): _description_
        snap_tol_m (float, optional):
            Dedupes near-identical crossings along the line (distance along the line). Defaults to 1.0.
        endpoint_exclusion_m (float, optional):
            Drops boundary points too close to the line endpoints (defaults to snap_tol_m). Defaults to 1.0.

    Returns:
        tuple[list[BoundaryPoint], list[tuple[float, float]]]:
            boundary_points:
                Points where the line intersects the boundary (including overlap endpoints),
                expressed as measures along the line + corresponding on-line Point.
            overlap_intervals:
                Measure ranges (start_m, end_m) where the line overlaps the boundary with
                non-zero length (line runs *along* the boundary). These are used to skip
                InteriorPoints in ambiguous overlap cases.
    """
    if line.geom_type != "LineString":
        raise TypeError(f"Expected LineString, got {line.geom_type}")

    if not line.intersects(boundary):
        return [], []

    intersection = line.intersection(boundary)
    intersection_points, overlap_lines = _extract_intersection_points_and_overlap_lines(
        intersection
    )
    if not intersection_points and not overlap_lines:
        return [], []

    line_length_m = float(line.length)

    # --- BoundaryPoints (from point intersections) ---
    measures: list[float] = []
    for p in intersection_points:
        m = float(line.project(p))
        if m <= endpoint_exclusion_m or (line_length_m - m) <= endpoint_exclusion_m:
            continue
        measures.append(m)

    measures = _dedupe_measures(measures, tol_m=snap_tol_m)
    boundary_points = [
        BoundaryPoint(measure=m, point=line.interpolate(m)) for m in measures
    ]

    # --- Overlap intervals (from 1D intersections) ---
    overlap_intervals: list[tuple[float, float]] = []
    for overlap_line in overlap_lines:
        overlap_boundary = overlap_line.boundary
        if overlap_boundary.is_empty:
            continue

        if overlap_boundary.geom_type == "Point":
            endpoints = [overlap_boundary]
        elif overlap_boundary.geom_type == "MultiPoint":
            endpoints = list(overlap_boundary.geoms)
        else:
            endpoints = []

        if len(endpoints) < 2:
            continue

        start_m = float(line.project(endpoints[0]))
        end_m = float(line.project(endpoints[1]))
        if end_m < start_m:
            start_m, end_m = end_m, start_m

        if (end_m - start_m) > snap_tol_m:
            overlap_intervals.append((start_m, end_m))

    return boundary_points, overlap_intervals


def _cut_line_at_measures(line: LineString, measures: list[float]) -> list[Segment]:
    """Cut a LineString into exact substrings at measures along the line."""
    if line.geom_type != "LineString":
        raise TypeError(f"Expected LineString, got {line.geom_type}")

    line_length_m = float(line.length)
    internal_marks = sorted(m for m in measures if 0.0 < m < line_length_m)

    marks = [0.0] + internal_marks + [line_length_m]
    segments: list[Segment] = []

    for start_m, end_m in zip(marks[:-1], marks[1:]):
        seg = substring(line, start_m, end_m)
        if seg.geom_type != "LineString":
            raise RuntimeError(f"substring produced {seg.geom_type}")
        segments.append(Segment(start=float(start_m), end=float(end_m), geometry=seg))

    return segments


def _interior_points_from_boundary_points(
    line: LineString,
    boundary_points: list[BoundaryPoint],
    overlap_intervals: list[tuple[float, float]] | None = None,
    *,
    interior_ratio: float = 0.5,
    overlap_tol_m: float = 0.0,
) -> list[InteriorPoint]:
    """Create InteriorPoints between consecutive boundary crossings (skip overlap intervals).

    InteriorPoints are only created for intervals bounded by two crossings.

    overlap_intervals:
      Measure ranges where the line overlaps the boundary (1D intersection). Any
      between-crossings interval that overlaps these ranges is skipped.
    """
    if line.geom_type != "LineString":
        raise TypeError(f"Expected LineString, got {line.geom_type}")
    if not (0.0 < interior_ratio < 1.0):
        raise ValueError("interior_ratio must be in (0, 1)")

    overlap_intervals = overlap_intervals or []
    line_length_m = float(line.length)

    measures = sorted(bp.measure for bp in boundary_points)
    marks = [0.0] + measures + [line_length_m]

    interior_points: list[InteriorPoint] = []
    for i in range(1, len(marks) - 2):  # exclude endpoint-touching intervals
        start_m = float(marks[i])
        end_m = float(marks[i + 1])

        # Skip ambiguous overlap-with-boundary intervals.
        if any(
            _intervals_overlap(start_m, end_m, a, b, tol_m=overlap_tol_m)
            for a, b in overlap_intervals
        ):
            continue

        m = float(start_m + interior_ratio * (end_m - start_m))
        interior_points.append(
            InteriorPoint(
                start=start_m, end=end_m, measure=m, point=line.interpolate(m)
            )
        )

    return interior_points


def build_boundary(shapes: Iterable[BaseGeometry]) -> BaseGeometry:
    """Build a single boundary geometry from an iterable of polygonal shapes."""
    return unary_union([g.boundary for g in shapes])


def cut_line_by_boundary_points(
    line: LineString,
    boundary: BaseGeometry,
    *,
    snap_tol_m: float = 1.0,
    min_segment_len_m: float = 0.0,
    endpoint_exclusion_m: float = 1.0,
) -> tuple[list[Segment], list[BoundaryPoint]]:
    """Cut a line at boundary crossings.

    This splits `line` into exact substrings at every intersection with `boundary`.
    Crossings are detected as (line ∩ boundary) points. If the line overlaps the
    boundary for a non-zero length, the overlap endpoints are treated as crossings.

    Diagram:
        u----|-----|----v   (| = boundary crossing)
        u----x-----x----v   (x = cut locations)

    Args:
        line: LineString to split. Must be in a projected CRS (units in meters/feet).
        boundary: A single boundary geometry (e.g., unary_union(polygons.boundary)).
        snap_tol_m: Tolerance (distance along the line) used to deduplicate nearly
            identical crossing measures. This helps stabilize results under floating
            precision and complex boundary unions.
        min_segment_len_m: If > 0, drop some crossing points so that all resulting
            segments are at least this long (in line units).
        endpoint_exclusion_m: Ignore crossings whose measure is within this
            distance of either line endpoint (start/end). It's recommended to have this
            be equal to snap_tol_m.

    Returns:
        tuple[list[Segment],list[BoundaryPoint]]:
        - segments: Exact substrings of the original line; segment lengths sum to
          the original line length.
        - boundary_points: Boundary crossing locations used as cut marks, expressed
          as measures along the original line plus the corresponding on-line Point.
    """
    boundary_points, _ = _find_boundary_points_and_overlaps(
        line, boundary, snap_tol_m=snap_tol_m, endpoint_exclusion_m=endpoint_exclusion_m
    )

    boundary_points = _filter_points_drop_short_segments(
        boundary_points,
        line_length_m=float(line.length),
        min_segment_len_m=min_segment_len_m,
    )

    segments = _cut_line_at_measures(line, [bp.measure for bp in boundary_points])
    return segments, boundary_points


def cut_line_by_interior_points(
    line: LineString,
    boundary: BaseGeometry,
    *,
    snap_tol_m: float = 1.0,
    min_segment_len_m: float = 0.0,
    interior_ratio: float = 0.5,
    endpoint_exclusion_m: float = 1.0,
) -> tuple[list[Segment], list[InteriorPoint]]:
    """Cut a line at interior points between consecutive boundary crossings.

    This first finds boundary crossings (line ∩ boundary) between-crossings:

        start + interior_ratio * (end - start)

    where start/end are consecutive boundary crossing measures along the original line.

    Diagram:
        u----|-----|----v   (| = boundary crossing)
        u----|--x--|----v   (x = interior cut point)

    Args:
        line: LineString to split. Must be in a projected CRS (units in meters/feet).
        boundary: A single boundary geometry (e.g., unary_union(polygons.boundary)).
        snap_tol_m: Tolerance (distance along the line) used to deduplicate nearly
            identical boundary crossing measures.
        min_segment_len_m: If > 0, drop some interior cut points so that all resulting
            segments are at least this long (in line units).
        interior_ratio: Position of the interior cut within each between-crossings
            interval. Must be in (0, 1). Use 0.5 for midpoint.
        endpoint_exclusion_m: Ignore crossings whose measure is within this
            distance of either line endpoint (start/end). It's recommended to have this
            be equal to snap_tol_m.

    Returns:
        tuple[list[Segment],list[InteriorPoint]]:
        - segments: Exact substrings of the original line cut at interior points;
        segment lengths sum to the original line length.
        - interior_points: Interior cut locations (measures along the original line
        plus corresponding on-line Point). Only created for intervals bounded by
        two boundary crossings.

    Notes:
        - If no interior_points exist, no cuts are applied and `segments` contains a
          single Segment equal to the original line.
        - If the line overlaps the boundary for a non-zero length, the corresponding
          between-crossings interval is skipped (no InteriorPoint) to avoid ambiguous
          "on the boundary" points.
    """
    boundary_points, overlap_intervals = _find_boundary_points_and_overlaps(
        line, boundary, snap_tol_m=snap_tol_m, endpoint_exclusion_m=endpoint_exclusion_m
    )

    boundary_points = _filter_points_drop_short_segments(
        boundary_points,
        line_length_m=float(line.length),
        min_segment_len_m=min_segment_len_m,
    )

    interior_points = _interior_points_from_boundary_points(
        line, boundary_points, overlap_intervals, interior_ratio=interior_ratio
    )

    interior_points = _filter_points_min_length(
        interior_points,
        line_length_m=float(line.length),
        min_segment_len_m=min_segment_len_m,
    )

    segments = _cut_line_at_measures(line, [ip.measure for ip in interior_points])
    return segments, interior_points
