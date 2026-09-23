from __future__ import annotations

from math import hypot
from typing import Iterable, Sequence

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def center(box: BBox) -> Point:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def box_iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_polygon(point: Point, polygon: Sequence[Sequence[float]]) -> bool:
    """Ray-casting test. Empty polygons intentionally match nothing."""
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def side_of_line(point: Point, line: Sequence[Sequence[float]]) -> float:
    if len(line) != 2:
        return 0.0
    (x1, y1), (x2, y2) = line
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def crossed_line(a: Point, b: Point, line: Sequence[Sequence[float]]) -> bool:
    if len(line) != 2:
        return False
    sa, sb = side_of_line(a, line), side_of_line(b, line)
    return sa != 0.0 and sb != 0.0 and (sa > 0) != (sb > 0)


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def normalize_point(point: Point, width: int, height: int) -> Point:
    return point[0] / max(width, 1), point[1] / max(height, 1)


def merge_intervals(
    intervals: Iterable[tuple[float, float]], gap: float = 0.75, minimum: float = 0.0
) -> list[tuple[float, float]]:
    ordered = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not ordered:
        return []
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if e - s >= minimum]
