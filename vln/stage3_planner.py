"""Stage 3: symbolic search planner -- boustrophedon (lawn-mower) sweep + A* detours.

Pure python/numpy (no simulator, no ML) -> fully unit-testable. All coordinates are R-frame (n, e).
"""
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import CameraCfg, GeometryCfg

log = logging.getLogger(__name__)
Pt = Tuple[float, float]


class PlanningError(RuntimeError):
    pass


@dataclass(frozen=True)
class Rect:
    n_min: float
    e_min: float
    n_max: float
    e_max: float

    def inflate(self, m: float) -> "Rect":
        return Rect(self.n_min - m, self.e_min - m, self.n_max + m, self.e_max + m)

    def contains(self, p: Pt) -> bool:
        return self.n_min <= p[0] <= self.n_max and self.e_min <= p[1] <= self.e_max


def footprint(altitude_m: float, cam: CameraCfg) -> Tuple[float, float]:
    """Ground footprint (across-track width = image width, along-track height = image height)."""
    w = 2.0 * altitude_m * math.tan(math.radians(cam.hfov_deg) / 2.0)
    return w, w * cam.height / cam.width


def search_rect(geom: GeometryCfg) -> Rect:
    """The 40x40 m search square in the R frame (offset from R by geom.area_center_ne)."""
    n0, e0, n1, e1 = geom.area_bounds()
    return Rect(n0, e0, n1, e1)


def _covered(path: Sequence[Pt], q: Pt, half_n: float, half_e: float) -> bool:
    """Exact test: does a north-up camera footprint (half extents half_n x half_e) flown along the polyline
    `path` ever contain point q? Per segment the two box constraints are linear in t -> interval intersection."""
    for a, b in zip(path, path[1:]):
        lo, hi = 0.0, 1.0
        for pa, pb, qq, half in ((a[0], b[0], q[0], half_n), (a[1], b[1], q[1], half_e)):
            d = pb - pa
            if abs(d) < 1e-12:
                if abs(qq - pa) > half:
                    lo, hi = 1.0, 0.0
                    break
                continue
            t0, t1 = sorted(((qq - half - pa) / d, (qq + half - pa) / d))
            lo, hi = max(lo, t0), min(hi, t1)
        if lo <= hi:
            return True
    return False


def coverage_margin(waypoints: Sequence[Pt], start: Pt, point: Pt, altitude_m: float, cam: CameraCfg) -> float:
    """Robustness of the sweep for `point` (R frame): the largest position error m (metres, per axis) the UAV
    could have while the point is still inside the camera footprint at some moment (capped at half the
    along-track footprint, ~5.6 m at 10 m altitude). -1.0 => not covered at all.
    Used for config sanity checks / logging only (the pipeline never feeds ground truth into planning)."""
    w, h = footprint(altitude_m, cam)
    path = [start] + list(waypoints)
    if not _covered(path, point, h / 2.0, w / 2.0):
        return -1.0
    lo, hi = 0.0, min(w, h) / 2.0
    if _covered(path, point, h / 2.0 - hi, w / 2.0 - hi):
        return hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _covered(path, point, h / 2.0 - mid, w / 2.0 - mid) else (lo, mid)
    return lo


def boustrophedon(area: Rect, swath_w: float, overlap: float, start: Pt) -> List[Pt]:
    """N-S legs stacked along east. Image width (east, since yaw is locked north) is the swath."""
    width = area.e_max - area.e_min
    if width <= swath_w:
        lanes = [0.5 * (area.e_min + area.e_max)]
    else:
        step_max = swath_w * (1.0 - overlap)
        k = math.ceil((width - swath_w) / step_max) + 1
        lanes = list(np.linspace(area.e_min + swath_w / 2, area.e_max - swath_w / 2, k))

    def first_cost(order: Sequence[float]) -> float:
        e0 = order[0]
        return min(math.hypot(start[0] - area.n_min, start[1] - e0),
                   math.hypot(start[0] - area.n_max, start[1] - e0))

    if first_cost(lanes[::-1]) < first_cost(lanes):
        lanes = lanes[::-1]
    north_first = (area.n_min
                   if abs(start[0] - area.n_min) <= abs(start[0] - area.n_max) else area.n_max)
    wps: List[Pt] = []
    n_a, n_b = north_first, (area.n_max if north_first == area.n_min else area.n_min)
    for i, e in enumerate(lanes):
        a, b = (n_a, n_b) if i % 2 == 0 else (n_b, n_a)
        wps += [(a, float(e)), (b, float(e))]
    return wps


class OccupancyGrid:
    def __init__(self, bounds: Rect, res: float, boxes: Sequence[Tuple[float, float, float, float]],
                 margin: float):
        self.b, self.res = bounds, res
        self.rows = int(math.ceil((bounds.n_max - bounds.n_min) / res)) + 1
        self.cols = int(math.ceil((bounds.e_max - bounds.e_min) / res)) + 1
        self.occ = np.zeros((self.rows, self.cols), dtype=bool)
        for (n0, e0, n1, e1) in boxes:
            r = Rect(n0, e0, n1, e1).inflate(margin)
            for i in range(self.rows):
                for j in range(self.cols):
                    if r.contains(self.to_pt((i, j))):
                        self.occ[i, j] = True

    def to_cell(self, p: Pt) -> Tuple[int, int]:
        i = int(round((p[0] - self.b.n_min) / self.res))
        j = int(round((p[1] - self.b.e_min) / self.res))
        return min(max(i, 0), self.rows - 1), min(max(j, 0), self.cols - 1)

    def to_pt(self, c: Tuple[int, int]) -> Pt:
        return self.b.n_min + c[0] * self.res, self.b.e_min + c[1] * self.res

    def free_pt(self, p: Pt) -> bool:
        return self.b.contains(p) and not self.occ[self.to_cell(p)]

    def segment_free(self, a: Pt, b: Pt) -> bool:
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        k = max(2, int(L / (self.res * 0.5)) + 1)
        return all(self.free_pt((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
                   for t in np.linspace(0.0, 1.0, k))


def astar(grid: OccupancyGrid, a: Pt, b: Pt) -> List[Pt]:
    s, g = grid.to_cell(a), grid.to_cell(b)
    if grid.occ[s] or grid.occ[g]:
        raise PlanningError(f"A* endpoint inside a no-fly zone: {a} -> {b}")
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

    def h(c):  # octile distance
        dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    openq: List = [(h(s), 0.0, s)]
    came: Dict = {}
    gs = {s: 0.0}
    while openq:
        _, cost, cur = heapq.heappop(openq)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return [grid.to_pt(c) for c in reversed(path)]
        if cost > gs.get(cur, 1e18):
            continue
        for di, dj, w in moves:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < grid.rows and 0 <= nb[1] < grid.cols) or grid.occ[nb]:
                continue
            if di and dj and (grid.occ[cur[0] + di, cur[1]] or grid.occ[cur[0], cur[1] + dj]):
                continue  # no corner cutting
            nc = cost + w
            if nc < gs.get(nb, 1e18):
                gs[nb] = nc
                came[nb] = cur
                heapq.heappush(openq, (nc + h(nb), nc, nb))
    raise PlanningError(f"A* found no path {a} -> {b}")


def simplify(grid: OccupancyGrid, path: List[Pt]) -> List[Pt]:
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not grid.segment_free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plan_search(geom: GeometryCfg, cam: CameraCfg) -> List[Pt]:
    """Return the ordered R-frame waypoints to fly (start position excluded)."""
    area = search_rect(geom)
    log.info("Search area (R frame): N [%.1f, %.1f]  E [%.1f, %.1f]", area.n_min, area.n_max, area.e_min, area.e_max)
    swath_w, swath_h = footprint(geom.altitude_m, cam)
    log.info("Footprint at %.1f m: %.1f x %.1f m", geom.altitude_m, swath_w, swath_h)

    raw = boustrophedon(area, swath_w, geom.lane_overlap, geom.start_offset_ne)
    grid = OccupancyGrid(area.inflate(2 * geom.safety_margin_m), geom.grid_res_m,
                         geom.no_fly_boxes, geom.safety_margin_m)
    outside = not grid.b.contains(geom.start_offset_ne)
    if outside:
        transit = math.hypot(raw[0][0] - geom.start_offset_ne[0], raw[0][1] - geom.start_offset_ne[1])
        log.warning("UAV start (N %.1f, E %.1f in R frame) is outside the search area: the %.0f m transit leg to the "
                    "first lane is a straight line and is NOT obstacle-checked (trees/houses!). Dead-reckoning "
                    "drift also accumulates over this leg.", geom.start_offset_ne[0], geom.start_offset_ne[1],
                    transit)
    elif not grid.free_pt(geom.start_offset_ne):
        raise PlanningError("UAV start position lies inside a no-fly zone.")

    kept = [w for w in raw if grid.free_pt(w)]
    if len(kept) < len(raw):
        log.warning("Dropped %d lane endpoints that fall inside no-fly zones.", len(raw) - len(kept))

    plan: List[Pt] = []
    cur = geom.start_offset_ne
    for w in kept:
        if outside and cur == geom.start_offset_ne:
            plan.append(w)
        elif grid.segment_free(cur, w):
            plan.append(w)
        else:
            detour = simplify(grid, astar(grid, cur, w))
            plan += detour[1:]
        cur = w
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                 for a, b in zip([geom.start_offset_ne] + plan, plan))
    log.info("Plan: %d waypoints, %.0f m (~%.0f s at %.1f m/s)", len(plan), length,
             length / geom.cruise_speed_mps, geom.cruise_speed_mps)
    return plan
