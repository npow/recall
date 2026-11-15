"""Shadow Canvas bookkeeping: world-anchored tiles and their LIVE/SHADOWED state.

Pure geometry with no model dependency. ``Canvas.update`` reports the events an
orchestrator must apply to the world model: create a tile, fork a tile's shadow
when the camera leaves it, and hand a shadow back when the camera returns.
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from worldsync.poses import DEFAULT_INTRINSICS, INTRINSICS_SIZE


class TileState(str, Enum):
    LIVE = "live"
    SHADOWED = "shadowed"


@dataclass
class Tile:
    id: int
    pose: np.ndarray
    created_chunk: int
    state: TileState = TileState.LIVE
    history: list[tuple[int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Event:
    kind: str  # "create" | "fork" | "handoff"
    tile: int
    chunk: int
    visibility: float


@dataclass(frozen=True)
class CanvasConfig:
    create_below: float = 0.6   # create a tile when existing tiles cover less of the view
    leave_below: float = 0.1    # LIVE -> SHADOWED when this little of a tile is visible
    enter_above: float = 0.2    # SHADOWED -> LIVE once this much of a tile is visible
    depth: float = 8.0          # assumed scene depth (world units) for parallax-aware overlap
    grid: tuple[int, int] = (9, 16)

    def __post_init__(self):
        if not 0 <= self.leave_below < self.enter_above <= 1:
            raise ValueError("require 0 <= leave_below < enter_above <= 1 for hysteresis")


class Camera:
    """Pinhole camera used for overlap tests, in LingBot's OpenCV convention."""

    def __init__(self, intrinsics: np.ndarray = DEFAULT_INTRINSICS, size: tuple[int, int] = INTRINSICS_SIZE,
                 grid: tuple[int, int] = (9, 16)):
        self.fx, self.fy, self.cx, self.cy = intrinsics
        self.height, self.width = size
        ys = (np.arange(grid[0]) + 0.5) * self.height / grid[0]
        xs = (np.arange(grid[1]) + 0.5) * self.width / grid[1]
        u, v = np.meshgrid(xs, ys)
        self.rays = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], -1).reshape(-1, 3)

    def sample_points(self, pose: np.ndarray, depth: float) -> np.ndarray:
        """World points seen by ``pose`` at a fixed depth along each sampled ray."""
        return self.rays * depth @ pose[:3, :3].T + pose[:3, 3]

    def sees(self, pose: np.ndarray, points: np.ndarray, depth_range=(1e-6, np.inf)) -> np.ndarray:
        """Points inside the image whose depth lies in ``depth_range``."""
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        z = local[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * local[:, 0] / z + self.cx
            v = self.fy * local[:, 1] / z + self.cy
        return (z >= depth_range[0]) & (z <= depth_range[1]) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)


class Canvas:

    def __init__(self, config: CanvasConfig = CanvasConfig(), camera: Camera | None = None):
        self.config = config
        self.camera = camera or Camera(grid=config.grid)
        self.tiles: list[Tile] = []

    @property
    def depth_range(self) -> tuple[float, float]:
        # A tile holds surfaces near the assumed depth; much nearer or farther is new content.
        return self.config.depth / 2, self.config.depth * 2

    def visibility(self, tile_pose: np.ndarray, view_pose: np.ndarray) -> float:
        """Fraction of a tile's view that ``view_pose`` can see."""
        points = self.camera.sample_points(tile_pose, self.config.depth)
        return float(self.camera.sees(view_pose, points, self.depth_range).mean())

    def coverage(self, view_pose: np.ndarray) -> float:
        """Fraction of ``view_pose``'s view already seen by any tile."""
        if not self.tiles:
            return 0.0
        points = self.camera.sample_points(view_pose, self.config.depth)
        covered = np.zeros(len(points), dtype=bool)
        for tile in self.tiles:
            covered |= self.camera.sees(tile.pose, points, self.depth_range)
        return float(covered.mean())

    def update(self, view_pose: np.ndarray, chunk: int) -> list[Event]:
        """Advance tile states for the camera at ``view_pose``; return events in order."""
        events = []
        for tile in self.tiles:
            seen = self.visibility(tile.pose, view_pose)
            if tile.state is TileState.LIVE and seen < self.config.leave_below:
                events.append(self._transition(tile, TileState.SHADOWED, "fork", chunk, seen))
            elif tile.state is TileState.SHADOWED and seen >= self.config.enter_above:
                events.append(self._transition(tile, TileState.LIVE, "handoff", chunk, seen))
        if self.coverage(view_pose) < self.config.create_below:
            tile = Tile(len(self.tiles), np.array(view_pose, dtype=float), chunk)
            tile.history.append((chunk, "create"))
            self.tiles.append(tile)
            events.append(Event("create", tile.id, chunk, 1.0))
        return events

    @staticmethod
    def _transition(tile: Tile, state: TileState, kind: str, chunk: int, seen: float) -> Event:
        tile.state = state
        tile.history.append((chunk, kind))
        return Event(kind, tile.id, chunk, seen)

    def shadowed(self) -> list[Tile]:
        return [t for t in self.tiles if t.state is TileState.SHADOWED]

    def summary(self) -> list[dict]:
        return [{"id": t.id, "state": t.state.value, "created_chunk": t.created_chunk,
                 "position": t.pose[:3, 3].round(3).tolist(),
                 "forward": t.pose[:3, 2].round(3).tolist(), "history": t.history} for t in self.tiles]
