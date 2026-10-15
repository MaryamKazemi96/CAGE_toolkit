"""ROS map to OptiTrack conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MapToOptiTrackTransform:
    matrix: np.ndarray
    translation_m: np.ndarray
    scale: float
    reflection: bool

    @classmethod
    def from_json(cls, path: Path) -> "MapToOptiTrackTransform":
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        return cls(np.asarray(value["matrix"], dtype=float), np.asarray(value["translation_m"], dtype=float),
                   float(value["scale"]), bool(value["reflection"]))

    def apply(self, xy: np.ndarray) -> np.ndarray:
        points = np.asarray(xy, dtype=float)
        if points.shape[-1] != 2:
            raise ValueError("Expected (..., 2) map coordinates")
        return self.scale * (points @ self.matrix.T) + self.translation_m
