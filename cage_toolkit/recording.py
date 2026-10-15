"""Recording API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .metadata import load_metadata, recording_key
from .optitrack import TakeHeader, read_csv
from .spatial import MapToOptiTrackTransform


@dataclass
class Recording:
    dataset: "CAGEDataset"
    spec: dict
    source: str
    header: TakeHeader
    data: pd.DataFrame

    @property
    def session(self) -> int:
        return int(self.spec["session"])

    @property
    def scenario(self) -> int:
        return int(self.spec["scenario"])

    @property
    def humans(self) -> list[str]:
        return list(self.spec["humans"])

    @property
    def robots(self) -> list[str]:
        return list(self.spec["robots"])

    @property
    def bodies(self) -> list[str]:
        return self.humans + self.robots

    @property
    def duration_s(self) -> float:
        return float(self.spec["sources"][self.source]["duration_s"])

    def bounds(self, body: str) -> dict:
        # Reuse saved visibility bounds.
        return self.spec.get("analysis_body_bounds", self.spec["sources"][self.source].get("body_bounds", {})).get(body, {})

    def trimmed(self) -> pd.DataFrame:
        """Return data trimmed to saved bounds."""
        result = self.data.copy()
        for body in self.bodies:
            bound = self.bounds(body)
            if not bound:
                raise ValueError("This metadata index has no cached analysis bounds. Rebuild it without --without-bounds.")
            start, end = bound.get("start_frame"), bound.get("end_frame")
            columns = [name for name in result if name.startswith(body + ".")]
            if not columns:
                continue
            if start is None or end is None:
                result.loc[:, columns] = float("nan")
            else:
                result.loc[(result.Frame < start) | (result.Frame > end), columns] = float("nan")
        return result

    def goal_timestamp_for_elapsed(self, elapsed_s: float) -> pd.Timestamp:
        goal = self.spec["temporal_alignment"]["goal_to_optitrack"]
        if not goal.get("accepted") or not goal.get("corrected_optitrack_start"):
            raise ValueError(f"S{self.session}/scenario {self.scenario} has no accepted goal/OptiTrack alignment")
        start = pd.Timestamp(goal["corrected_optitrack_start"])
        return start + pd.to_timedelta(elapsed_s, unit="s")

class CAGEDataset:
    """Dataset rooted at the project directory."""

    def __init__(self, root: str | Path = ".", metadata_path: str | Path | None = None):
        self.root = Path(root).resolve()
        supplied = Path(metadata_path) if metadata_path else None
        if supplied is not None and not supplied.is_absolute():
            supplied = self.root / supplied
        self.metadata = load_metadata(self.root, supplied)

    def recording(self, session: int, scenario: int, source: str = "solved") -> Recording:
        if source not in {"raw", "solved"}:
            raise ValueError("source must be 'raw' or 'solved'")
        key = recording_key(session, scenario)
        try:
            spec = self.metadata["recordings"][key]
        except KeyError as error:
            raise KeyError(f"No recording S{session}, scenario {scenario}") from error
        file = self.root / spec["sources"][source]["path"]
        header, data = read_csv(file)
        return Recording(self, spec, source, header, data)

    def station_regions(self) -> dict:
        return self.metadata["map"].get("station_regions", {})

    def spatial_transform(self) -> MapToOptiTrackTransform:
        return MapToOptiTrackTransform.from_json(self.root / self.metadata["spatial_alignment"]["transform_path"])
