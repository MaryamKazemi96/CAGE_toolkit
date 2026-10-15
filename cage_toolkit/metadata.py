"""Build and read recording metadata."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .optitrack import body_position_columns, read_body_positions, read_header


METADATA_VERSION = 2
DEFAULT_METADATA = Path("toolkit/config/recording_metadata.yaml")


def normalise_body(name: str) -> str:
    return name.replace(" ", "")


def recording_key(session: int, scenario: int) -> str:
    return f"s{session}_scenario{scenario}"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _relative_or_external(path: str | Path, root: Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(candidate)


def _find_take(root: Path, source: str, session: int, scenario: int) -> Path:
    matches = sorted((root / "data" / "OptiTrack" / source / f"session {session}").glob(f"*-{scenario}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {source} CSV for session={session}, scenario={scenario}; found {len(matches)}")
    return matches[0]


def _bounds(path: Path, bodies: list[str], waiting_area_x: float) -> dict[str, dict[str, float | int | None]]:
    """Find visible trajectory bounds."""
    table = read_body_positions(path, bodies)
    frames = table["Frame"].to_numpy()
    times = table["Time"].to_numpy(float)
    result: dict[str, dict[str, float | int | None]] = {}
    for body in bodies:
        columns = body_position_columns(body)
        if any(column not in table for column in columns):
            result[body] = {"start_frame": None, "end_frame": None, "start_s": None, "end_s": None}
            continue
        xy = table[columns].to_numpy(float)
        # Ignore the waiting area.
        present = xy[:, 0] > waiting_area_x
        present &= ~pd.isna(xy).any(axis=1)
        # Ignore short tracking runs.
        changes = pd.Series(present).ne(pd.Series(present).shift()).cumsum()
        intervals = []
        for _, group in pd.Series(present).groupby(changes):
            index = group.index.to_numpy()
            if bool(group.iloc[0]) and len(index) >= 30:
                intervals.append(index)
        if not intervals:
            result[body] = {"start_frame": None, "end_frame": None, "start_s": None, "end_s": None}
        else:
            first, last = intervals[0][0], intervals[-1][-1]
            result[body] = {"start_frame": int(frames[first]), "end_frame": int(frames[last]),
                            "start_s": float(times[first]), "end_s": float(times[last])}
    return result


def build_metadata(root: Path, output: Path, *, include_bounds: bool = True) -> dict[str, Any]:
    """Create a recording index."""
    root = root.resolve()
    config = read_yaml(root / "config" / "recordings.yaml")
    sync = pd.read_csv(root / "results" / "synchronization" / "per_recording_sync_summary.csv")
    goals = pd.read_csv(root / "results" / "synchronization" / "goal_alignment_accepted.csv")
    stations_path = root / "results" / "semantic_map" / "station_regions.csv"
    stations = pd.read_csv(stations_path) if stations_path.exists() else pd.DataFrame()
    records: dict[str, Any] = {}
    for session_name, session_cfg in config["sessions"].items():
        session = int(str(session_name).split("_")[-1])
        for scenario_value, scenario_cfg in session_cfg["scenarios"].items():
            scenario = int(scenario_value)
            bodies = [normalise_body(value) for value in scenario_cfg["humans"] + scenario_cfg.get("robots", [])]
            sources: dict[str, Any] = {}
            for source in ("raw", "solved"):
                file = _find_take(root, source, session, scenario)
                take = read_header(file)
                payload: dict[str, Any] = {
                    "path": str(file.relative_to(root)), "take_name": take.take_name,
                    "capture_fps": take.capture_frame_rate, "export_fps": take.export_frame_rate,
                    "total_frames": take.total_frames, "duration_s": (take.total_frames - 1) / take.export_frame_rate,
                    "coordinate_space": take.coordinate_space, "length_units": take.length_units,
                }
                if include_bounds:
                    payload["body_bounds"] = _bounds(file, bodies, float(config["preprocessing"]["waiting_area_x"]))
                sources[source] = payload
            synced = sync[(sync.session == session) & (sync.scenario == scenario)]
            if len(synced) != 1:
                raise ValueError(f"No unique synchronization summary for S{session} scenario {scenario}")
            row = synced.iloc[0]
            goal_rows = goals[(goals.session == session) & (goals.scenario == scenario)]
            if len(goal_rows) > 1:
                raise ValueError(f"More than one accepted goal alignment for S{session} scenario {scenario}")
            temporal: dict[str, Any] = {
                "goal_to_optitrack": {"accepted": bool(row.goal_accepted),
                                       "corrected_optitrack_start": None if pd.isna(row.corrected_optitrack_start) else str(row.corrected_optitrack_start)},
            }
            if len(goal_rows) == 1:
                temporal["goal_to_optitrack"]["goal_file"] = _relative_or_external(goal_rows.iloc[0].goal_file, root)
            records[recording_key(session, scenario)] = {
                "session": session, "scenario": scenario, "density": str(session_cfg["density"]),
                "humans": [normalise_body(value) for value in scenario_cfg["humans"]],
                "robots": [normalise_body(value) for value in scenario_cfg.get("robots", [])],
                # Keep raw visibility bounds for both sources.
                "analysis_body_bounds": deepcopy(sources["raw"].get("body_bounds", {})),
                "analysis_bounds_provenance": "raw OptiTrack visibility, x > waiting_area_x, runs >= 30 frames",
                "sources": sources, "temporal_alignment": temporal,
            }
    station_payload = {row.station: {"x_m": float(row.x_m), "y_m": float(row.y_m),
                                      "p90_radius_m": float(row.p90_click_region_radius_m), "n_clicks": int(row.n_clicks)}
                       for row in stations.itertuples(index=False)}
    payload = {
        "schema_version": METADATA_VERSION,
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "coordinate_convention": {"horizontal_axes": "OptiTrack x/y after original y/z swap", "vertical_axis": "z"},
        "map": {"semantic_image": "data/figures/env6_semantic.png", "extent_m": [-4.6, 6.282, -3.192, 3.058],
                "station_regions": station_payload},
        "spatial_alignment": {"transform_path": "results/spatial_alignment/ros_map_to_optitrack_transform.json",
                              "validation_path": "results/spatial_alignment/global_transform_validation.csv"},
        "recordings": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
    return payload


def load_metadata(root: Path, path: Path | None = None) -> dict[str, Any]:
    metadata_path = path or root / DEFAULT_METADATA
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata index not found: {metadata_path}. Run `cage metadata build` first.")
    metadata = read_yaml(metadata_path)
    if metadata.get("schema_version") != METADATA_VERSION:
        raise ValueError(f"Unsupported metadata schema in {metadata_path}")
    return metadata
