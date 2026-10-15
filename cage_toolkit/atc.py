"""Export CAGE pedestrian tracks in the ATC CSV layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ATC_COLUMNS = [
    "time", "person_id", "x", "y", "z", "velocity", "motion_angle", "facing_angle",
]


def export_atc(recording, output: Path) -> dict[str, object]:
    """Write one solved or raw CAGE recording as a headerless ATC-style CSV."""
    goal = recording.spec["temporal_alignment"]["goal_to_optitrack"]
    if not goal.get("accepted") or not goal.get("corrected_optitrack_start"):
        raise ValueError(f"S{recording.session}/scenario {recording.scenario} has no finalized wall-clock start")

    start_epoch_s = recording.goal_timestamp_for_elapsed(0).timestamp()
    data = recording.trimmed()
    time_s = data["Time"].to_numpy(float)
    rows: list[pd.DataFrame] = []
    for body in recording.humans:
        person_id = int(body.removeprefix("RigidBody"))
        xyz_m = data[[f"{body}.position.x", f"{body}.position.y", f"{body}.position.z"]].to_numpy(float)
        present = np.isfinite(xyz_m).all(axis=1)
        speed_m_s = np.full(len(data), np.nan)
        motion_angle = np.full(len(data), np.nan)
        if len(data) > 1:
            dt = np.diff(time_s)
            delta_xy = np.diff(xyz_m[:, :2], axis=0)
            paired = present[:-1] & present[1:] & np.isfinite(dt) & (dt > 0)
            indexes = np.flatnonzero(paired) + 1
            speed_m_s[indexes] = np.linalg.norm(delta_xy[paired], axis=1) / dt[paired]
            motion_angle[indexes] = np.arctan2(delta_xy[paired, 1], delta_xy[paired, 0])
        selected = np.flatnonzero(present)
        if len(selected) == 0:
            continue
        rows.append(pd.DataFrame({
            "time": np.round(start_epoch_s + time_s[selected], 3),
            "person_id": person_id,
            "x": np.rint(xyz_m[selected, 0] * 1000.0).astype(int),
            "y": np.rint(xyz_m[selected, 1] * 1000.0).astype(int),
            "z": xyz_m[selected, 2] * 1000.0,
            "velocity": speed_m_s[selected] * 1000.0,
            "motion_angle": motion_angle[selected],
            # Hat orientation is not a validated body-facing measurement.
            "facing_angle": np.nan,
        }))
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=ATC_COLUMNS)
    result = result.sort_values(["time", "person_id"], kind="stable")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, header=False, columns=ATC_COLUMNS, float_format="%.6f", na_rep="")

    metadata = {
        "format": "ATC pedestrian tracking CSV layout",
        "columns": ATC_COLUMNS,
        "recording": {"session": recording.session, "scenario": recording.scenario, "source": recording.source},
        "rows": int(len(result)),
        "person_ids": sorted(int(value) for value in result.person_id.unique()),
        "time": "Unix epoch seconds with millisecond precision, based on the finalized OptiTrack recording start plus OptiTrack elapsed time",
        "position": "CAGE OptiTrack horizontal x/y and vertical z in millimetres; x/y are rounded to integer millimetres as in ATC; not aligned to the ATC venue",
        "velocity": "Horizontal finite difference in mm/s between adjacent valid OptiTrack samples",
        "motion_angle": "atan2(dy, dx) in radians for the same finite difference",
        "facing_angle": "Blank: CAGE hat orientation is not validated as pedestrian body-facing direction",
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
