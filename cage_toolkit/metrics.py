"""Trajectory metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .recording import Recording


def _trajectory(data: pd.DataFrame, body: str) -> tuple[np.ndarray, np.ndarray]:
    values = data[["Time", f"{body}.position.x", f"{body}.position.y"]].to_numpy(float)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) == 0:
        return np.empty(0), np.empty((0, 2))
    return values[:, 0], values[:, 1:]


def recording_metrics(recording: Recording) -> pd.DataFrame:
    """Return per-person movement metrics."""
    data = recording.trimmed()
    rows = []
    for body in recording.humans:
        time, xy = _trajectory(data, body)
        row = {"session": recording.session, "scenario": recording.scenario, "source": recording.source, "body": body,
               "tracking_duration_s": float(time[-1] - time[0]) if len(time) > 1 else np.nan,
               "mean_speed_mps": np.nan, "path_length_m": np.nan, "mean_acceleration_mps2": np.nan}
        if len(time) > 2:
            dt = np.diff(time)
            distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            valid = dt > 0
            row["path_length_m"] = float(distance[valid].sum())
            speed = distance[valid] / dt[valid]
            row["mean_speed_mps"] = float(np.mean(speed)) if len(speed) else np.nan
            if len(speed) > 2:
                speed_time = time[1:][valid]
                speed_dt = np.diff(speed_time)
                acceleration = np.diff(speed) / speed_dt
                row["mean_acceleration_mps2"] = float(np.mean(np.abs(acceleration[np.isfinite(acceleration)])))
        rows.append(row)
    return pd.DataFrame(rows)
