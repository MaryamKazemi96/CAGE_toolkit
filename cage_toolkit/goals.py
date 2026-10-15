"""Goal-log helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_goal_events(path: Path) -> pd.DataFrame:
    """Load a goal log and add a Helsinki epoch timestamp."""
    events = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "hat_id", "to_station"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"{path} is missing required goal fields: {sorted(missing)}")
    timestamps = pd.DatetimeIndex(events["timestamp"])
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("Europe/Helsinki")
    else:
        timestamps = timestamps.tz_convert("Europe/Helsinki")
    events = events.copy()
    events["timestamp_epoch_s"] = timestamps.asi8 / 1e9
    return events.sort_values("timestamp_epoch_s").reset_index(drop=True)


def assigned_station(events: pd.DataFrame, hat_id: int, epoch_s: float) -> str | None:
    """Return the latest assigned destination."""
    person = events[(events.hat_id == hat_id) & (events.timestamp_epoch_s <= epoch_s)]
    return None if person.empty else str(person.iloc[-1].to_station)
