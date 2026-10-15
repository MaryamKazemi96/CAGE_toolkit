"""OptiTrack CSV readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TakeHeader:
    take_name: str
    capture_frame_rate: float
    export_frame_rate: float
    total_frames: int
    rotation_type: str
    length_units: str
    coordinate_space: str


def read_header(path: Path) -> TakeHeader:
    """Read the take metadata row."""
    with path.open(encoding="utf-8") as stream:
        cells = stream.readline().rstrip("\n").split(",")
    values = dict(zip(cells[::2], cells[1::2]))
    required = ("Take Name", "Capture Frame Rate", "Export Frame Rate", "Total Frames in Take",
                "Rotation Type", "Length Units", "Coordinate Space")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"{path} is not a supported OptiTrack CSV; missing {missing}")
    return TakeHeader(
        take_name=values["Take Name"], capture_frame_rate=float(values["Capture Frame Rate"]),
        export_frame_rate=float(values["Export Frame Rate"]), total_frames=int(values["Total Frames in Take"]),
        rotation_type=values["Rotation Type"], length_units=values["Length Units"],
        coordinate_space=values["Coordinate Space"],
    )


def column_names(path: Path) -> list[str]:
    """Return toolkit field names."""
    header = pd.read_csv(path, skiprows=2, nrows=5, header=None, low_memory=False)
    names: list[str] = []
    for column in range(header.shape[1]):
        if column == 0:
            names.append("Frame")
        elif column == 1:
            names.append("Time")
        else:
            body = str(header.iloc[1, column]).strip().replace(" ", "").replace(":", ".")
            kind = str(header.iloc[3, column]).strip().replace(" ", "_").lower()
            axis = header.iloc[4, column]
            names.append(f"{body}.{kind}" if pd.isna(axis) else f"{body}.{kind}.{str(axis).lower()}")
    return names


def read_csv(path: Path, *, normalize_horizontal: bool = True) -> tuple[TakeHeader, pd.DataFrame]:
    """Load one take and expose horizontal coordinates as position.x/y."""
    header = read_header(path)
    data = pd.read_csv(path, skiprows=7, header=None, names=column_names(path), low_memory=False)
    for name in data.columns:
        try:
            data[name] = pd.to_numeric(data[name])
        except (ValueError, TypeError):
            pass
    if normalize_horizontal:
        for name in list(data.columns):
            if name.endswith(".position.y"):
                vertical = name
                horizontal_y = name[:-1] + "z"
                if horizontal_y in data:
                    data[vertical], data[horizontal_y] = data[horizontal_y].copy(), data[vertical].copy()
    return header, data


def read_body_positions(path: Path, bodies: list[str]) -> pd.DataFrame:
    """Read the columns needed for indexing."""
    names = column_names(path)
    required = ["Frame", "Time"]
    for body in bodies:
        required.extend([f"{body}.position.x", f"{body}.position.y", f"{body}.position.z"])
    indexes = [index for index, name in enumerate(names) if name in required]
    data = pd.read_csv(path, skiprows=7, header=None, usecols=indexes, low_memory=False)
    data.columns = [names[index] for index in indexes]
    for name in data.columns:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    # Source y is vertical; use source z for horizontal y.
    for body in bodies:
        vertical, horizontal_y = f"{body}.position.y", f"{body}.position.z"
        if vertical in data and horizontal_y in data:
            data[vertical], data[horizontal_y] = data[horizontal_y].copy(), data[vertical].copy()
    return data


def body_position_columns(body: str) -> list[str]:
    return [f"{body}.position.x", f"{body}.position.y"]
