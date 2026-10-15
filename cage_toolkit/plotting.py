"""Plotting helpers."""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .goals import assigned_station, load_goal_events
from .recording import Recording


def _draw_map(recording: Recording, axis: plt.Axes, *, semantic_map: bool = True) -> None:
    meta = recording.dataset.metadata["map"]
    image = recording.dataset.root / (meta["semantic_image"] if semantic_map else "data/figures/env6.png")
    extent = meta["extent_m"]
    if image.exists():
        axis.imshow(plt.imread(image), extent=extent, origin="upper", alpha=0.55, zorder=0)
    if semantic_map:
        for name, station in recording.dataset.station_regions().items():
            axis.scatter(station["x_m"], station["y_m"], marker="X", color="black", s=42, zorder=5)
            axis.text(station["x_m"] + 0.08, station["y_m"] + 0.08, name, weight="bold", fontsize=8, zorder=5)
    axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel="OptiTrack x [m]", ylabel="OptiTrack y [m]")


def draw_trajectory(recording: Recording, axis: plt.Axes, *, trim: bool = True, semantic_map: bool = True,
                    legend: bool = True) -> None:
    data = recording.trimmed() if trim else recording.data
    _draw_map(recording, axis, semantic_map=semantic_map)
    colors = plt.cm.tab10.colors
    for index, body in enumerate(recording.humans):
        xy = data[[f"{body}.position.x", f"{body}.position.y"]].to_numpy(float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy):
            axis.plot(xy[:, 0], xy[:, 1], color=colors[index % len(colors)], lw=1.15, label=body.replace("RigidBody", "Hat "))
    for index, body in enumerate(recording.robots):
        xy = data[[f"{body}.position.x", f"{body}.position.y"]].to_numpy(float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy):
            axis.plot(xy[:, 0], xy[:, 1], color="black" if index == 0 else "tab:red", ls="--", lw=1.8, label=body)
    axis.set_title(f"CAGE — session {recording.session}, scenario {recording.scenario} ({recording.source})")
    if legend:
        axis.legend(ncol=2, fontsize=8, loc="upper right")


def trajectory_figure(recording: Recording, *, trim: bool = True, semantic_map: bool = True) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    draw_trajectory(recording, axis, trim=trim, semantic_map=semantic_map)
    return figure


def goal_assignment_figure(recording: Recording, *, every_s: float = 2.0, tail_s: float = 7.0) -> plt.Figure:
    """Plot assigned goals over trajectories."""
    if every_s <= 0 or tail_s <= 0:
        raise ValueError("every_s and tail_s must be positive")
    goal = recording.spec["temporal_alignment"]["goal_to_optitrack"]
    if not goal.get("accepted") or not goal.get("goal_file"):
        raise ValueError(f"S{recording.session}/scenario {recording.scenario} has no accepted goal/OptiTrack alignment; no goal-labelled plot was generated.")
    events = load_goal_events(recording.dataset.root / goal["goal_file"])
    data = recording.trimmed()
    time = data.Time.to_numpy(float)
    samples = np.arange(float(time[0]), float(time[-1]) + every_s / 2, every_s)
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    _draw_map(recording, axis)
    colors = plt.cm.tab10.colors
    start_epoch = recording.goal_timestamp_for_elapsed(0).timestamp()
    for color_index, body in enumerate(recording.humans):
        # Hat IDs match rigid-body numbers.
        hat_id = int(body.removeprefix("RigidBody"))
        xy = data[[f"{body}.position.x", f"{body}.position.y"]].to_numpy(float)
        color = colors[color_index % len(colors)]
        previous_target: str | None = None
        for elapsed in samples:
            index = np.searchsorted(time, elapsed, side="right") - 1
            if index < 0 or not np.isfinite(xy[index]).all():
                continue
            trail = (time >= max(time[0], elapsed - tail_s)) & (time <= elapsed)
            valid = trail & np.isfinite(xy).all(axis=1)
            axis.plot(xy[valid, 0], xy[valid, 1], color=color, alpha=0.22, lw=0.8)
            target = assigned_station(events, hat_id, start_epoch + elapsed)
            # Label each changed assignment once.
            if target and target != previous_target:
                axis.scatter(xy[index, 0], xy[index, 1], s=14, color=color, zorder=6)
                axis.text(xy[index, 0] + 0.06, xy[index, 1] + 0.06, f"H{hat_id}→{target}", color=color, fontsize=6)
            previous_target = target
    axis.set_title(f"Assigned goals, session {recording.session} scenario {recording.scenario}\n(labels are latest clicks; clicks can be delayed)")
    return figure


def goal_event_detail_figure(recording: Recording, events: pd.DataFrame, *, events_per_body: int = 5) -> plt.Figure:
    """Plot selected goal events for each person."""
    if events_per_body <= 0:
        raise ValueError("events_per_body must be positive")
    data = recording.data
    time = data.Time.to_numpy(float)
    start_epoch = recording.goal_timestamp_for_elapsed(0).timestamp()
    bodies = recording.humans
    columns = min(3, len(bodies))
    rows = math.ceil(len(bodies) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.3 * rows), squeeze=False, constrained_layout=True)
    colors = plt.cm.tab10.colors
    for axis, body, color in zip(axes.flat, bodies, colors, strict=False):
        _draw_map(recording, axis)
        hat_id = int(body.removeprefix("RigidBody"))
        xy = data[[f"{body}.position.x", f"{body}.position.y"]].to_numpy(float)
        finite = np.isfinite(xy).all(axis=1)
        if finite.any():
            axis.plot(xy[finite, 0], xy[finite, 1], color=color, alpha=0.38, lw=0.9, zorder=2)
        track_time, track_xy = time[finite], xy[finite]
        person_events = events[events.hat_id.eq(hat_id)].head(events_per_body)
        drawn = 0
        for event_number, event in enumerate(person_events.itertuples(index=False), start=1):
            elapsed = float(event.timestamp_epoch_s) - start_epoch
            if len(track_time) < 2 or elapsed < track_time[0] or elapsed > track_time[-1]:
                continue
            position = np.array((np.interp(elapsed, track_time, track_xy[:, 0]), np.interp(elapsed, track_time, track_xy[:, 1])))
            axis.scatter(*position, marker="o", s=55, facecolor="white", edgecolor=color, linewidth=1.8, zorder=6)
            from_station = str(getattr(event, "from_station", "?"))
            to_station = str(event.to_station)
            axis.text(position[0] + 0.09, position[1] + 0.09, f"{event_number}: {from_station}→{to_station}", color="black", fontsize=8,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8}, zorder=7)
            drawn += 1
        axis.set_title(f"Hat {hat_id}: {drawn}/{len(person_events)} first goal events", fontsize=10)
    for axis in axes.flat[len(bodies):]:
        axis.set_axis_off()
    figure.suptitle(f"Goal-event alignment — session {recording.session}, scenario {recording.scenario}\n"
                     "Marker N: station at click (FROM) → newly assigned station (TO); only first five per person", fontsize=12)
    return figure


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    if output.suffix.lower() == ".png":
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
