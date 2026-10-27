"""Transparent goal-conditioned trajectory-prediction baselines.

The benchmark deliberately uses simple, non-trained predictors.  It is meant
to make the value of CAGE's time-aligned destination annotations measurable,
not to claim a state-of-the-art forecasting result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .goals import load_goal_events
from .recording import CAGEDataset


METHOD_CV = "Goal-agnostic constant velocity"
METHOD_GOAL = "Goal-conditioned direct-to-station"


@dataclass(frozen=True)
class PredictionConfig:
    """Evaluation settings for the simple, event-aligned benchmark."""

    horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    history_s: float = 0.5
    sample_step_s: float = 0.1
    max_interpolation_gap_s: float = 0.15


def _interpolate_xy(time: np.ndarray, xy: np.ndarray, query: float, max_gap_s: float) -> np.ndarray | None:
    """Linearly interpolate one location without bridging a tracking gap."""
    right = int(np.searchsorted(time, query, side="left"))
    if right == 0 or right == len(time):
        return None
    left = right - 1
    if time[right] - time[left] > max_gap_s:
        return None
    fraction = (query - time[left]) / (time[right] - time[left])
    return xy[left] + fraction * (xy[right] - xy[left])


def _track(data: pd.DataFrame, body: str) -> tuple[np.ndarray, np.ndarray]:
    values = data[["Time", f"{body}.position.x", f"{body}.position.y"]].to_numpy(float)
    values = values[np.isfinite(values).all(axis=1)]
    if not len(values):
        return np.empty(0), np.empty((0, 2))
    # Keep one position per timestamp so interpolation remains well defined.
    keep = np.r_[True, np.diff(values[:, 0]) > 0]
    values = values[keep]
    return values[:, 0], values[:, 1:]


def _prediction_rows(
    *, dataset: CAGEDataset, session: int, scenario: int, config: PredictionConfig
) -> tuple[list[dict], list[dict]]:
    """Score both methods for all valid goal events in one recording."""
    spec = dataset.metadata["recordings"][f"s{session}_scenario{scenario}"]
    alignment = spec["temporal_alignment"]["goal_to_optitrack"]
    if not alignment.get("accepted") or not alignment.get("goal_file"):
        return [], []
    stations = dataset.station_regions()
    events = load_goal_events(dataset.root / alignment["goal_file"])
    start_epoch = pd.Timestamp(alignment["corrected_optitrack_start"]).timestamp()
    # Match the standard solved-data analysis view: normalized coordinates and
    # cached raw-visibility bounds.
    data = dataset.recording(session, scenario, source="solved").trimmed()
    scores: list[dict] = []
    examples: list[dict] = []

    for body in spec["humans"]:
        hat_id = int(body.removeprefix("RigidBody"))
        person_events = events[events.hat_id.eq(hat_id)].reset_index(drop=True)
        time, xy = _track(data, body)
        if len(time) < 2:
            continue
        for event_index, event in person_events.iterrows():
            station = str(event.to_station)
            if station not in stations:
                continue
            event_time = float(event.timestamp_epoch_s - start_epoch)
            next_event_time = (float(person_events.iloc[event_index + 1].timestamp_epoch_s - start_epoch)
                               if event_index + 1 < len(person_events) else np.inf)
            current = _interpolate_xy(time, xy, event_time, config.max_interpolation_gap_s)
            past = _interpolate_xy(time, xy, event_time - config.history_s, config.max_interpolation_gap_s)
            if current is None or past is None:
                continue
            velocity = (current - past) / config.history_s
            speed = float(np.linalg.norm(velocity))
            destination = np.array((stations[station]["x_m"], stations[station]["y_m"]), dtype=float)
            offset = destination - current
            distance = float(np.linalg.norm(offset))
            goal_velocity = np.zeros(2) if distance < 1e-9 else offset / distance * speed

            for horizon in config.horizons_s:
                # Stop this evaluation window when the next assignment changes the task.
                if event_time + horizon >= next_event_time:
                    continue
                sample_times = np.arange(config.sample_step_s, horizon + config.sample_step_s / 2, config.sample_step_s)
                truth = [_interpolate_xy(time, xy, event_time + delta, config.max_interpolation_gap_s) for delta in sample_times]
                if any(point is None for point in truth):
                    continue
                truth_xy = np.asarray(truth)
                predictions = {
                    METHOD_CV: current + sample_times[:, None] * velocity,
                    # Stop at the centroid rather than travelling through it.
                    METHOD_GOAL: current + np.minimum(sample_times[:, None] * speed, distance) * (offset / distance if distance > 1e-9 else 0),
                }
                base = {
                    "session": session, "scenario": scenario, "density": spec["density"], "body": body,
                    "hat_id": hat_id, "event_index": int(event_index + 1), "event_elapsed_s": event_time,
                    "from_station": str(event.get("from_station", "")), "to_station": station,
                    "horizon_s": horizon, "speed_at_event_mps": speed,
                }
                for method, forecast in predictions.items():
                    errors = np.linalg.norm(forecast - truth_xy, axis=1)
                    scores.append({**base, "method": method, "ade_m": float(errors.mean()), "fde_m": float(errors[-1])})

                # Store one deterministic example per eligible event for qualitative plots.
                if horizon == max(config.horizons_s):
                    examples.append({**base, "current": current, "past": past, "truth": truth_xy,
                                     "cv": predictions[METHOD_CV], "goal": predictions[METHOD_GOAL],
                                     "destination": destination})
    return scores, examples


def _summary(scores: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    grouped = scores.groupby(columns + ["method"], as_index=False).agg(
        predictions=("fde_m", "size"), ade_mean_m=("ade_m", "mean"), ade_median_m=("ade_m", "median"),
        ade_std_m=("ade_m", "std"), fde_mean_m=("fde_m", "mean"), fde_median_m=("fde_m", "median"),
        fde_std_m=("fde_m", "std"),
    )
    return grouped.sort_values(columns + ["method"]).reset_index(drop=True)


def _write_markdown(summary: pd.DataFrame, output: Path, config: PredictionConfig) -> None:
    pivot = summary.pivot(index="horizon_s", columns="method", values="fde_mean_m")
    lines = [
        "# Goal-conditioned vs goal-agnostic trajectory prediction",
        "",
        "This is a non-learning, event-aligned baseline benchmark. Both methods estimate speed from the preceding "
        f"{config.history_s:g} s of motion. The goal-agnostic method keeps that velocity; the goal-conditioned "
        "method moves at the same speed toward the newly assigned station centroid and stops there.",
        "",
        "A prediction is evaluated only when the participant is tracked through the full horizon and no later goal "
        "assignment occurs first. ADE is mean error over 0.1 s forecast samples; FDE is error at the horizon.",
        "The error figure uses box plots: the centre line is the median and each box spans the interquartile range.",
        "Qualitative examples are written as separate one-hat images for every session's scenarios 2, 3, and 4 "
        "(all hats for Session 8, one representative hat elsewhere); "
        "their selected events are listed in `qualitative_examples.csv`.",
        "",
        "## Mean final-displacement error (m)",
        "",
        "| Horizon (s) | Goal agnostic | Goal conditioned | Relative change |",
        "|---:|---:|---:|---:|",
    ]
    for horizon, values in pivot.iterrows():
        agnostic = values.get(METHOD_CV, np.nan)
        conditioned = values.get(METHOD_GOAL, np.nan)
        change = 100 * (conditioned - agnostic) / agnostic if np.isfinite(agnostic) and agnostic else np.nan
        lines.append(f"| {horizon:.1f} | {agnostic:.3f} | {conditioned:.3f} | {change:+.1f}% |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_errors(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharex=True, constrained_layout=True)
    colors = {METHOD_CV: "#D62728", METHOD_GOAL: "#0072B2"}
    for method, table in summary.groupby("method"):
        table = table.sort_values("horizon_s")
        axes[0].plot(table.horizon_s, table.ade_mean_m, marker="o", lw=2.2, color=colors[method], label=method)
        axes[0].fill_between(table.horizon_s, table.ade_mean_m - table.ade_std_m,
                             table.ade_mean_m + table.ade_std_m, color=colors[method], alpha=0.17)
        axes[1].plot(table.horizon_s, table.fde_mean_m, marker="o", lw=2.2, color=colors[method], label=method)
        axes[1].fill_between(table.horizon_s, table.fde_mean_m - table.fde_std_m,
                             table.fde_mean_m + table.fde_std_m, color=colors[method], alpha=0.17)
    axes[0].set(title="Average displacement error", ylabel="ADE [m]", xlabel="Forecast horizon [s]")
    axes[1].set(title="Final displacement error", ylabel="FDE [m]", xlabel="Forecast horizon [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xticks(sorted(summary.horizon_s.unique()))
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.11),
                  frameon=False, title="Lines: mean; shaded bands: ±1 standard deviation")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_error_boxplots(scores: pd.DataFrame, output: Path) -> None:
    """Plot non-negative prediction errors as grouped median-and-quartile box plots."""

    from matplotlib.patches import Patch

    methods = [METHOD_CV, METHOD_GOAL]
    colors = {METHOD_CV: "#D62728", METHOD_GOAL: "#0072B2"}
    labels = {METHOD_CV: "Goal agnostic", METHOD_GOAL: "Goal conditioned"}
    horizons = np.sort(scores.horizon_s.unique())
    positions = np.arange(len(horizons), dtype=float)
    offsets = (-0.18, 0.18)
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.25), sharex=True, constrained_layout=True)
    panels = (
        (axes[0], "ade_m", "Average displacement error", "ADE [m]"),
        (axes[1], "fde_m", "Final displacement error", "FDE [m]"),
    )
    for axis, column, title, ylabel in panels:
        for method, offset in zip(methods, offsets, strict=True):
            values = [scores.loc[(scores.method == method) & (scores.horizon_s == horizon), column].to_numpy()
                      for horizon in horizons]
            boxes = axis.boxplot(
                values, positions=positions + offset, widths=0.30, patch_artist=True,
                showfliers=False, whis=(5, 95),
                boxprops={"facecolor": colors[method], "edgecolor": colors[method], "alpha": 0.50, "linewidth": 1.2},
                medianprops={"color": colors[method], "linewidth": 2.0},
                whiskerprops={"color": colors[method], "linewidth": 1.0},
                capprops={"color": colors[method], "linewidth": 1.0},
            )
            for median in boxes["medians"]:
                median.set_color(colors[method])
        axis.set(title=title, ylabel=ylabel, xlabel="Forecast horizon [s]")
        axis.set_xticks(positions, [f"{h:g}" for h in horizons])
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.24)
        axis.tick_params(axis="both", labelsize=9, width=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.legend(handles=[Patch(facecolor=colors[method], edgecolor=colors[method], alpha=0.50, label=labels[method])
                           for method in methods], loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9,
                  title="Box: 25th--75th percentile; line: median; whiskers: 5th--95th percentile", title_fontsize=8.5)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_error_bars(summary: pd.DataFrame, output: Path) -> None:
    """Draw a single-column-paper grouped-bar view without replacing the line plot."""
    methods = [METHOD_CV, METHOD_GOAL]
    colors = {METHOD_CV: "#D62728", METHOD_GOAL: "#0072B2"}
    horizons = np.sort(summary.horizon_s.unique())
    positions = np.arange(len(horizons), dtype=float)
    width = 0.36
    # 3.45 in is a typical IEEE/RA-L two-column paper column width. Stacking
    # the metrics keeps bar and text sizes legible at final print scale.
    figure, axes = plt.subplots(2, 1, figsize=(3.45, 5.15), sharex=True, constrained_layout=True)
    for method_index, method in enumerate(methods):
        table = summary[summary.method.eq(method)].set_index("horizon_s").reindex(horizons)
        offset = (method_index - 0.5) * width
        for axis, mean, std, title, ylabel in (
            (axes[0], table.ade_mean_m, table.ade_std_m, "Average displacement error", "ADE [m]"),
            (axes[1], table.fde_mean_m, table.fde_std_m, "Final displacement error", "FDE [m]"),
        ):
            axis.bar(positions + offset, mean, width, yerr=std, capsize=3.5, color=colors[method],
                     alpha=0.88, label=method, error_kw={"elinewidth": 1.25})
            axis.set_title(title, fontsize=10.5, pad=5)
            axis.set_ylabel(ylabel, fontsize=10)
            axis.grid(axis="y", alpha=0.25)
            axis.tick_params(axis="both", labelsize=9, width=0.9)
    for axis in axes:
        axis.set_xticks(positions, [f"{h:g}" for h in horizons])
    axes[1].set_xlabel("Forecast horizon [s]", fontsize=10)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.16),
                  frameon=False, fontsize=8.5, title="Bars: mean; error bars: ±1 SD", title_fontsize=8.5)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_error_bars_side_by_side(summary: pd.DataFrame, output: Path) -> None:
    """Draw a clean double-column alternative with ADE and FDE side by side."""
    methods = [METHOD_CV, METHOD_GOAL]
    colors = {METHOD_CV: "#D62728", METHOD_GOAL: "#0072B2"}
    labels = {METHOD_CV: "Goal agnostic", METHOD_GOAL: "Goal conditioned"}
    horizons = np.sort(summary.horizon_s.unique())
    positions = np.arange(len(horizons), dtype=float)
    width = 0.37
    # 7.16 in is a typical two-column IEEE/RA-L text width.
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), sharex=True, constrained_layout=True)
    panels = (
        (axes[0], "ade_mean_m", "ade_std_m", "Average displacement error", "ADE [m]"),
        (axes[1], "fde_mean_m", "fde_std_m", "Final displacement error", "FDE [m]"),
    )
    for method_index, method in enumerate(methods):
        table = summary[summary.method.eq(method)].set_index("horizon_s").reindex(horizons)
        offset = (method_index - 0.5) * width
        for axis, mean_name, std_name, title, ylabel in panels:
            axis.bar(positions + offset, table[mean_name], width, yerr=table[std_name], capsize=3.2,
                     color=colors[method], alpha=0.9, label=labels[method], error_kw={"elinewidth": 1.1})
            axis.set_title(title, fontsize=11, pad=6)
            axis.set_ylabel(ylabel, fontsize=10)
            axis.grid(axis="y", alpha=0.24)
            axis.tick_params(axis="both", labelsize=9, width=0.8)
            # Remove the enclosing box while retaining the axes needed to read values.
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    for axis in axes:
        axis.set_xticks(positions, [f"{h:g}" for h in horizons])
        axis.set_xlabel("Forecast horizon [s]", fontsize=10)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14),
                  frameon=False, fontsize=9, title="Bars: mean; error bars: ±1 SD", title_fontsize=9)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_error_violins(scores: pd.DataFrame, output: Path) -> None:
    """Show the full per-event ADE/FDE distributions in a double-column figure."""
    from matplotlib.patches import Patch

    methods = [METHOD_CV, METHOD_GOAL]
    colors = {METHOD_CV: "#D62728", METHOD_GOAL: "#0072B2"}
    labels = {METHOD_CV: "Goal agnostic", METHOD_GOAL: "Goal conditioned"}
    horizons = np.sort(scores.horizon_s.unique())
    centers = np.arange(len(horizons), dtype=float)
    offsets = (-0.19, 0.19)
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.35), sharex=True, constrained_layout=True)
    for axis, column, title, ylabel in (
        (axes[0], "ade_m", "Average displacement error", "ADE [m]"),
        (axes[1], "fde_m", "Final displacement error", "FDE [m]"),
    ):
        for method, offset in zip(methods, offsets, strict=True):
            values = [scores.loc[(scores.horizon_s == horizon) & scores.method.eq(method), column].to_numpy()
                      for horizon in horizons]
            violin = axis.violinplot(values, positions=centers + offset, widths=0.32,
                                     showmeans=False, showmedians=True, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(colors[method])
                body.set_edgecolor(colors[method])
                body.set_alpha(0.60)
            violin["cmedians"].set_color(colors[method])
            violin["cmedians"].set_linewidth(1.4)
        axis.set(title=title, ylabel=ylabel)
        axis.set_xticks(centers, [f"{h:g}" for h in horizons])
        axis.set_xlabel("Forecast horizon [s]")
        axis.grid(axis="y", alpha=0.24)
        axis.tick_params(axis="both", labelsize=9, width=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        # Keep rare extreme extrapolations in the CSV while limiting this view
        # to the pooled 99.5th percentile for readability.
        display_limit = float(scores[column].quantile(0.995))
        axis.set_ylim(bottom=0, top=1.04 * display_limit)
    figure.legend(handles=[Patch(facecolor=colors[method], edgecolor=colors[method], alpha=0.60, label=labels[method])
                           for method in methods],
                  loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9,
                  title="Median line; display capped at pooled 99.5th percentile",
                  title_fontsize=8.5)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_one_example(dataset: CAGEDataset, item: dict, output: Path) -> None:
    """Write one readable qualitative comparison, never a multi-panel collage."""
    figure, axis = plt.subplots(figsize=(7.8, 6.3), constrained_layout=True)
    meta = dataset.metadata["map"]
    image = dataset.root / "data/figures/env6.png"
    if image.exists():
        axis.imshow(plt.imread(image), extent=meta["extent_m"], origin="upper", alpha=0.42, zorder=0)
    axis.plot([item["past"][0], item["current"][0]], [item["past"][1], item["current"][1]], color="black", lw=2.2, label="Observed history (0.5 s)")
    axis.plot(item["truth"][:, 0], item["truth"][:, 1], color="#009E73", lw=2.6, label="Observed future")
    axis.plot(item["cv"][:, 0], item["cv"][:, 1], color="#D62728", lw=2.2, ls="--", label="Goal agnostic")
    axis.plot(item["goal"][:, 0], item["goal"][:, 1], color="#0072B2", lw=2.2, ls="--", label="Goal conditioned")
    axis.scatter(*item["current"], s=46, color="black", zorder=5)
    axis.scatter(*item["destination"], s=120, marker="*", color="#D55E00", edgecolor="white", linewidth=0.8, zorder=6, label="Assigned station")
    axis.set(xlim=meta["extent_m"][:2], ylim=meta["extent_m"][2:], aspect="equal", xlabel="OptiTrack x [m]", ylabel="OptiTrack y [m]")
    axis.set_title(f"Session {item['session']}, scenario {item['scenario']}, Hat {item['hat_id']} → {item['to_station']}\n"
                   f"{item['horizon_s']:.0f} s forecast at assignment {item['event_index']}")
    axis.legend(loc="upper right", fontsize=8, framealpha=0.94)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_examples(dataset: CAGEDataset, examples: list[dict], output: Path) -> pd.DataFrame:
    """Write Session-8 all-hat examples plus representative examples for other sessions."""
    rows: list[dict] = []
    for session in sorted({item["session"] for item in examples}):
        for scenario in (2, 3, 4):
            scenario_examples = [item for item in examples if item["session"] == session and item["scenario"] == scenario]
            hat_ids: list[int | None] = sorted({item["hat_id"] for item in scenario_examples}) if session == 8 else [None]
            for hat_id in hat_ids:
                candidates = [item for item in scenario_examples if (hat_id is None or item["hat_id"] == hat_id) and 0.15 <= item["speed_at_event_mps"] <= 2.0]
                if not candidates:
                    candidates = [item for item in scenario_examples if hat_id is None or item["hat_id"] == hat_id]
                if not candidates:
                    continue
                # Prefer a clear method difference and use event time to break ties.
                item = min(candidates, key=lambda candidate: (-np.linalg.norm(candidate["goal"][-1] - candidate["cv"][-1]),
                                                                candidate["event_elapsed_s"]))
                filename = f"qualitative_s{session:02d}_sc{scenario:02d}_hat{item['hat_id']:02d}.png"
                _plot_one_example(dataset, item, output / filename)
                rows.append({
                    "session": item["session"], "scenario": scenario, "hat_id": item["hat_id"], "event_index": item["event_index"],
                    "event_elapsed_s": item["event_elapsed_s"], "to_station": item["to_station"], "horizon_s": item["horizon_s"],
                    "speed_at_event_mps": item["speed_at_event_mps"], "goal_agnostic_fde_m": float(np.linalg.norm(item["cv"][-1] - item["truth"][-1])),
                    "goal_conditioned_fde_m": float(np.linalg.norm(item["goal"][-1] - item["truth"][-1])), "figure": filename,
                })
    return pd.DataFrame(rows)


def run_prediction_comparison(dataset: CAGEDataset, output: Path, config: PredictionConfig = PredictionConfig()) -> pd.DataFrame:
    """Run the benchmark, write tables/plots, and return the horizon summary."""
    if not config.horizons_s or min(config.horizons_s) <= 0 or config.history_s <= 0 or config.sample_step_s <= 0:
        raise ValueError("History, sample step, and horizons must be positive")
    output.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict] = []
    examples: list[dict] = []
    records = sorted(dataset.metadata["recordings"].values(), key=lambda row: (row["session"], row["scenario"]))
    for record in records:
        scores, record_examples = _prediction_rows(dataset=dataset, session=int(record["session"]), scenario=int(record["scenario"]), config=config)
        score_rows.extend(scores)
        examples.extend(record_examples)
    scores = pd.DataFrame(score_rows)
    if scores.empty:
        raise ValueError("No valid event-aligned prediction windows were found")
    scores.to_csv(output / "per_event_prediction_errors.csv", index=False)
    summary = _summary(scores, ["horizon_s"])
    by_density = _summary(scores, ["density", "horizon_s"])
    by_scenario = _summary(scores, ["scenario", "horizon_s"])
    summary.to_csv(output / "summary_by_horizon.csv", index=False)
    by_density.to_csv(output / "summary_by_density.csv", index=False)
    by_scenario.to_csv(output / "summary_by_scenario.csv", index=False)
    _write_markdown(summary, output / "README.md", config)
    _plot_error_boxplots(scores, output / "error_by_horizon_boxplot.png")
    _plot_error_violins(scores, output / "error_by_horizon_violinplot.png")
    selections = _plot_examples(dataset, examples, output)
    selections.to_csv(output / "qualitative_examples.csv", index=False)
    return summary
