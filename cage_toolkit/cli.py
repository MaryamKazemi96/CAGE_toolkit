"""CAGE command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .metadata import DEFAULT_METADATA, build_metadata
from .metrics import recording_metrics
from .plotting import draw_trajectory, goal_assignment_figure, save_figure, trajectory_figure
from .recording import CAGEDataset
from .ros import (
    export_amcl_pose,
    export_odometry,
    export_robot_trajectories,
    plot_ros_optitrack_overlays,
    ros_bag_frame_inventory,
    ros_bag_inventory,
    ros_bag_topic_inventory,
    transform_pose_csv,
)
from .survey import analyze_survey


def _dataset(args: argparse.Namespace) -> CAGEDataset:
    return CAGEDataset(args.root, args.metadata)


def _metadata_path(root: Path, supplied: str | None) -> Path:
    return Path(supplied) if supplied else root / DEFAULT_METADATA


def _add_recording_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session", required=True, type=int)
    parser.add_argument("--scenario", required=True, type=int, choices=range(1, 5))
    parser.add_argument("--source", default="solved", choices=("raw", "solved"))


def _paper_figures(dataset: CAGEDataset, output: Path, sessions: list[int] | None, source: str) -> None:
    records = dataset.metadata["recordings"]
    available = sorted({int(record["session"]) for record in records.values()})
    for session in sessions or available:
        figure, axes = plt.subplots(1, 4, figsize=(18, 4.7), constrained_layout=True)
        for scenario, axis in zip(range(1, 5), axes):
            record = dataset.recording(session, scenario, source)
            draw_trajectory(record, axis, legend=False)
            axis.set_title(f"Scenario {scenario}")
        figure.suptitle(f"CAGE session {session}: trajectories ({source})")
        save_figure(figure, output / f"session_{session}_scenarios.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cage", description="Metadata-driven tools for the CAGE dataset")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Dataset root (contains data/, results/, toolkit/)")
    parser.add_argument("--metadata", help="Metadata YAML path; default: toolkit/config/recording_metadata.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    meta = commands.add_parser("metadata", help="Build or inspect the metadata index")
    meta_commands = meta.add_subparsers(dest="metadata_command", required=True)
    build = meta_commands.add_parser("build", help="Index source-native bounds and validated alignment evidence")
    build.add_argument("--without-bounds", action="store_true", help="Only index take headers; runtime trimming will be unavailable")

    inspect = commands.add_parser("inspect", help="Show one recording's metadata without loading its CSV")
    _add_recording_arguments(inspect)

    plot = commands.add_parser("plot", help="Create analysis or publication-quality plots")
    plot_commands = plot.add_subparsers(dest="plot_command", required=True)
    trajectory = plot_commands.add_parser("trajectory", help="Plot a source-native trajectory overlay")
    _add_recording_arguments(trajectory)
    trajectory.add_argument("--output", type=Path, required=True)
    trajectory.add_argument("--untrimmed", action="store_true")
    goals = plot_commands.add_parser("goals", help="Plot trajectory samples labelled by latest assigned goal")
    _add_recording_arguments(goals)
    goals.add_argument("--output", type=Path, required=True)
    goals.add_argument("--every-s", type=float, default=2.0)
    ros_overlay = plot_commands.add_parser("ros-overlay", help="Spatially overlay transformed ROS /amcl_pose and solved OptiTrack robot trajectories")
    ros_overlay.add_argument("--output", type=Path, default=Path("results/spatial_alignment/ros_optitrack_overlays"))
    ros_overlay.add_argument("--source", choices=("raw", "solved"), default="solved")
    ros_overlay.add_argument("--session", type=int, action="append", help="Optional session filter")
    ros_overlay.add_argument("--append-manifest", action="store_true", help="Merge this run into an existing manifest.csv")
    paper = commands.add_parser("paper-figures", help="Generate four-scenario session figures")
    paper.add_argument("--output", type=Path, default=Path("results/toolkit_paper_figures"))
    paper.add_argument("--session", type=int, action="append")
    paper.add_argument("--source", choices=("raw", "solved"), default="solved")

    metric = commands.add_parser("metrics", help="Compute per-person transparent movement summaries")
    _add_recording_arguments(metric)
    metric.add_argument("--output", type=Path, required=True)

    spatial = commands.add_parser("spatial", help="Apply the validated ROS map to OptiTrack transform")
    spatial_commands = spatial.add_subparsers(dest="spatial_command", required=True)
    transform = spatial_commands.add_parser("transform-csv", help="Transform exported /amcl_pose x/y columns; bags remain unchanged")
    transform.add_argument("input", type=Path)
    transform.add_argument("output", type=Path)
    export = spatial_commands.add_parser("export-amcl", help="Read /amcl_pose from a ROS 2 bag and export immutable-derived CSV")
    export.add_argument("bag", type=Path, help="Directory containing one or more .db3 files")
    export.add_argument("output", type=Path)
    export_odom = spatial_commands.add_parser("export-odom", help="Read odometry from a ROS 2 bag and export CSV")
    export_odom.add_argument("bag", type=Path, help="Directory containing one or more .db3 files")
    export_odom.add_argument("output", type=Path)

    ros = commands.add_parser("ros", help="Inspect immutable ROS 2 bag contents")
    ros_commands = ros.add_subparsers(dest="ros_command", required=True)
    inventory = ros_commands.add_parser("inventory", help="Write a bag/topic availability table without deserializing messages")
    inventory.add_argument("--output", type=Path, default=Path("results/ros_bag_inventory.csv"))
    topics = ros_commands.add_parser("topics", help="Write every recorded topic, type, count, and storage-time bounds")
    topics.add_argument("--output", type=Path, default=Path("results/ros_bag_topic_inventory.csv"))
    topics.add_argument("--session", type=int, action="append", help="Optional session filter")
    frames = ros_commands.add_parser("frames", help="Read navigation messages and report all observed frame IDs and header/storage timestamp differences")
    frames.add_argument("--output", type=Path, default=Path("results/ros_bag_frame_inventory.csv"))
    frames.add_argument("--session", type=int, action="append", help="Optional session filter")
    frames.add_argument("--append", action="store_true", help="Merge this run into an existing output, replacing inspected bags")
    trajectories = ros_commands.add_parser("export-trajectories", help="Export AMCL and odometry trajectories for the selected bags")
    trajectories.add_argument("--output", type=Path, default=Path("results/ros_trajectory_exports"))
    trajectories.add_argument("--session", type=int, action="append", help="Optional session filter")
    trajectories.add_argument("--include-optitrack-coordinates", action="store_true", help="Add transformed OptiTrack x/y to AMCL CSVs")
    trajectories.add_argument("--append-manifest", action="store_true", help="Update an existing manifest when exporting sessions separately")

    survey = commands.add_parser("survey", help="Run the compact survey summary")
    survey.add_argument("--input", type=Path, default=Path("data/survey/survey.csv"))
    survey.add_argument("--output", type=Path, default=Path("results/toolkit_survey"))
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "metadata":
        output = _metadata_path(root, args.metadata)
        index = build_metadata(root, output, include_bounds=not args.without_bounds)
        print(f"Wrote {len(index['recordings'])} recordings to {output}")
        return
    if args.command == "survey":
        result = analyze_survey(root / args.input, root / args.output)
        print(result.to_string(index=False))
        return
    dataset = _dataset(args)
    if args.command == "inspect":
        spec = dataset.metadata["recordings"][f"s{args.session}_scenario{args.scenario}"]
        print(pd.Series(spec).to_string())
    elif args.command == "plot":
        if args.plot_command == "ros-overlay":
            result = plot_ros_optitrack_overlays(dataset, args.output, source=args.source, sessions=args.session,
                                                  append_manifest=args.append_manifest)
            print(result[["session", "scenario", "bag_kind", "status"]].to_string(index=False))
            print(f"Saved ROS/OptiTrack overlays and manifest to {args.output}")
        else:
            record = dataset.recording(args.session, args.scenario, args.source)
            figure = trajectory_figure(record, trim=not args.untrimmed) if args.plot_command == "trajectory" else goal_assignment_figure(record, every_s=args.every_s)
            save_figure(figure, args.output)
            print(f"Saved {args.output}")
    elif args.command == "paper-figures":
        _paper_figures(dataset, args.output, args.session, args.source)
        print(f"Saved figures to {args.output}")
    elif args.command == "metrics":
        result = recording_metrics(dataset.recording(args.session, args.scenario, args.source))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(result.to_string(index=False))
    elif args.command == "spatial":
        if args.spatial_command == "transform-csv":
            transform_pose_csv(args.input, args.output, dataset.spatial_transform())
        elif args.spatial_command == "export-amcl":
            export_amcl_pose(args.bag, args.output, dataset.spatial_transform())
        else:
            export_odometry(args.bag, args.output)
        print(f"Saved trajectory CSV to {args.output}")
    elif args.command == "ros":
        if args.ros_command == "inventory":
            result = ros_bag_inventory(root)
        elif args.ros_command == "topics":
            result = ros_bag_topic_inventory(root, args.session)
        elif args.ros_command == "frames":
            result = ros_bag_frame_inventory(root, args.session)
            if args.append and args.output.exists():
                old = pd.read_csv(args.output)
                result = pd.concat((old, result), ignore_index=True).drop_duplicates("bag_path", keep="last")
                result = result.sort_values(["session", "bag_kind", "bag_path"], na_position="last")
        else:
            transform = dataset.spatial_transform() if args.include_optitrack_coordinates else None
            result = export_robot_trajectories(root, args.output, transform=transform, sessions=args.session,
                                               append_manifest=args.append_manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.ros_command == "export-trajectories":
            print(result[["session", "bag_kind", "amcl_status", "odom_status"]].to_string(index=False))
            print(f"Saved ROS trajectory CSVs to {args.output}")
            return
        result.to_csv(args.output, index=False)
        if args.ros_command == "topics":
            print(f"Indexed {len(result)} bag/topic rows.")
        else:
            print(result.to_string(index=False))
        print(f"Saved ROS {args.ros_command} report to {args.output}")


if __name__ == "__main__":
    main()
