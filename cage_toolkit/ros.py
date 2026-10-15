"""ROS bag helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import numpy as np

from .spatial import MapToOptiTrackTransform


_AMCL_POSE_TYPE = "geometry_msgs/msg/PoseWithCovarianceStamped"
_ODOMETRY_TYPE = "nav_msgs/msg/Odometry"


def _epoch_ns_to_utc(epoch_ns: int) -> str:
    """Format an integer Unix epoch in UTC without losing nanoseconds."""
    seconds, nanoseconds = divmod(int(epoch_ns), 1_000_000_000)
    return f"{datetime.fromtimestamp(seconds, timezone.utc):%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}+00:00"


def _output_path(path: Path, root: Path) -> str:
    """Return a dataset-relative path when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _topic_inventory_rows(bag_path: Path) -> list[dict[str, object]]:
    """Return topic counts and storage-time bounds."""
    totals: dict[tuple[str, str], dict[str, object]] = {}
    for database in sorted(bag_path.glob("*.db3")):
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            query = """
                SELECT topics.name, topics.type, COUNT(messages.id),
                       MIN(messages.timestamp), MAX(messages.timestamp)
                FROM topics LEFT JOIN messages ON messages.topic_id = topics.id
                GROUP BY topics.id, topics.name, topics.type
            """
            for name, message_type, count, lower, upper in connection.execute(query):
                key = (str(name), str(message_type))
                row = totals.setdefault(key, {
                    "topic": str(name), "message_type": str(message_type),
                    "message_count": 0, "storage_timestamp_start_ns": None,
                    "storage_timestamp_end_ns": None,
                })
                row["message_count"] = int(row["message_count"]) + int(count)
                if lower is not None:
                    row["storage_timestamp_start_ns"] = min(
                        int(lower), row["storage_timestamp_start_ns"]
                    ) if row["storage_timestamp_start_ns"] is not None else int(lower)
                if upper is not None:
                    row["storage_timestamp_end_ns"] = max(
                        int(upper), row["storage_timestamp_end_ns"]
                    ) if row["storage_timestamp_end_ns"] is not None else int(upper)
        finally:
            connection.close()
    if not totals:
        raise ValueError(f"No ROS topics in {bag_path}")
    return [totals[key] for key in sorted(totals)]


def _topic_rows(bag_path: Path) -> list[tuple[str, str, int]]:
    """Return ``(name, type, message_count)`` across every SQLite segment."""
    return [
        (str(row["topic"]), str(row["message_type"]), int(row["message_count"]))
        for row in _topic_inventory_rows(bag_path)
    ]


def _amcl_topics(bag_path: Path) -> list[str]:
    """Find AMCL pose topics."""
    return [
        name for name, message_type, count in _topic_rows(bag_path)
        if count and message_type == _AMCL_POSE_TYPE
        and (name == "/amcl_pose" or name.endswith("/amcl_pose"))
    ]


def _topics_with_suffix(bag_path: Path, message_type: str, suffix: str) -> list[str]:
    """Find root or namespaced topics with a given suffix."""
    return [
        name for name, stored_type, count in _topic_rows(bag_path)
        if count and stored_type == message_type and (name == suffix or name.endswith(suffix))
    ]


def transform_pose_csv(input_csv: Path, output_csv: Path, transform: MapToOptiTrackTransform) -> None:
    """Add OptiTrack coordinates to a map-pose CSV."""
    import pandas as pd
    data = pd.read_csv(input_csv)
    if not {"x", "y"}.issubset(data.columns):
        raise ValueError("Pose CSV must contain x and y columns in ROS map frame")
    data[["optitrack_x", "optitrack_y"]] = transform.apply(data[["x", "y"]].to_numpy(float))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_csv, index=False)


def export_amcl_pose(bag_path: Path, output_csv: Path, transform: MapToOptiTrackTransform | None = None) -> None:
    """Write AMCL poses to CSV."""
    result = read_amcl_pose(bag_path)
    if transform is not None:
        result[["optitrack_x", "optitrack_y"]] = transform.apply(result[["x", "y"]].to_numpy(float))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)


def read_amcl_pose(bag_path: Path):
    """Read AMCL pose samples from every bag segment."""
    try:
        import pandas as pd
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from rclpy.serialization import deserialize_message
    except ImportError as error:
        raise RuntimeError("ROS bag export requires a sourced ROS 2 environment (rclpy and geometry_msgs)") from error
    topic_names = _amcl_topics(bag_path)
    rows = []
    for database in sorted(bag_path.glob("*.db3")):
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        topics = connection.execute(
            "SELECT id FROM topics WHERE name IN ({})".format(
                ", ".join("?" for _ in topic_names)
            ),
            topic_names,
        ).fetchall() if topic_names else []
        for (topic_id,) in topics:
            topic_name = connection.execute("SELECT name FROM topics WHERE id=?", (topic_id,)).fetchone()[0]
            for timestamp, data in connection.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,)):
                message = deserialize_message(data, PoseWithCovarianceStamped)
                point = message.pose.pose.position
                if np.isfinite(point.x) and np.isfinite(point.y):
                    orientation = message.pose.pose.orientation
                    rows.append({
                        "timestamp_epoch_s": timestamp / 1e9,
                        "header_timestamp_epoch_s": _header_timestamp_ns(message.header) / 1e9,
                        "topic": topic_name,
                        "frame_id": message.header.frame_id,
                        "x": point.x, "y": point.y, "z": point.z,
                        "orientation_x": orientation.x, "orientation_y": orientation.y,
                        "orientation_z": orientation.z, "orientation_w": orientation.w,
                    })
        connection.close()
    if not rows:
        raise ValueError(f"No readable AMCL-pose messages (*/amcl_pose) in {bag_path}")
    result = pd.DataFrame(rows).sort_values("timestamp_epoch_s").reset_index(drop=True)
    result.attrs["amcl_topics"] = topic_names
    return result


def read_odometry(bag_path: Path):
    """Read odometry trajectory samples from every bag segment."""
    try:
        import pandas as pd
        from nav_msgs.msg import Odometry
        from rclpy.serialization import deserialize_message
    except ImportError as error:
        raise RuntimeError("Odometry export requires a sourced ROS 2 environment (rclpy and nav_msgs)") from error
    topic_names = _topics_with_suffix(bag_path, _ODOMETRY_TYPE, "/odom")
    rows = []
    for database in sorted(bag_path.glob("*.db3")):
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        topics = connection.execute(
            "SELECT id, name FROM topics WHERE name IN ({})".format(
                ", ".join("?" for _ in topic_names)
            ), topic_names,
        ).fetchall() if topic_names else []
        for topic_id, topic_name in topics:
            for timestamp, data in connection.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,)):
                message = deserialize_message(data, Odometry)
                point = message.pose.pose.position
                orientation = message.pose.pose.orientation
                linear = message.twist.twist.linear
                angular = message.twist.twist.angular
                if np.isfinite(point.x) and np.isfinite(point.y):
                    rows.append({
                        "timestamp_epoch_s": timestamp / 1e9,
                        "header_timestamp_epoch_s": _header_timestamp_ns(message.header) / 1e9,
                        "topic": topic_name,
                        "frame_id": message.header.frame_id,
                        "child_frame_id": message.child_frame_id,
                        "x": point.x, "y": point.y, "z": point.z,
                        "orientation_x": orientation.x, "orientation_y": orientation.y,
                        "orientation_z": orientation.z, "orientation_w": orientation.w,
                        "linear_x": linear.x, "linear_y": linear.y, "linear_z": linear.z,
                        "angular_x": angular.x, "angular_y": angular.y, "angular_z": angular.z,
                    })
        connection.close()
    if not rows:
        raise ValueError(f"No readable odometry messages (*/odom) in {bag_path}")
    result = pd.DataFrame(rows).sort_values("timestamp_epoch_s").reset_index(drop=True)
    result.attrs["odom_topics"] = topic_names
    return result


def export_odometry(bag_path: Path, output_csv: Path) -> None:
    """Write odometry trajectory samples to CSV."""
    result = read_odometry(bag_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)


def export_robot_trajectories(root: Path, output: Path, *, transform: MapToOptiTrackTransform | None = None,
                              sessions: list[int] | None = None, append_manifest: bool = False):
    """Export AMCL and odometry trajectories from all selected bags."""
    import pandas as pd

    bag_root = root / "data" / "Rosbags"
    bag_paths = sorted({path.parent for path in bag_root.rglob("*.db3")})
    rows: list[dict[str, object]] = []
    for bag_path in bag_paths:
        session, bag_kind = _bag_session_and_kind(root, bag_path)
        if sessions and session not in sessions:
            continue
        common = {
            "session": session,
            "bag_kind": bag_kind,
            "bag_path": str(bag_path.relative_to(root)),
        }
        folder = output / f"s{session}_{bag_kind}_{bag_path.parent.name}_{bag_path.name}"
        result: dict[str, object] = {
            **common, "status": "ok", "detail": "",
            "amcl_optitrack_coordinates": transform is not None,
        }
        try:
            amcl = read_amcl_pose(bag_path)
            if transform is not None:
                amcl[["optitrack_x", "optitrack_y"]] = transform.apply(amcl[["x", "y"]].to_numpy(float))
            amcl_path = folder / "amcl_pose.csv"
            amcl_path.parent.mkdir(parents=True, exist_ok=True)
            amcl.to_csv(amcl_path, index=False)
            result.update({
                "amcl_status": "exported", "amcl_topics": ";".join(amcl.attrs["amcl_topics"]),
                "amcl_rows": len(amcl), "amcl_csv": _output_path(amcl_path, root),
            })
        except (sqlite3.DatabaseError, RuntimeError, ValueError) as error:
            result.update({"amcl_status": "not_exported", "amcl_detail": str(error)})
        try:
            odom = read_odometry(bag_path)
            odom_path = folder / "odometry.csv"
            odom_path.parent.mkdir(parents=True, exist_ok=True)
            odom.to_csv(odom_path, index=False)
            result.update({
                "odom_status": "exported", "odom_topics": ";".join(odom.attrs["odom_topics"]),
                "odom_rows": len(odom), "odometry_csv": _output_path(odom_path, root),
            })
        except (sqlite3.DatabaseError, RuntimeError, ValueError) as error:
            result.update({"odom_status": "not_exported", "odom_detail": str(error)})
        if result.get("amcl_status") != "exported" and result.get("odom_status") != "exported":
            result["status"] = "not_exported"
            result["detail"] = str(result.get("amcl_detail") or result.get("odom_detail") or "No trajectory topics")
        rows.append(result)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest_path = output / "manifest.csv"
    if append_manifest and manifest_path.exists():
        old = pd.read_csv(manifest_path)
        manifest = pd.concat((old, manifest), ignore_index=True).drop_duplicates("bag_path", keep="last")
    manifest = manifest.sort_values(["session", "bag_kind", "bag_path"])
    manifest.to_csv(manifest_path, index=False)
    return manifest


def ros_bag_topic_inventory(root: Path, sessions: list[int] | None = None):
    """List every stored topic, type, count, and storage-time bound."""
    import pandas as pd

    bag_root = root / "data" / "Rosbags"
    bag_paths = sorted({path.parent for path in bag_root.rglob("*.db3")})
    rows: list[dict[str, object]] = []
    for bag_path in bag_paths:
        session, bag_kind = _bag_session_and_kind(root, bag_path)
        if sessions and session not in sessions:
            continue
        common = {
            "session": session,
            "bag_kind": bag_kind,
            "bag_path": str(bag_path.relative_to(root)),
            "db3_segments": len(list(bag_path.glob("*.db3"))),
        }
        try:
            for row in _topic_inventory_rows(bag_path):
                entry = {**common, "status": "ok", "detail": "", **row}
                for column in ("storage_timestamp_start_ns", "storage_timestamp_end_ns"):
                    value = entry[column]
                    entry[column.removesuffix("_ns") + "_utc"] = _epoch_ns_to_utc(int(value)) if value is not None else None
                    # CSV readers can round large integers; keep nanoseconds as text.
                    entry[column] = str(value) if value is not None else None
                rows.append(entry)
        except (sqlite3.DatabaseError, ValueError) as error:
            rows.append({**common, "status": "unreadable", "detail": str(error)})
    return pd.DataFrame(rows).sort_values(["session", "bag_kind", "bag_path", "topic"], na_position="last")


def _header_timestamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _frame_inventory_for_bag(bag_path: Path) -> dict[str, object]:
    """Read navigation headers and frame IDs."""
    try:
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
        from tf2_msgs.msg import TFMessage
        from rclpy.serialization import deserialize_message
    except ImportError as error:
        raise RuntimeError("Frame inspection requires a sourced ROS 2 environment (geometry_msgs, nav_msgs, sensor_msgs, tf2_msgs, rclpy)") from error

    message_classes = {
        _AMCL_POSE_TYPE: PoseWithCovarianceStamped,
        "nav_msgs/msg/Odometry": Odometry,
        "sensor_msgs/msg/LaserScan": LaserScan,
        "tf2_msgs/msg/TFMessage": TFMessage,
    }
    values: dict[str, set[str]] = {
        "amcl_frame_ids": set(), "odom_frame_ids": set(), "odom_child_frame_ids": set(),
        "scan_frame_ids": set(), "tf_frame_pairs": set(), "tf_static_frame_pairs": set(),
    }
    offsets: dict[str, list[int]] = {"amcl": [], "odom": [], "scan": [], "tf": [], "tf_static": []}
    topic_names: dict[str, set[str]] = {"amcl": set(), "odom": set(), "scan": set(), "tf": set(), "tf_static": set()}
    for database in sorted(bag_path.glob("*.db3")):
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            topics = connection.execute("SELECT id, name, type FROM topics").fetchall()
            for topic_id, name, message_type in topics:
                message_class = message_classes.get(message_type)
                if message_class is None:
                    continue
                for timestamp, data in connection.execute(
                    "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic_id,)
                ):
                    message = deserialize_message(data, message_class)
                    if message_type == _AMCL_POSE_TYPE:
                        key = "amcl"
                        values["amcl_frame_ids"].add(message.header.frame_id)
                        header_ns = _header_timestamp_ns(message.header)
                    elif message_type == "nav_msgs/msg/Odometry":
                        key = "odom"
                        values["odom_frame_ids"].add(message.header.frame_id)
                        values["odom_child_frame_ids"].add(message.child_frame_id)
                        header_ns = _header_timestamp_ns(message.header)
                    elif message_type == "sensor_msgs/msg/LaserScan":
                        key = "scan"
                        values["scan_frame_ids"].add(message.header.frame_id)
                        header_ns = _header_timestamp_ns(message.header)
                    else:
                        key = "tf_static" if str(name).endswith("/tf_static") else "tf"
                        for transform in message.transforms:
                            values[f"{key}_frame_pairs"].add(f"{transform.header.frame_id}->{transform.child_frame_id}")
                            offsets[key].append(_header_timestamp_ns(transform.header) - int(timestamp))
                        topic_names[key].add(str(name))
                        continue
                    topic_names[key].add(str(name))
                    offsets[key].append(header_ns - int(timestamp))
        finally:
            connection.close()
    result: dict[str, object] = {}
    for key, names in topic_names.items():
        result[f"{key}_topics"] = ";".join(sorted(names))
        result[f"{key}_header_minus_storage_min_ns"] = min(offsets[key]) if offsets[key] else None
        result[f"{key}_header_minus_storage_max_ns"] = max(offsets[key]) if offsets[key] else None
    for key, frame_values in values.items():
        result[key] = ";".join(sorted(value for value in frame_values if value))
    return result


def ros_bag_frame_inventory(root: Path, sessions: list[int] | None = None):
    """Return full navigation-frame and header/storage timestamp inspection."""
    import pandas as pd

    bag_root = root / "data" / "Rosbags"
    bag_paths = sorted({path.parent for path in bag_root.rglob("*.db3")})
    rows: list[dict[str, object]] = []
    for bag_path in bag_paths:
        session, bag_kind = _bag_session_and_kind(root, bag_path)
        if sessions and session not in sessions:
            continue
        common = {"session": session, "bag_kind": bag_kind, "bag_path": str(bag_path.relative_to(root))}
        try:
            rows.append({**common, "status": "ok", "detail": "", **_frame_inventory_for_bag(bag_path)})
        except (sqlite3.DatabaseError, RuntimeError, ValueError) as error:
            rows.append({**common, "status": "unreadable", "detail": str(error)})
    return pd.DataFrame(rows).sort_values(["session", "bag_kind", "bag_path"], na_position="last")


def ros_bag_inventory(root: Path):
    """Summarize the ROS 2 bags."""
    import pandas as pd

    bag_root = root / "data" / "Rosbags"
    bag_paths = sorted({path.parent for path in bag_root.rglob("*.db3")})
    rows: list[dict[str, object]] = []
    for bag_path in bag_paths:
        session, bag_kind = _bag_session_and_kind(root, bag_path)
        common = {
            "session": session,
            "bag_kind": bag_kind,
            "bag_path": str(bag_path.relative_to(root)),
            "db3_segments": len(list(bag_path.glob("*.db3"))),
        }
        try:
            topic_inventory = _topic_inventory_rows(bag_path)
            topics = [
                (str(row["topic"]), str(row["message_type"]), int(row["message_count"]))
                for row in topic_inventory
            ]
        except (sqlite3.DatabaseError, ValueError) as error:
            rows.append({**common, "status": "unreadable", "detail": str(error)})
            continue
        counts = {name: count for name, _, count in topics}
        amcl = [name for name, message_type, count in topics if count and message_type == _AMCL_POSE_TYPE and (name == "/amcl_pose" or name.endswith("/amcl_pose"))]
        def topic_summary(suffix: str) -> tuple[str, int]:
            names = [name for name in counts if name == suffix or name.endswith(suffix)]
            return ";".join(names), sum(counts[name] for name in names)
        odom_name, odom_count = topic_summary("/odom")
        scan_name, scan_count = topic_summary("/scan")
        tf_name, tf_count = topic_summary("/tf")
        storage_start = min(
            int(row["storage_timestamp_start_ns"]) for row in topic_inventory
            if row["storage_timestamp_start_ns"] is not None
        )
        storage_end = max(
            int(row["storage_timestamp_end_ns"]) for row in topic_inventory
            if row["storage_timestamp_end_ns"] is not None
        )
        rows.append({
            **common,
            "status": "ok",
            "detail": "",
            "topic_count": len(topics),
            "total_messages": sum(count for _, _, count in topics),
            "storage_timestamp_start_ns": str(storage_start),
            "storage_timestamp_end_ns": str(storage_end),
            "storage_timestamp_start_utc": _epoch_ns_to_utc(storage_start),
            "storage_timestamp_end_utc": _epoch_ns_to_utc(storage_end),
            "storage_duration_s": (storage_end - storage_start) / 1e9,
            "amcl_pose_topic": ";".join(amcl),
            "amcl_pose_messages": sum(counts[name] for name in amcl),
            "odom_topic": odom_name,
            "odom_messages": odom_count,
            "scan_topic": scan_name,
            "scan_messages": scan_count,
            "tf_topic": tf_name,
            "tf_messages": tf_count,
        })
    return pd.DataFrame(rows).sort_values(["session", "bag_kind", "bag_path"], na_position="last")


def _bag_session_and_kind(root: Path, bag_path: Path) -> tuple[int, str]:
    """Get session number and bag kind from a bag path."""
    relative = bag_path.relative_to(root / "data" / "Rosbags")
    kind = relative.parts[0]
    for part in relative.parts:
        if part.startswith("session "):
            return int(part.removeprefix("session ")), kind
    raise ValueError(f"Cannot identify session from ROS bag path {bag_path}")


def _bag_span(bag_path: Path) -> tuple[float, float]:
    """Return the storage-time bounds."""
    spans: list[tuple[int, int]] = []
    for database in sorted(bag_path.glob("*.db3")):
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        lower, upper = connection.execute("SELECT MIN(timestamp), MAX(timestamp) FROM messages").fetchone()
        connection.close()
        if lower is not None and upper is not None:
            spans.append((int(lower), int(upper)))
    if not spans:
        raise ValueError(f"No ROS messages in {bag_path}")
    return min(item[0] for item in spans) / 1e9, max(item[1] for item in spans) / 1e9


def _draw_robot_overlay_map(recording, axis) -> None:
    """Draw the map and stations for a robot overlay."""
    import matplotlib.pyplot as plt

    metadata = recording.dataset.metadata["map"]
    image = recording.dataset.root / metadata["semantic_image"]
    extent = metadata["extent_m"]
    if image.exists():
        axis.imshow(plt.imread(image), extent=extent, origin="upper", alpha=0.55, zorder=0)
    for station, value in recording.dataset.station_regions().items():
        axis.scatter(value["x_m"], value["y_m"], marker="X", color="black", s=36, zorder=5)
        axis.text(value["x_m"] + 0.08, value["y_m"] + 0.08, station, weight="bold", fontsize=8, zorder=5)
    axis.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel="OptiTrack x [m]", ylabel="OptiTrack y [m]")


def plot_ros_optitrack_overlays(dataset, output: Path, *, source: str = "solved", sessions: list[int] | None = None,
                                append_manifest: bool = False):
    """Plot ROS map trajectories with OptiTrack robot tracks."""
    import matplotlib.pyplot as plt
    import pandas as pd

    output.mkdir(parents=True, exist_ok=True)
    bag_root = dataset.root / "data" / "Rosbags"
    bag_paths = sorted({path.parent for path in bag_root.rglob("*.db3")})
    if sessions:
        bag_paths = [path for path in bag_paths if _bag_session_and_kind(dataset.root, path)[0] in sessions]
    transform = dataset.spatial_transform()
    recording_cache = {}
    rows: list[dict[str, object]] = []
    for bag_path in bag_paths:
        session, bag_kind = _bag_session_and_kind(dataset.root, bag_path)
        common = {"session": session, "bag_kind": bag_kind, "bag_path": str(bag_path.relative_to(dataset.root))}
        try:
            bag_start, bag_end = _bag_span(bag_path)
        except (sqlite3.DatabaseError, ValueError) as error:
            rows.append({**common, "scenario": None, "status": "skipped_unreadable_bag", "detail": str(error)})
            continue
        candidates = []
        for scenario in range(1, 5):
            key = (session, scenario)
            if key not in recording_cache:
                recording_cache[key] = dataset.recording(session, scenario, source)
            recording = recording_cache[key]
            goal = recording.spec["temporal_alignment"]["goal_to_optitrack"]
            start = pd.Timestamp(goal["corrected_optitrack_start"]).timestamp()
            overlap = max(0.0, min(bag_end, start + recording.duration_s) - max(bag_start, start))
            candidates.append((overlap, scenario, recording, start))
        overlap_s, scenario, recording, corrected_start = max(candidates, key=lambda item: item[0])
        if overlap_s <= 0:
            rows.append({**common, "scenario": scenario, "status": "skipped_no_recording_overlap", "detail": "no overlap with any same-session recording interval"})
            continue
        try:
            ros = read_amcl_pose(bag_path)
        except (sqlite3.DatabaseError, ValueError, RuntimeError) as error:
            rows.append({**common, "scenario": scenario, "status": "skipped_no_amcl_pose", "detail": str(error), "clock_overlap_s": overlap_s})
            continue
        ros[["optitrack_x", "optitrack_y"]] = transform.apply(ros[["x", "y"]].to_numpy(float))
        amcl_topics = ", ".join(ros.attrs["amcl_topics"])
        data = recording.trimmed()
        figure, axis = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
        _draw_robot_overlay_map(recording, axis)
        expected_robot = {"base": "Robot01", "delivery": "Robot02"}.get(bag_kind)
        robots = [expected_robot] if expected_robot in recording.robots else recording.robots
        robot_colors = ("black", "tab:blue", "tab:green", "tab:red")
        for color, robot in zip(robot_colors, robots):
            xy = data[[f"{robot}.position.x", f"{robot}.position.y"]].to_numpy(float)
            finite = np.isfinite(xy).all(axis=1)
            if finite.any():
                axis.plot(xy[finite, 0], xy[finite, 1], color=color, lw=1.5, label=f"OptiTrack {robot}", zorder=3)
        axis.plot(ros.optitrack_x, ros.optitrack_y, color="tab:orange", lw=1.1, alpha=0.85,
                  label=f"ROS {bag_kind} {amcl_topics} → OptiTrack", zorder=4)
        axis.set_title(f"S{session} scenario {scenario} — {bag_kind} bag\nspatial overlay only; global map transform")
        axis.legend(fontsize=8, loc="best")
        filename = f"s{session}_scenario{scenario}_{bag_kind}_{bag_path.parent.name}_overlay.png"
        path = output / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        output_path = str(path.relative_to(dataset.root)) if path.is_relative_to(dataset.root) else str(path)
        rows.append({**common, "scenario": scenario, "status": "spatial_overlay", "detail": "global map transform applied; no temporal offset applied",
                     "clock_overlap_s": overlap_s, "ros_bag_start": pd.Timestamp(bag_start, unit="s", tz="Europe/Helsinki").isoformat(),
                     "ros_bag_end": pd.Timestamp(bag_end, unit="s", tz="Europe/Helsinki").isoformat(),
                     "optitrack_start": pd.Timestamp(corrected_start, unit="s", tz="Europe/Helsinki").isoformat(),
                     "optitrack_end": pd.Timestamp(corrected_start + recording.duration_s, unit="s", tz="Europe/Helsinki").isoformat(),
                     "ros_minus_optitrack_start_s": bag_start - corrected_start,
                     "ros_minus_optitrack_end_s": bag_end - (corrected_start + recording.duration_s),
                     "optitrack_robot": ",".join(robots), "amcl_topics": amcl_topics,
                     "amcl_samples_plotted": len(ros), "output": output_path})
    result = pd.DataFrame(rows)
    manifest = output / "manifest.csv"
    combined = result
    if append_manifest and manifest.exists():
        old = pd.read_csv(manifest)
        combined = pd.concat((old, result), ignore_index=True).drop_duplicates("bag_path", keep="last")
    combined = combined.sort_values(["session", "bag_kind", "bag_path"], na_position="last")
    combined.to_csv(manifest, index=False)
    time_columns = [column for column in ("session", "scenario", "bag_kind", "bag_path", "ros_bag_start", "ros_bag_end", "optitrack_start", "optitrack_end", "ros_minus_optitrack_start_s", "ros_minus_optitrack_end_s", "clock_overlap_s", "status") if column in combined]
    combined[time_columns].to_csv(output / "ros_bag_recording_time_comparison.csv", index=False)
    return result
