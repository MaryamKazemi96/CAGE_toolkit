# CAGE Toolkit

Python tools for the CAGE multi-modal dataset.

## What the toolkit does

The toolkit reads raw and solved OptiTrack exports without changing either.
For analysis, it exposes the horizontal arena plane as `x, y` (the native
OptiTrack export stores height in `Y` and horizontal depth in `Z`) and applies
cached, per-body visibility bounds to solved trajectories.
 ROS map-frame AMCL poses can be
transformed into the OptiTrack frame with the released spatial transform;
odometry stays in its recorded `odom` frame.

## Install

The repository includes small required assets under `data/` and `results/`:
the `env` environment figures and the validated spatial transform. Download
the recordings separately; do not add raw OptiTrack takes or ROS bags here.
From a local CAGE dataset root (the directory containing its `data/` and
`results/` folders), install the toolkit:

```bash
python3 -m pip install -e toolkit
cage metadata build
```

The core dependencies are NumPy, pandas, PyYAML, matplotlib, and SciPy. ROS 2
is not required for core trajectory use. A ROS installation is only needed if
you choose to extract `/amcl_pose` yourself from bags.

## Quick start

```bash
# Examine take duration, people/robots, cached bounds, and alignment evidence.
cage --root . inspect --session 2 --scenario 4

# Plot a single trajectory export. This loads only that export plus metadata.
cage --root . plot trajectory --session 2 --scenario 4 --source solved \
  --output results/example_s2_sc4.png

# Latest goal assignments over trajectories. An assignment is not an arrival.
cage --root . plot goals --session 2 --scenario 4 \
  --output results/example_s2_sc4_goals.png

# Transparent per-person movement summaries.
cage --root . metrics --session 2 --scenario 4 \
  --output results/example_s2_sc4_metrics.csv

# Four-scenario session figure.
cage --root . paper-figures --session 2

# Compact, reproducible survey cleaning and density summary.
cage --root . survey
```

For development without installation, prefix commands with
`PYTHONPATH=toolkit python3 -m cage_toolkit`.

## Why there is a metadata index

`config/recording_metadata.yaml` is a generated, human-readable index of all
32 recordings. It stores take paths/header metadata, participant/robot lists,
time-alignment evidence, station regions, and cached trajectory bounds.

The index performs the expensive/legacy raw-visibility check once. The cached
**analysis bounds** are derived from raw visibility (`x > waiting_area_x`, runs
of at least 30 frames), because solved files can fill a person’s missing
measurements. Therefore `dataset.recording(..., source="solved").trimmed()`
loads the solved take only; it does not load raw data. The raw and solved
source-native bounds are retained in the YAML for auditing.

Regenerate the index only after changing input takes or `config/recordings.yaml`:

```bash
cage --root /path/to/CAGE metadata build
```

This command reads data but never edits CSV, TAK, bag, map, or goal-log files.

## Time and coordinate conventions

All trajectory functions normalize the CAGE OptiTrack export to horizontal
`position.x` / `position.y`; the source export’s vertical Y becomes `position.z`.
Times in OptiTrack CSVs are elapsed seconds from their take start.


## ATC-compatible pedestrian CSV

Export one recording in the eight-column ATC tracking layout:

```bash
PYTHONPATH=toolkit python3 toolkit/scripts/export_optitrack_atc.py \
  --root . --session 2 --scenario 4 --source solved \
  --output results/atc_export/cage-s02-sc04.csv
```

The CSV has no header. Its fields are `time`, `person_id`, `x`, `y`, `z`,
`velocity`, `motion_angle`, and `facing_angle`, in the same order as ATC.
Time is Unix seconds with millisecond precision; positions and velocity are in
millimetres and millimetres/second. A metadata JSON file is written beside the
CSV.

## ROS 2 bags

Each bag directory contains ROS 2 SQLite storage (`metadata.yaml` plus one or
more `.db3` segments). Bags are grouped as `data/Rosbags/base/session N/` and
`data/Rosbags/delivery/session N/`. The former is compared to OptiTrack
`Robot01`; the latter to `Robot02`.

Start with an inventory. These commands only read SQLite metadata and messages;
they do not require ROS 2 and never change a bag:

```bash
# One row per bag: health, segments, duration, and navigation-topic counts.
cage --root . ros inventory

# One row per recorded topic: exact name, ROS type, message count, and the
# first/last SQLite storage timestamp (nanoseconds since the Unix epoch).
cage --root . ros topics
```

The outputs are `results/ros_bag_inventory.csv`.

### ROS interface and conventions

Most CAGE bags use root topics such as `/amcl_pose`, `/odom`, and `/scan`.
S8/scenario 4 base instead records the equivalent `/robot1/...` namespace
(`'/robot1/amcl_pose'`, `'/robot1/odom'`, `'/robot1/scan'`, `'/robot1/tf'`).
This is retained in reports because it is meaningful provenance. Toolkit AMCL
readers discover both root and namespaced `*/amcl_pose` topics automatically,
including across split `.db3` files; users should not hard-code `/amcl_pose`.
Only this AMCL-topic selection is abstracted: other topic names remain exactly
as recorded, so use the topic inventory when writing custom ROS code.

To inspect actual frame IDs and the difference between a message's header stamp
and its SQLite storage time, source ROS 2 and run:

```bash
source /opt/ros/humble/setup.bash  # use the installed ROS 2 distribution
cage --root . ros frames
```

This reads every AMCL, odometry, scan, and TF message and writes
`results/ros_bag_frame_inventory.csv`. Header timestamps and storage timestamps
are reported separately; the toolkit never substitutes one for the other or
assumes they share an OptiTrack clock. Standard ROS inspection is also
available, for example `ros2 bag info <bag-directory>` and
`ros2 bag play <bag-directory>` in a compatible ROS 2 environment.

### Trajectory CSV exports

To create ready-to-use trajectory tables for every readable bag:

```bash
source /opt/ros/humble/setup.bash
cage --root . ros export-trajectories
```

This creates `results/ros_trajectory_exports/`. Each bag has its own folder
and the folder contains:

- `amcl_pose.csv`: AMCL positions in the recorded `map` frame. Columns include
  storage and header timestamps, the source topic and frame, position, and
  orientation.
- `odometry.csv`: odometry positions, orientation, linear/angular velocity,
  and the recorded `odom` and child frames. These coordinates are not map or
  OptiTrack coordinates.
- `manifest.csv`: one row per bag showing which CSVs were produced and why a
  topic could not be exported.

The default exports keep AMCL in the ROS map frame. To add derived OptiTrack
coordinates beside the original map coordinates, use:

```bash
cage --root . ros export-trajectories --include-optitrack-coordinates
```

This option adds `optitrack_x` and `optitrack_y` only to `amcl_pose.csv`; it
does not transform odometry. A single bag can also be exported with
`cage --root . spatial export-amcl <bag> <output.csv>` or
`cage --root . spatial export-odom <bag> <output.csv>`.



To transform an exported map-frame pose CSV (`x,y` columns) without modifying a
bag:

```bash
cage --root . spatial transform-csv poses_from_amcl.csv results/poses_optitrack.csv

# Or, from a sourced ROS 2 environment, export root or namespaced AMCL poses
# directly from a bag. The output retains `timestamp_epoch_s`, `x`, and `y`,
# and adds `optitrack_x`, `optitrack_y` when the transform is supplied.
cage --root . spatial export-amcl \
  data/Rosbags/delivery/session\ 2/session_2026-05-20_15-34-47/session \
  results/s2_amcl_optitrack.csv
```

The same command accepts the namespaced, split-bag case without a special
flag:

```bash
cage --root . spatial export-amcl \
  data/Rosbags/base/session\ 8/session_2026-06-05_15-36-51/split_bag \
  results/s8_sc4_base_amcl_optitrack.csv
```

For programmatic map-pose access in a sourced ROS 2 environment:

```python
from pathlib import Path
from cage_toolkit.ros import read_amcl_pose

poses = read_amcl_pose(Path("data/Rosbags/base/session 8/"
                            "session_2026-06-05_15-36-51/split_bag"))
print(poses.attrs["amcl_topics"])  # ['/robot1/amcl_pose']
# columns: timestamp_epoch_s, x, y (all x/y remain in the ROS map frame)
```

To render every readable ROS bag against the matching *solved* OptiTrack robot
track, using the global spatial transform, run:

```bash
cage --root . plot ros-overlay
```

This writes in `results/spatial_alignment/ros_optitrack_overlays/`.

## Python API

```python
from cage_toolkit import CAGEDataset

dataset = CAGEDataset(".")
recording = dataset.recording(session=2, scenario=4, source="solved")
trajectory = recording.trimmed()  # solved file only; uses YAML-cached bounds
print(recording.duration_s)

ros_map_to_opti = dataset.spatial_transform()
optitrack_xy = ros_map_to_opti.apply([[0.0, 0.0]])
```

## Layout and scope

The toolkit expects the following published layout:

```text
data/OptiTrack/{raw,solved}/session N/*.csv
data/goals/
data/Rosbags/                 # optional for core use
data/survey/survey.csv        # optional for survey command


## License


## Acknowledgement

This work was developed as part of research on human–robot coexistence and multi-robot systems at Aalto University.

## Citation

If this toolkit or the CAGE data contributes to published work, cite the CAGE
dataset paper. 

## Contact

The cage toolkit has been developed by Maryam Kazemi. For questions, issues, or further information, please contact: maryam.kazemieskeri@aalto.fi.