#!/usr/bin/env python3
"""Export one CAGE OptiTrack recording in the ATC tracking CSV layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from cage_toolkit.atc import export_atc
from cage_toolkit.recording import CAGEDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--scenario", type=int, required=True, choices=range(1, 5))
    parser.add_argument("--source", choices=("solved", "raw"), default="solved")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = CAGEDataset(args.root)
    metadata = export_atc(dataset.recording(args.session, args.scenario, args.source), args.output)
    print(f"Wrote {metadata['rows']} rows to {args.output}")


if __name__ == "__main__":
    main()
