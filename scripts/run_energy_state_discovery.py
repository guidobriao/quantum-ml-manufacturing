#!/usr/bin/env python3
"""Run the local classical energy-state discovery stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qml_thesis.energy_state_discovery import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/energy_state_discovery.yaml"),
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
