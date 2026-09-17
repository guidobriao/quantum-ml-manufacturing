#!/usr/bin/env python3
"""Run local classical/QML experiments without any hardware connection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qml_thesis.qml_experiments import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qml_experiments.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
