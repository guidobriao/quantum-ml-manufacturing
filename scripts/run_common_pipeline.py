#!/usr/bin/env python3
"""Run the local common pipeline through power/PLC temporal fusion only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qml_thesis.common_pipeline import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/common_pipeline.yaml"),
        help="Repository-relative pipeline configuration",
    )
    args = parser.parse_args()
    summary = run(args.config.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
