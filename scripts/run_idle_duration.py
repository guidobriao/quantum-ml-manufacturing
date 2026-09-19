"""Run the idle-duration regression stage (phase-2)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse

from qml_thesis.idle_duration import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Idle-duration regression stage")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "idle_duration.yaml")
    summary = run(parser.parse_args().config)
    print(summary)
