"""Run the ground-truth labelling stage (phase-2)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse

from qml_thesis.ground_truth_labels import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground-truth labelling stage")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ground_truth.yaml")
    print(run(parser.parse_args().config))
