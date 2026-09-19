"""Run the parallel 2x2 experiment (phase-2)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse

from qml_thesis.parallel_experiment import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel 2x2 experiment")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "parallel_experiment.yaml")
    print(run(parser.parse_args().config))