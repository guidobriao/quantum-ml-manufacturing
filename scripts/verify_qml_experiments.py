#!/usr/bin/env python3
"""Read-only verification of local QML experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import yaml

from qml_thesis.qml_experiments import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qml_experiments.yaml"
OUTPUT = ROOT / "data/processed/qml_experiments.sqlite"
SUMMARY = ROOT / "results/qml/run_summary.json"


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    for path_key, hash_key in (
        ("discovery_sqlite", "discovery_sqlite_sha256"),
        ("common_sqlite", "common_sqlite_sha256"),
        ("segmentation_config", "segmentation_config_sha256"),
        ("inference_logic", "inference_logic_sha256"),
    ):
        checks[f"frozen_{path_key}_unchanged"] = sha256_file(ROOT / config["frozen_inputs"][path_key]) == config["frozen_inputs"][hash_key]
    with sqlite3.connect(f"file:{OUTPUT}?mode=ro", uri=True) as connection:
        experiments = {
            split: {row[0] for row in connection.execute("SELECT DISTINCT experiment_id FROM split_membership WHERE split=?", (split,))}
            for split in ("train", "validation", "test")
        }
        checks["chronological_experiment_split_exact"] = (
            experiments["train"] == set(config["split"]["train_experiments"])
            and experiments["validation"] == set(config["split"]["validation_experiments"])
            and experiments["test"] == set(config["split"]["test_experiments"])
            and not (experiments["train"] & experiments["validation"] or experiments["train"] & experiments["test"] or experiments["validation"] & experiments["test"])
        )
        ordered = {row[0]: (row[1], row[2]) for row in connection.execute("SELECT split,MIN(segment_start_utc),MAX(segment_end_utc) FROM split_membership GROUP BY split")}
        checks["split_time_order"] = ordered["train"][1] < ordered["validation"][0] < ordered["validation"][1] < ordered["test"][0]
        selected = config["features"]["selected"]
        qml_columns = {row[1] for row in connection.execute("PRAGMA table_info(qml_sample)")}
        forbidden = {"Busy", "Done", "StationEntryxBG5", "ReadyAtStationxBG1", "StationExitxBG6", "OperationNo", "OrderNo"}
        checks["selected_features_present"] = set(selected).issubset(qml_columns)
        checks["plc_mes_absent_from_qml_sample"] = not (qml_columns & forbidden)
        checks["qml_sample_balanced"] = connection.execute(
            "SELECT COUNT(*) FROM (SELECT split,target,COUNT(*) n FROM qml_sample GROUP BY split,target) GROUP BY split HAVING MIN(n)<>MAX(n)"
        ).fetchone() is None
        models = {row[0] for row in connection.execute("SELECT DISTINCT model FROM metrics")}
        checks["all_required_models"] = {"LogisticRegression", "SVM_RBF", "RandomForest", "XGBoost", "QSVC", "VQC"}.issubset(models)
        qml_conditions = {row[0] for row in connection.execute("SELECT DISTINCT condition FROM metrics WHERE model IN ('QSVC','VQC')")}
        checks["all_local_conditions"] = {"ideal", "finite_shots", "noisy"}.issubset(qml_conditions)
        checks["vqc_convergence_recorded"] = connection.execute("SELECT COUNT(*) FROM vqc_convergence").fetchone()[0] > 0
        checks["kernel_clustering_label_free"] = connection.execute("SELECT selection_used_inferred_state FROM quantum_kernel_clustering_metrics").fetchone()[0] == 0
        checks["metrics_complete"] = connection.execute("SELECT COUNT(*) FROM metrics WHERE accuracy IS NULL OR f1_macro IS NULL OR training_seconds IS NULL OR inference_seconds IS NULL").fetchone()[0] == 0
    matrix_dir = ROOT / config["outputs"]["matrices"]
    expected = [
        "qsvc_ideal_shots-exact_rep-1.npz", "qsvc_finite_shots_shots-256_rep-1.npz",
        "qsvc_finite_shots_shots-1024_rep-1.npz", "qsvc_noisy_shots-1024_rep-1.npz",
        "quantum_kernel_clustering_ideal.npz",
    ]
    checks["kernel_matrices_saved"] = all((matrix_dir / name).exists() for name in expected)
    ideal = np.load(matrix_dir / "qsvc_ideal_shots-exact_rep-1.npz")
    checks["kernel_shapes_correct"] = ideal["train"].shape == (40, 40) and ideal["validation"].shape == (16, 40) and ideal["test"].shape == (24, 40)
    checks["no_remote_or_qpu_usage_reported"] = not summary["ibm_runtime_service_used"] and not summary["real_backend_used"] and not summary["qpu_used"]
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
