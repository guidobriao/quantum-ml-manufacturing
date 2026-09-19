"""Smoke test: full idle-duration pipeline on a tiny slice. Target < 2 minutes."""
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from qml_thesis.idle_duration import run

SMOKE_CONFIG = {
    "version": 1,
    "seed": 20260915,
    "source": {
        "ground_truth_sqlite": str(ROOT / "data/processed/ground_truth.sqlite"),
        "common_sqlite": str(ROOT / "data/processed/common_pipeline.sqlite"),
    },
    "dataset": {
        "minimum_dwell_seconds": 1.0,
        "filters": {
            "experiments": ["Exp20190124", "Exp20190514", "Exp20190617"],
            "stations": [30],
        },
    },
    "features": {
        "lookback_seconds": [1, 10],
        "classical_all": [
            "activepowerl1_1s_mean", "activepowerl1_1s_std",
            "activepowerl1_10s_mean", "activepowerl1_10s_std",
            "flow_1s_mean", "flow_10s_mean", "pressure_1s_mean", "pressure_10s_mean",
            "n_samples_1s", "prev_duration", "time_since_processing_s",
            "cycles_before", "station_id", "hour_of_day", "state_age_seconds",
            "OperationNo", "OrderNo", "PartNumber",
        ],
        "quantum_selected": [
            "activepowerl1_1s_mean", "activepowerl1_10s_mean", "flow_10s_mean",
            "pressure_10s_mean", "prev_duration", "time_since_processing_s",
        ],
    },
    "split": {
        "strategy": "chronological_experiment_holdout",
        "train_experiments": ["Exp20190124"],
        "validation_experiments": ["Exp20190514"],
        "test_experiments": ["Exp20190617"],
    },
    "sampling": {"per_station": 3},
    "classical": {
        "svr": {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"},
        "random_forest": {"n_estimators": 20, "max_depth": 4},
        "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
                    "tree_method": "hist", "device": "cpu"},
    },
    "quantum": {
        "feature_map": {"reps": 2, "entanglement": "linear"},
        "qsvr": {"C_candidates": [1.0], "epsilon_candidates": [0.1],
                 "conditions": [["ideal", None]]},
        "vqr": {"maxiter": 2},
    },
    "decision": {"thresholds_seconds": [28]},
    "outputs": {"sqlite": str(ROOT / "data/processed/smoke_idle.sqlite"),
                "results": str(ROOT / "results/smoke")},
}


def main() -> None:
    started = time.time()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
        yaml.safe_dump(SMOKE_CONFIG, stream)
        config_path = stream.name

    summary = run(Path(config_path))
    print(f"[smoke] pipeline completed in {time.time() - started:.1f}s: {summary}")

    import sqlite3
    import pandas as pd
    connection = sqlite3.connect(SMOKE_CONFIG["outputs"]["sqlite"])
    metrics = pd.read_sql("SELECT * FROM idle_metrics", connection)
    decision = pd.read_sql("SELECT * FROM idle_decision_metrics", connection)
    predictions = pd.read_sql("SELECT * FROM idle_predictions", connection)
    connection.close()

    expected_models = ["naive_station_median", "naive_previous_idle",
                       "svr", "random_forest", "xgboost",
                       "svr_qml_subset", "QSVR",
                       "svr_paired_6feat", "mlp_paired_6feat", "VQR"]
    missing = [f for f in expected_models if f not in set(metrics["model"])]
    assert not missing, f"SMOKE FAIL: models missing from metrics: {missing}"
    expected_conditions = {"naive", "classical_full", "classical", "ideal"}
    actual_conditions = set(metrics["condition"])
    assert expected_conditions <= actual_conditions, (
        f"SMOKE FAIL: conditions missing: {expected_conditions - actual_conditions}")
    assert len(decision) > 0, "SMOKE FAIL: empty decision metrics"
    assert len(predictions) > 0, "SMOKE FAIL: empty predictions"
    assert (predictions["split"] == "test").any(), "SMOKE FAIL: no test predictions"

    print("[smoke] models produced:", sorted(set(metrics["model"])))
    print("[smoke] decision rows:", len(decision))
    print("[smoke] PASS")

    Path(SMOKE_CONFIG["outputs"]["sqlite"]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()