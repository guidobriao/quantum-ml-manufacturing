"""Idle-duration residual regression pipeline (phase-2).

Trigger: entry into an idle state (plc_state_machine.IDLE_STATES).
Target: duration in seconds of the idle state segment (log1p-transformed).
Inputs (all strictly backward-looking, as-of trigger time):
  - multi-scale energy/sensor statistics on [t-lookback, t];
  - previous-state context (state, duration) within the same PLC session;
  - time since last processing end, processing cycles so far in session;
  - MES as-of fields at trigger (state_age, OperationNo, OrderNo, PartNumber);
  - station_id, hour of day.
Labels (durations) use future information: legitimate annotation.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from .common_io import _write_csv
from .plc_state_machine import IDLE_STATES

ENERGY_COLUMNS = ["ActivePowerL1", "Flow", "Pressure"]
MES_ASOF = ["state_age_seconds", "OperationNo", "OrderNo", "PartNumber"]
CONTEXT_COLUMNS = [
    "prev_state", "prev_duration", "time_since_processing_s",
    "cycles_before", "station_id", "hour_of_day",
]


# ------------------------------------------------------------------ dataset

def load_state_segments(ground_truth_sqlite: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{ground_truth_sqlite}?mode=ro", uri=True) as connection:
        return pd.read_sql_query("SELECT * FROM gt_state_segments", connection)


def load_fused(common_sqlite: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{common_sqlite}?mode=ro", uri=True) as connection:
        return pd.read_sql_query(
            "SELECT experiment_id, station_id, session_id, timestamp_epoch_ms_original, "
            "ActivePowerL1, Flow, Pressure, state_age_seconds, "
            "OperationNo, OrderNo, PartNumber FROM power_operation_fused",
            connection,
        )


def add_context(segments: pd.DataFrame) -> pd.DataFrame:
    """Previous-state context within each experiment/station/session group."""
    keys = ["experiment_id", "station_id", "session_id"]
    work = segments.sort_values(keys + ["start_ms"]).copy()
    grouped = work.groupby(keys, sort=False)
    work["prev_state"] = grouped["plc_state"].shift(1)
    work["prev_duration"] = grouped["duration_seconds"].shift(1)
    is_processing = work["plc_state"].eq("processing")
    work["cycles_before"] = (
        is_processing.groupby([work[k] for k in keys]).cumsum()
        - is_processing.astype(int)
    )
    proc_end = work["end_ms"].where(is_processing)
    work["last_processing_end_ms"] = proc_end.groupby([work[k] for k in keys]).ffill()
    work["time_since_processing_s"] = (
        (work["start_ms"] - work["last_processing_end_ms"]) / 1000.0
    )
    return work


def energy_features(idle: pd.DataFrame, fused: pd.DataFrame,
                    lookback_seconds: list[int]) -> pd.DataFrame:
    """Multi-scale backward statistics of energy/sensor channels at each trigger."""
    fused_sorted = fused.sort_values(
        ["experiment_id", "station_id", "session_id", "timestamp_epoch_ms_original"]
    ).copy()
    arrays: dict[tuple, tuple[np.ndarray, pd.DataFrame]] = {}
    for key, group in fused_sorted.groupby(
            ["experiment_id", "station_id", "session_id"], sort=False):
        arrays[key] = (group["timestamp_epoch_ms_original"].to_numpy(), group)

    features = []
    for record in idle.itertuples(index=False):
        key = (record.experiment_id, record.station_id, record.session_id)
        if key not in arrays:
            features.append({})
            continue
        timestamps, group = arrays[key]
        trigger_ms = int(record.start_ms)
        row: dict[str, float] = {}
        left = np.searchsorted(timestamps, trigger_ms, side="right")
        for scale in lookback_seconds:
            lower = np.searchsorted(timestamps, trigger_ms - scale * 1000, side="left")
            window = group.iloc[lower:left]
            for column in ENERGY_COLUMNS:
                series = window[column].dropna()
                prefix = f"{column.lower()}_{scale}s"
                row[f"{prefix}_mean"] = float(series.mean()) if len(series) else np.nan
                row[f"{prefix}_std"] = float(series.std()) if len(series) > 1 else 0.0
            row[f"n_samples_{scale}s"] = int(len(window))
        features.append(row)
    return pd.DataFrame(features, index=idle.index)


def build_dataset(config: dict) -> pd.DataFrame:
    segments = add_context(load_state_segments(Path(config["source"]["ground_truth_sqlite"])))
    filters = config["dataset"].get("filters") or {}
    if filters.get("experiments"):
        segments = segments[segments["experiment_id"].isin(filters["experiments"])]
    if filters.get("stations"):
        segments = segments[segments["station_id"].isin(filters["stations"])]
    minimum_dwell = config["dataset"]["minimum_dwell_seconds"]
    idle = segments[
        segments["plc_state"].isin(IDLE_STATES)
        & (segments["duration_seconds"] >= minimum_dwell)
    ].reset_index(drop=True)

    fused = load_fused(Path(config["source"]["common_sqlite"]))
    energy = energy_features(idle, fused, config["features"]["lookback_seconds"])

    # MES as-of at trigger: row of fused closest at or before trigger within group.
    mes_rows = []
    fused_sorted = fused.sort_values(
        ["experiment_id", "station_id", "session_id", "timestamp_epoch_ms_original"])
    for record in idle.itertuples(index=False):
        group = fused_sorted[
            (fused_sorted["experiment_id"] == record.experiment_id)
            & (fused_sorted["station_id"] == record.station_id)
            & (fused_sorted["session_id"] == record.session_id)
        ]
        position = group["timestamp_epoch_ms_original"].searchsorted(
            int(record.start_ms), side="right") - 1
        if position >= 0:
            mes_rows.append(group.iloc[position][MES_ASOF].to_dict())
        else:
            mes_rows.append({column: np.nan for column in MES_ASOF})

    dataset = pd.concat(
        [idle[["experiment_id", "station_id", "session_id", "start_ms", "duration_seconds"]],
         energy, pd.DataFrame(mes_rows)], axis=1)
    dataset["prev_state"] = idle["prev_state"].values
    dataset["prev_duration"] = idle["prev_duration"].values
    dataset["time_since_processing_s"] = idle["time_since_processing_s"].values
    dataset["cycles_before"] = idle["cycles_before"].values
    dataset["hour_of_day"] = pd.to_datetime(dataset["start_ms"], unit="ms").dt.hour
    dataset["target_log1p"] = np.log1p(dataset["duration_seconds"])
    return dataset


# ------------------------------------------------------------------ sampling

def duration_quantile_sample(frame: pd.DataFrame, per_station: int, seed: int) -> pd.DataFrame:
    """Deterministic per-station sample stratified by duration quantile bins."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in frame.groupby("station_id", sort=False):
        ordered = group.sort_values("target_log1p")
        bins = np.array_split(ordered.index.to_numpy(), max(per_station, 1))
        chosen: list = []
        for bin_indices in bins:
            if len(bin_indices):
                chosen.append(int(rng.choice(bin_indices)))
        parts.append(group.loc[sorted(chosen)])
    return pd.concat(parts)


def assign_split_by_experiment(frame: pd.DataFrame, split_config: dict) -> pd.DataFrame:
    """Chronological experiment-level split, no label mapping needed here."""
    sets = {
        experiment: split
        for split, name in [("train", "train_experiments"),
                            ("validation", "validation_experiments"),
                            ("test", "test_experiments")]
        for experiment in split_config[name]
    }
    work = frame.copy()
    work["split"] = work["experiment_id"].map(sets).fillna("excluded")
    return work[work["split"] != "excluded"].reset_index(drop=True)


# ------------------------------------------------------------------ metrics

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics on log1p targets, reported back in original seconds."""
    true_s = np.expm1(y_true)
    pred_s = np.expm1(y_pred)
    return {
        "mae_s": float(np.mean(np.abs(true_s - pred_s))),
        "rmse_s": float(np.sqrt(np.mean((true_s - pred_s) ** 2))),
        "median_ae_s": float(np.median(np.abs(true_s - pred_s))),
        "mae_log": float(np.mean(np.abs(y_true - y_pred))),
    }


# ------------------------------------------------------------------ baselines

def naive_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add naive baseline prediction columns to dataset (no metric computation)."""
    train = dataset[dataset["split"] == "train"]
    medians = train.groupby("station_id")["duration_seconds"].median()
    global_median = train["duration_seconds"].median()

    # previous idle: chronological sequence over the WHOLE dataset (backward-looking)
    ordered = dataset.sort_values("start_ms").copy()
    previous: dict[int, float] = {}
    preds = []
    for record in ordered.itertuples(index=False):
        preds.append(previous.get(int(record.station_id), global_median))
        previous[int(record.station_id)] = record.duration_seconds
    ordered["naive_previous_idle"] = preds
    dataset = dataset.merge(
        ordered[["start_ms", "station_id", "naive_previous_idle"]],
        on=["start_ms", "station_id"], how="left")
    dataset["naive_station_median"] = (
        dataset["station_id"].map(medians).fillna(global_median))
    return dataset


def naive_metrics(dataset: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for naive baselines from dataset with prediction columns."""
    rows = []
    for split in ["train", "validation", "test"]:
        part = dataset[dataset["split"] == split]
        for name in ["naive_station_median", "naive_previous_idle"]:
            metrics = regression_metrics(
                np.log1p(part["duration_seconds"].to_numpy()),
                np.log1p(part[name].to_numpy()))
            rows.append({"model": name, "condition": "naive", "split": split,
                         "n": int(len(part)), **metrics})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ models

def classical_models(config: dict, seed: int) -> dict[str, object]:
    classical = config["classical"]
    return {
        "svr": SVR(**classical["svr"]),
        "random_forest": RandomForestRegressor(
            random_state=seed, **classical["random_forest"]),
        "xgboost": XGBRegressor(random_state=seed, **classical["xgboost"]),
    }


def _as_kernel(result):
    return result[0] if isinstance(result, tuple) else result


def kernel_svr_regression(config: dict, seed: int, arrays: dict, record=None) -> list[dict]:
    """QSVR: FidelityQuantumKernel + SVR(kernel='precomputed')."""
    from qiskit_machine_learning.utils import algorithm_globals
    from qiskit.circuit.library import ZZFeatureMap
    from .qml_common import make_sampler, quantum_kernel

    algorithm_globals.random_seed = seed
    features = config["features"]["quantum_selected"]
    n_qubits = len(features)
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits, reps=config["quantum"]["feature_map"]["reps"],
        entanglement=config["quantum"]["feature_map"]["entanglement"])

    rows = []
    for condition, shots in [("ideal", None),
                              ("finite_shots", 256),
                              ("finite_shots", 1024)]:
        kernel = _as_kernel(quantum_kernel(feature_map, condition, shots, seed))
        x = {split: frame[features].to_numpy() for split, frame in arrays.items()}
        k_train = kernel.evaluate(x["train"])
        k_validation = kernel.evaluate(x["validation"], x["train"])
        k_test = kernel.evaluate(x["test"], x["train"])
        best, best_mae = None, np.inf
        for c in config["quantum"]["qsvr"]["C_candidates"]:
            for epsilon in config["quantum"]["qsvr"]["epsilon_candidates"]:
                model = SVR(kernel="precomputed", C=c, epsilon=epsilon)
                model.fit(k_train, arrays["train"]["target_log1p"].to_numpy())
                prediction = model.predict(k_validation)
                mae = mean_absolute_error(
                    arrays["validation"]["target_log1p"].to_numpy(), prediction)
                if mae < best_mae:
                    best, best_mae = model, mae
        for split, matrix in [("validation", k_validation), ("test", k_test)]:
            prediction = best.predict(matrix)
            if record:
                record("QSVR", f"{condition}_{shots or 'exact'}", split, arrays[split], prediction)
            rows.append({
                "model": "QSVR", "condition": f"{condition}_{shots or 'exact'}",
                "split": split, "n": int(len(arrays[split])),
                **regression_metrics(arrays[split]["target_log1p"].to_numpy(), prediction)})
    return rows


def run(config_path: Path) -> dict[str, object]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seed = config["seed"]

    dataset = build_dataset(config)
    dataset = assign_split_by_experiment(dataset, config["split"])
    dataset = dataset.dropna(subset=config["features"]["classical_all"] + ["target_log1p"])
    dataset = pd.get_dummies(dataset, columns=["prev_state"], dtype=float)
    feature_columns = [c for c in dataset.columns
                       if c in config["features"]["classical_all"] or c.startswith("prev_state_")]

    dataset = naive_predictions(dataset)   # patched: full-history previous idle
    predictions: list[dict] = []

    def record(model, condition, split, frame, prediction, is_log=True):
        values = np.atleast_1d(np.asarray(prediction, dtype=float))
        for row, value in zip(frame.itertuples(index=False), values):
            seconds = float(np.expm1(value)) if is_log else float(value)
            predictions.append({
                "model": model, "condition": condition, "split": split,
                "station_id": int(row.station_id), "start_ms": int(row.start_ms),
                "duration_seconds": float(row.duration_seconds),
                "prediction": seconds,
            })

    rows: list[dict] = []
    for name in ["naive_station_median", "naive_previous_idle"]:
        for split in ["train", "validation", "test"]:
            part = dataset[dataset["split"] == split]
            record(name, "naive", split, part, part[name].to_numpy(), is_log=False)
    rows.extend(naive_metrics(dataset).to_dict("records"))

    classical = classical_models(config, seed)
    for name, model in classical.items():
        train = dataset[dataset["split"] == "train"]
        model.fit(train[feature_columns].to_numpy(), train["target_log1p"].to_numpy())
        for split in ["validation", "test"]:
            part = dataset[dataset["split"] == split]
            prediction = model.predict(part[feature_columns].to_numpy())
            record(name, "classical_full", split, part, prediction)
            rows.append({"model": name, "condition": "classical_full", "split": split,
                         "n": int(len(part)),
                         **regression_metrics(part["target_log1p"].to_numpy(), prediction)})

    subsample = pd.concat(
        [duration_quantile_sample(dataset[dataset["split"] == split],
                                  config["sampling"]["per_station"], seed)
         for split in ["train", "validation", "test"]],
        ignore_index=True,
    )
    arrays = {split: subsample[subsample["split"] == split]
              for split in ["train", "validation", "test"]}
    rows.extend(kernel_svr_regression(config, seed, arrays, record))
    rows.extend(paired_parametric_regression(config, seed, arrays, record))
    for name, model in classical.items():
        model.fit(arrays["train"][feature_columns].to_numpy(),
                  arrays["train"]["target_log1p"].to_numpy())
        for split in ["validation", "test"]:
            prediction = model.predict(arrays[split][feature_columns].to_numpy())
            record(f"{name}_qml_subset", "classical", split, arrays[split], prediction)
            rows.append({"model": f"{name}_qml_subset", "condition": "classical",
                         "split": split, "n": int(len(arrays[split])),
                         **regression_metrics(arrays[split]["target_log1p"].to_numpy(), prediction)})

    metrics = pd.DataFrame(rows)
    prediction_frame = pd.DataFrame(predictions)
    decision = decision_metrics(prediction_frame, config["decision"]["thresholds_seconds"])

    output = Path(config["outputs"]["sqlite"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as connection:
        dataset.to_sql("idle_dataset", connection, index=False, if_exists="replace")
        metrics.to_sql("idle_metrics", connection, index=False, if_exists="replace")
        prediction_frame.to_sql("idle_predictions", connection, index=False, if_exists="replace")
        decision.to_sql("idle_decision_metrics", connection, index=False, if_exists="replace")
        subsample.to_sql("idle_qml_subsample", connection, index=False, if_exists="replace")
    return {"dataset_rows": int(len(dataset)), "metric_rows": int(len(metrics)),
            "decision_rows": int(len(decision))}


def paired_parametric_regression(config, seed, arrays, record) -> list[dict]:
    """Paired comparisons on the QML subsample and the six quantum features:
    SVR(rbf) vs QSVR (kernel pair) and MLPRegressor vs VQR (parametric pair)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR as SkSVR
    from sklearn.preprocessing import MinMaxScaler
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit_machine_learning.optimizers import COBYLA
    from qiskit_machine_learning.algorithms import VQR
    from qiskit_machine_learning.utils import algorithm_globals

    features = config["features"]["quantum_selected"]
    rows = []
    x = {s: arrays[s][features].to_numpy() for s in arrays}
    y = {s: arrays[s]["target_log1p"].to_numpy() for s in arrays}

    paired_svr = SkSVR(kernel="rbf", C=1.0, epsilon=0.1)
    paired_svr.fit(x["train"], y["train"])
    for split in ["validation", "test"]:
        prediction = paired_svr.predict(x[split])
        record("svr_paired_6feat", "classical", split, arrays[split], prediction)
        rows.append({"model": "svr_paired_6feat", "condition": "classical",
                     "split": split, "n": int(len(y[split])),
                     **regression_metrics(y[split], prediction)})

    from sklearn.preprocessing import StandardScaler
    mlp_scaler = StandardScaler()
    xs_scaled = {"train": mlp_scaler.fit_transform(x["train"])}
    for split in ["validation", "test"]:
        xs_scaled[split] = mlp_scaler.transform(x[split])
    mlp = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=3000,
                       early_stopping=False, random_state=seed)
    mlp.fit(xs_scaled["train"], y["train"])
    for split in ["validation", "test"]:
        prediction = mlp.predict(xs_scaled[split])
        record("mlp_paired_6feat", "classical", split, arrays[split], prediction)
        rows.append({"model": "mlp_paired_6feat", "condition": "classical",
                     "split": split, "n": int(len(y[split])),
                     **regression_metrics(y[split], prediction)})

    algorithm_globals.random_seed = seed
    feature_map = ZZFeatureMap(feature_dimension=len(features),
                               reps=config["quantum"]["feature_map"]["reps"],
                               entanglement=config["quantum"]["feature_map"]["entanglement"])
    ansatz = RealAmplitudes(len(features), reps=1, entanglement="linear")
    target_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))  # VQR output is in [-1,1]
    y_train_scaled = target_scaler.fit_transform(y["train"].reshape(-1, 1)).ravel()
    input_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    input_scaler.fit(x["train"])
    xs = {s: input_scaler.transform(x[s]) for s in x}
    vqr = VQR(feature_map=feature_map, ansatz=ansatz,
              optimizer=COBYLA(maxiter=config["quantum"].get("vqr", {}).get("maxiter", 30)),
              loss="squared_error")
    vqr.fit(xs["train"], y_train_scaled)
    for split in ["validation", "test"]:
        pred_scaled = np.asarray(vqr.predict(xs[split])).ravel()
        prediction = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        record("VQR", "ideal", split, arrays[split], prediction)
        rows.append({"model": "VQR", "condition": "ideal", "split": split,
                     "n": int(len(y[split])),
                     **regression_metrics(y[split], prediction)})
    return rows


def decision_metrics(predictions: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    """Threshold-time decision metrics per model, test split, with/without station 50."""
    test = predictions[predictions["split"] == "test"]
    rows = []
    for scope_name, scope in [("all_stations", test),
                              ("stations_10_80", test[test["station_id"] != 50])]:
        for (model, condition), group in scope.groupby(["model", "condition"]):
            for tt in thresholds:
                y = (group["duration_seconds"] > tt).to_numpy()
                yh = (group["prediction"] > tt).to_numpy()
                tp = int((y & yh).sum()); fp = int((~y & yh).sum())
                fn = int((y & ~yh).sum()); tn = int((~y & ~yh).sum())
                rows.append({"model": model, "condition": condition, "scope": scope_name,
                             "tt_seconds": tt, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "precision": tp / (tp + fp) if tp + fp else float("nan"),
                             "recall": tp / (tp + fn) if tp + fn else float("nan"),
                             "accuracy": (tp + tn) / len(group), "n": len(group)})
    return pd.DataFrame(rows)