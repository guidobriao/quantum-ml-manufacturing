"""Local-only classical and QML comparison on frozen inferred-state data."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
import qiskit_machine_learning
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.primitives import QMLSampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.utils import algorithm_globals
import sklearn
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score, confusion_matrix,
    f1_score, normalized_mutual_info_score, precision_score, recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import xgboost
from xgboost import XGBClassifier
import yaml

DISCOVERY_FEATURES = [
    "duration_seconds", "sample_count", "sampling_density_hz",
    "power_mean", "power_min", "power_max", "power_std", "power_median",
    "power_range", "power_delta", "power_slope", "energy_joule",
    "flow_mean", "flow_min", "flow_max", "flow_std", "flow_delta", "flow_slope",
    "pressure_mean", "pressure_min", "pressure_max", "pressure_std",
    "pressure_delta", "pressure_slope",
]


LABEL_MAP = {"idle_ready": 0, "loading_unloading_transfer": 1}
FORMULAS = {
    "duration_seconds": "(segment_end_epoch_ms - segment_start_epoch_ms) / 1000",
    "power_mean": "mean(ActivePowerL1)",
    "power_std": "sample standard deviation(ActivePowerL1), 0 for singleton",
    "power_slope": "OLS slope ActivePowerL1 ~ elapsed seconds",
    "flow_mean": "mean(Flow)",
    "pressure_mean": "mean(Pressure)",
}
REASONS = {
    "duration_seconds": "Required temporal context; low redundancy with power level.",
    "power_mean": "Direct, physically interpretable energy-regime level.",
    "power_std": "Within-segment variability without duplicating power mean.",
    "power_slope": "Direction and rate of energetic transition.",
    "flow_mean": "Independent process-flow context and low missingness.",
    "pressure_mean": "Independent pneumatic-load context and low missingness.",
}


def supervised_qml_sample(frame: pd.DataFrame, counts: dict[str, int]) -> pd.DataFrame:
    selected: list[int] = []
    for split, per_class in counts.items():
        for target in sorted(LABEL_MAP.values()):
            population = frame[(frame["split"] == split) & (frame["target"] == target)]
            stations = sorted(population["station_id"].unique())
            base, remainder = divmod(int(per_class), len(stations))
            for position, station in enumerate(stations):
                quota = base + int(position < remainder)
                selected.extend(_temporal_quantiles(population[population["station_id"] == station], quota).tolist())
    result = frame.loc[sorted(set(selected))].copy()
    result["qml_selected"] = True
    return result.sort_values(["split", "segment_start_epoch_ms", "segment_id"]).reset_index(drop=True)


def label_free_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    population = frame.copy()
    stations = sorted(population["station_id"].unique())
    base, remainder = divmod(int(count), len(stations))
    selected: list[int] = []
    for position, station in enumerate(stations):
        quota = base + int(position < remainder)
        selected.extend(_temporal_quantiles(population[population["station_id"] == station], quota).tolist())
    return frame.loc[sorted(set(selected))].sort_values(["segment_start_epoch_ms", "segment_id"]).reset_index(drop=True)


def feature_diagnostics(frame: pd.DataFrame, features: list[str], selected: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    correlations = frame[features].corr()
    for feature in features:
        experiment_medians = frame.groupby("experiment_id")[feature].median()
        station_medians = frame.groupby("station_id")[feature].median()
        mean_abs = max(abs(float(frame[feature].mean())), 1e-12)
        rows.append({
            "feature": feature,
            "formula": FORMULAS.get(feature, "As defined in the frozen approved feature pipeline."),
            "selected": feature in selected,
            "reason": REASONS.get(feature, "Excluded to limit qubit count and reduce redundancy with selected level, variability, slope, or sensor summaries."),
            "maximum_absolute_correlation_with_selected": float(correlations.loc[feature, [value for value in selected if value != feature]].abs().max()) if any(value != feature for value in selected) else 0.0,
            "missing_fraction": float(frame[feature].isna().mean()),
            "variance": float(frame[feature].var()),
            "min": float(frame[feature].min()), "p10": float(frame[feature].quantile(.10)),
            "median": float(frame[feature].median()), "p90": float(frame[feature].quantile(.90)),
            "max": float(frame[feature].max()),
            "experiment_median_range": float(experiment_medians.max() - experiment_medians.min()),
            "station_median_range": float(station_medians.max() - station_medians.min()),
            "experiment_median_relative_range": float((experiment_medians.max() - experiment_medians.min()) / mean_abs),
            "station_median_relative_range": float((station_medians.max() - station_medians.min()) / mean_abs),
            "scaling": "MinMaxScaler train-only to [0,pi], clipped outside train range",
        })
    return pd.DataFrame(rows), correlations.loc[selected, selected].rename_axis("feature").reset_index()


def evaluate_predictions(
    model: str, condition: str, split: str, truth: np.ndarray, prediction: np.ndarray,
    training_seconds: float, inference_seconds: float, metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = [0, 1]
    row: dict[str, Any] = {
        "model": model, "condition": condition, "split": split, "sample_count": len(truth),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "precision_macro": precision_score(truth, prediction, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(truth, prediction, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(truth, prediction, labels=labels, average="macro", zero_division=0),
        "f1_idle_ready": f1_score(truth, prediction, labels=labels, average=None, zero_division=0)[0],
        "f1_loading_unloading_transfer": f1_score(truth, prediction, labels=labels, average=None, zero_division=0)[1],
        "training_seconds": training_seconds, "inference_seconds": inference_seconds,
    }
    if metadata:
        row.update(metadata)
    matrix = confusion_matrix(truth, prediction, labels=labels)
    confusion = pd.DataFrame([
        {"model": model, "condition": condition, "split": split, "true_label": true, "predicted_label": predicted, "count": int(matrix[true, predicted])}
        for true in labels for predicted in labels
    ])
    return row, confusion


def circuit_diagnostics(feature_map: Any, ansatz: Any, fake_backend: Any, seed: int) -> pd.DataFrame:
    combined = feature_map.compose(ansatz)
    pass_manager = generate_preset_pass_manager(backend=fake_backend, optimization_level=1, seed_transpiler=seed)
    rows = []
    for name, circuit in (("feature_map", feature_map), ("ansatz", ansatz), ("combined", combined)):
        decomposed = circuit.decompose(reps=10)
        noisy = pass_manager.run(circuit)
        rows.append({
            "circuit": name, "qubits": circuit.num_qubits, "parameters": circuit.num_parameters,
            "ideal_depth": decomposed.depth(), "ideal_gate_count": int(sum(decomposed.count_ops().values())),
            "ideal_gates": json.dumps(dict(decomposed.count_ops()), default=int, sort_keys=True),
            "fake_backend_depth": noisy.depth(), "fake_backend_gate_count": int(sum(noisy.count_ops().values())),
            "fake_backend_gates": json.dumps(dict(noisy.count_ops()), default=int, sort_keys=True),
        })
    return pd.DataFrame(rows)


def make_sampler(condition: str, shots: int | None, seed: int, fake_backend: Any | None = None) -> tuple[Any, Any | None]:
    if condition == "ideal":
        return QMLSampler(shots=None, seed=seed), None
    if condition == "finite_shots":
        return QMLSampler(shots=int(shots), seed=seed), None
    if fake_backend is None:
        raise ValueError("Noisy sampling requires a local fake backend")
    noise_model = NoiseModel.from_backend(fake_backend)
    pass_manager = generate_preset_pass_manager(backend=fake_backend, optimization_level=1, seed_transpiler=seed)
    sampler = AerSamplerV2(
        default_shots=int(shots), seed=seed,
        options={"backend_options": {"noise_model": noise_model, "basis_gates": noise_model.basis_gates, "coupling_map": fake_backend.coupling_map}},
    )
    return sampler, pass_manager


def quantum_kernel(
    feature_map: Any, condition: str, shots: int | None, seed: int, fake_backend: Any | None = None
) -> FidelityQuantumKernel:
    sampler, pass_manager = make_sampler(condition, shots, seed, fake_backend)
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pass_manager)
    return FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity, enforce_psd=True)


def evaluate_kernel_condition(
    feature_map: Any, arrays: dict[str, tuple[np.ndarray, np.ndarray]], condition: str,
    shots: int | None, repetition: int, seed: int, c_candidates: list[float],
    matrix_dir: Path, fake_backend: Any | None = None, reference: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    kernel = quantum_kernel(feature_map, condition, shots, seed, fake_backend)
    x_train, y_train = arrays["train"]
    start = perf_counter(); train_matrix = kernel.evaluate(x_train); kernel_train_seconds = perf_counter() - start
    matrices = {"train": train_matrix}
    kernel_times = {"train": kernel_train_seconds}
    for split in ("validation", "test"):
        start = perf_counter(); matrices[split] = kernel.evaluate(arrays[split][0], x_train); kernel_times[split] = perf_counter() - start
    condition_id = f"{condition}_shots-{shots if shots is not None else 'exact'}_rep-{repetition}"
    np.savez_compressed(matrix_dir / f"qsvc_{condition_id}.npz", **matrices)
    best_c, best_score = None, -np.inf
    for candidate in c_candidates:
        model = QSVC(quantum_kernel="precomputed", C=float(candidate), class_weight="balanced", random_state=seed)
        model.fit(train_matrix, y_train)
        score = f1_score(arrays["validation"][1], model.predict(matrices["validation"]), average="macro")
        if score > best_score:
            best_c, best_score = float(candidate), float(score)
    model = QSVC(quantum_kernel="precomputed", C=best_c, class_weight="balanced", random_state=seed)
    start = perf_counter(); model.fit(train_matrix, y_train); fit_seconds = perf_counter() - start
    rows, confusions, predictions = [], [], []
    reference_diff = None
    if reference is not None:
        reference_diff = float(np.linalg.norm(train_matrix - reference["train"], ord="fro") / np.linalg.norm(reference["train"], ord="fro"))
    for split in ("validation", "test"):
        start = perf_counter(); predicted = model.predict(matrices[split]); model_inference = perf_counter() - start
        metadata = {
            "shots": shots, "repetition": repetition, "seed": seed, "C": best_c,
            "kernel_train_seconds": kernel_train_seconds,
            "kernel_inference_seconds": kernel_times[split],
            "kernel_matrix_rows": train_matrix.shape[0], "kernel_matrix_columns": train_matrix.shape[1],
            "relative_train_kernel_frobenius_error_vs_ideal": reference_diff,
        }
        row, confusion = evaluate_predictions("QSVC", condition, split, arrays[split][1], predicted, fit_seconds + kernel_train_seconds, model_inference + kernel_times[split], metadata)
        rows.append(row); confusions.append(confusion)
        predictions.extend({"model": "QSVC", "condition": condition_id, "split": split, "row": i, "truth": int(t), "prediction": int(p)} for i, (t, p) in enumerate(zip(arrays[split][1], predicted, strict=True)))
    details = {"condition_id": condition_id, "selected_C": best_c, "validation_selection_f1": best_score, "kernel_train_seconds": kernel_train_seconds, "kernel_validation_seconds": kernel_times["validation"], "kernel_test_seconds": kernel_times["test"]}
    return rows, confusions, predictions, matrices, details


