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

from .energy_state_discovery import DISCOVERY_FEATURES


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_inputs(root: Path, config: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    pairs = [
        ("discovery_sqlite", "discovery_sqlite_sha256"),
        ("common_sqlite", "common_sqlite_sha256"),
        ("segmentation_config", "segmentation_config_sha256"),
        ("inference_logic", "inference_logic_sha256"),
    ]
    for path_key, hash_key in pairs:
        path = root / config["frozen_inputs"][path_key]
        actual = sha256_file(path)
        expected = config["frozen_inputs"][hash_key]
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {path_key}: {actual} != {expected}")
        resolved[path_key] = actual
    return resolved


def load_segments(path: Path, features: list[str]) -> pd.DataFrame:
    columns = [
        "segment_id", "experiment_id", "station_id", "session_id",
        "segment_start_epoch_ms", "segment_end_epoch_ms", "segment_start_utc",
        "segment_end_utc", "inferred_state", *features,
    ]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(
            f"SELECT {','.join(columns)} FROM segments_with_inferred_state ORDER BY segment_start_epoch_ms,segment_id",
            connection,
        )
    return frame


def assign_temporal_split(frame: pd.DataFrame, split_config: dict[str, Any]) -> pd.DataFrame:
    sets = {
        "train": set(split_config["train_experiments"]),
        "validation": set(split_config["validation_experiments"]),
        "test": set(split_config["test_experiments"]),
    }
    result = frame.copy()
    result["split"] = pd.NA
    for name, experiments in sets.items():
        result.loc[result["experiment_id"].isin(experiments), "split"] = name
    if result["split"].isna().any():
        raise ValueError("Every segment must belong to exactly one chronological experiment split")
    result["target"] = result["inferred_state"].map(LABEL_MAP)
    result["supervised_eligible"] = result["target"].notna()
    return result


def _temporal_quantiles(group: pd.DataFrame, count: int) -> pd.Index:
    ordered = group.sort_values(["segment_start_epoch_ms", "segment_id"])
    if count >= len(ordered):
        return ordered.index
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered.iloc[np.unique(positions)].index


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


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in frame.groupby("split", sort=False):
        counts = group["inferred_state"].value_counts()
        rows.append({
            "split": split, "segment_count": len(group),
            "eligible_count": int(group["supervised_eligible"].sum()),
            "idle_ready_count": int(counts.get("idle_ready", 0)),
            "loading_unloading_transfer_count": int(counts.get("loading_unloading_transfer", 0)),
            "unknown_count": int(counts.get("unknown", 0)),
            "experiments": json.dumps(sorted(group["experiment_id"].unique().tolist())),
            "start_utc": group["segment_start_utc"].min(), "end_utc": group["segment_end_utc"].max(),
            "station_count": int(group["station_id"].nunique()), "session_count": int(group["session_id"].nunique()),
        })
    return pd.DataFrame(rows)


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


def classical_models(config: dict[str, Any], seed: int, y_train: np.ndarray) -> dict[str, Any]:
    cfg = config["classical"]
    positive = max(1, int((y_train == 1).sum()))
    negative = max(1, int((y_train == 0).sum()))
    return {
        "LogisticRegression": LogisticRegression(**cfg["logistic_regression"], random_state=seed),
        "SVM_RBF": SVC(**cfg["svm"], random_state=seed),
        "RandomForest": RandomForestClassifier(**cfg["random_forest"], random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(**cfg["xgboost"], random_state=seed, n_jobs=-1, scale_pos_weight=negative / positive, eval_metric="logloss"),
    }


def run_classical(
    config: dict[str, Any], seed: int, arrays: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, confusions, predictions = [], [], []
    x_train, y_train = arrays["train"]
    for name, model in classical_models(config, seed, y_train).items():
        start = perf_counter(); model.fit(x_train, y_train); training = perf_counter() - start
        for split in ("validation", "test"):
            x, y = arrays[split]
            start = perf_counter(); predicted = model.predict(x); inference = perf_counter() - start
            row, confusion = evaluate_predictions(name, "fair_6_feature", split, y, predicted, training, inference)
            rows.append(row); confusions.append(confusion)
            predictions.extend({"model": name, "condition": "fair_6_feature", "split": split, "row": i, "truth": int(t), "prediction": int(p)} for i, (t, p) in enumerate(zip(y, predicted, strict=True)))
    return pd.DataFrame(rows), pd.concat(confusions, ignore_index=True), pd.DataFrame(predictions)


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


def evaluate_vqc_condition(
    feature_map: Any, ansatz: Any, arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    condition: str, shots: int | None, repetition: int, seed: int, maxiter: int,
    fake_backend: Any | None = None,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    sampler, pass_manager = make_sampler(condition, shots, seed, fake_backend)
    history: list[float] = []
    def callback(_: np.ndarray, objective: float) -> None:
        history.append(float(objective))
    initial_rng = np.random.default_rng(seed)
    initial_point = initial_rng.uniform(-0.1, 0.1, ansatz.num_parameters)
    classifier = VQC(
        feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=maxiter),
        loss="cross_entropy", sampler=sampler, pass_manager=pass_manager,
        initial_point=initial_point, callback=callback,
    )
    start = perf_counter(); classifier.fit(*arrays["train"]); training = perf_counter() - start
    condition_id = f"{condition}_shots-{shots if shots is not None else 'exact'}_rep-{repetition}"
    rows, confusions, predictions = [], [], []
    for split in ("validation", "test"):
        start = perf_counter(); predicted = classifier.predict(arrays[split][0]).astype(int); inference = perf_counter() - start
        metadata = {"shots": shots, "repetition": repetition, "seed": seed, "optimizer": "COBYLA", "maxiter": maxiter, "objective_evaluations": len(history), "final_objective": history[-1] if history else None}
        row, confusion = evaluate_predictions("VQC", condition, split, arrays[split][1], predicted, training, inference, metadata)
        rows.append(row); confusions.append(confusion)
        predictions.extend({"model": "VQC", "condition": condition_id, "split": split, "row": i, "truth": int(t), "prediction": int(p)} for i, (t, p) in enumerate(zip(arrays[split][1], predicted, strict=True)))
    convergence = pd.DataFrame({"model": "VQC", "condition": condition, "shots": shots, "repetition": repetition, "seed": seed, "objective_evaluation": np.arange(1, len(history) + 1), "objective": history})
    details = {"condition_id": condition_id, "training_seconds": training, "objective_evaluations": len(history), "final_objective": history[-1] if history else None}
    return rows, confusions, predictions, convergence, details


def quantum_kernel_clustering(
    feature_map: Any, frame: pd.DataFrame, encoded: np.ndarray, seed: int, matrix_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    kernel = quantum_kernel(feature_map, "ideal", None, seed)
    start = perf_counter(); matrix = kernel.evaluate(encoded); kernel_seconds = perf_counter() - start
    np.savez_compressed(matrix_dir / "quantum_kernel_clustering_ideal.npz", kernel=matrix)
    start = perf_counter(); quantum_labels = SpectralClustering(n_clusters=2, affinity="precomputed", assign_labels="kmeans", random_state=seed).fit_predict(matrix); clustering_seconds = perf_counter() - start
    classical_labels = KMeans(n_clusters=2, n_init=50, random_state=seed).fit_predict(encoded)
    labeled = frame["target"].notna().to_numpy()
    truth = frame.loc[labeled, "target"].astype(int).to_numpy()
    output = frame[["segment_id", "experiment_id", "station_id", "session_id", "segment_start_utc", "inferred_state"]].copy()
    output["quantum_kernel_cluster"] = quantum_labels
    output["classical_kmeans_cluster"] = classical_labels
    metrics = {
        "sample_count": len(frame), "selection_used_inferred_state": False,
        "kernel_matrix_shape": json.dumps(list(matrix.shape)), "kernel_seconds": kernel_seconds,
        "spectral_clustering_seconds": clustering_seconds,
        "labeled_rows_for_posthoc_evaluation": int(labeled.sum()),
        "quantum_vs_pseudo_label_nmi": normalized_mutual_info_score(truth, quantum_labels[labeled]),
        "quantum_vs_pseudo_label_ari": adjusted_rand_score(truth, quantum_labels[labeled]),
        "classical_vs_pseudo_label_nmi": normalized_mutual_info_score(truth, classical_labels[labeled]),
        "classical_vs_pseudo_label_ari": adjusted_rand_score(truth, classical_labels[labeled]),
        "quantum_vs_classical_ari": adjusted_rand_score(quantum_labels, classical_labels),
    }
    return output, metrics


def plot_outputs(metrics: pd.DataFrame, confusions: pd.DataFrame, convergence: pd.DataFrame, ideal_kernel: np.ndarray, plots: Path) -> None:
    plots.mkdir(parents=True, exist_ok=True)
    test = metrics[metrics["split"] == "test"].copy()
    test["label"] = test["model"] + "\n" + test["condition"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(len(test)), test["f1_macro"])
    ax.set_xticks(np.arange(len(test)), test["label"], rotation=60, ha="right")
    ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1); ax.set_title("Fair test comparison")
    fig.tight_layout(); fig.savefig(plots / "test_macro_f1.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5)); image = ax.imshow(ideal_kernel, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Ideal QSVC training kernel"); fig.colorbar(image, ax=ax); fig.tight_layout()
    fig.savefig(plots / "ideal_qsvc_kernel.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, group in convergence.groupby(["condition", "shots", "repetition"], dropna=False):
        ax.plot(group["objective_evaluation"], group["objective"], label="/".join(map(str, key)), alpha=.8)
    ax.set_xlabel("Objective evaluation"); ax.set_ylabel("Cross-entropy objective"); ax.set_title("VQC convergence")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plots / "vqc_convergence.png", dpi=160); plt.close(fig)
    final = confusions[confusions["split"] == "test"]
    final.to_csv(plots / "confusion_matrices_test.csv", index=False)


def run(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    root = config_path.parents[1]
    frozen_hashes = verify_frozen_inputs(root, config)
    seed = int(config["seed"]); algorithm_globals.random_seed = seed; np.random.seed(seed)
    features = list(config["features"]["selected"])
    source = root / config["frozen_inputs"]["discovery_sqlite"]
    frame = assign_temporal_split(load_segments(source, DISCOVERY_FEATURES), config["split"])
    eligible = frame[frame["supervised_eligible"]].copy()
    feature_report, correlations = feature_diagnostics(eligible, DISCOVERY_FEATURES, features)
    splits = split_summary(frame)
    qml = supervised_qml_sample(eligible, config["split"]["qml_per_class"])
    scaler = MinMaxScaler(feature_range=tuple(config["features"]["quantum_scaling"]["feature_range"]), clip=True)
    train_mask = qml["split"] == "train"
    scaler.fit(qml.loc[train_mask, features])
    encoded = scaler.transform(qml[features])
    arrays = {
        split: (encoded[qml["split"].eq(split).to_numpy()], qml.loc[qml["split"] == split, "target"].astype(int).to_numpy())
        for split in ("train", "validation", "test")
    }
    scaling = pd.DataFrame({"feature": features, "train_min": scaler.data_min_, "train_max": scaler.data_max_, "scale": scaler.scale_, "offset": scaler.min_, "encoded_min": 0.0, "encoded_max": np.pi, "clip": True})

    results_dir = root / config["outputs"]["results"]
    matrix_dir = root / config["outputs"]["matrices"]
    plots_dir = root / config["outputs"]["plots"]
    results_dir.mkdir(parents=True, exist_ok=True); matrix_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)
    classical_metrics, classical_confusions, classical_predictions = run_classical(config, seed, arrays)

    qcfg = config["quantum"]
    feature_map = zz_feature_map(len(features), reps=int(qcfg["feature_map"]["reps"]), entanglement=qcfg["feature_map"]["entanglement"])
    ansatz = real_amplitudes(len(features), reps=int(qcfg["vqc"]["ansatz_reps"]), entanglement=qcfg["vqc"]["entanglement"])
    fake_backend = FakeJakartaV2()
    circuits = circuit_diagnostics(feature_map, ansatz, fake_backend, seed)

    metric_rows: list[dict[str, Any]] = classical_metrics.to_dict("records")
    confusion_frames = [classical_confusions]
    prediction_rows = classical_predictions.to_dict("records")
    execution_rows: list[dict[str, Any]] = []
    convergence_frames: list[pd.DataFrame] = []

    rows, confs, preds, ideal_matrices, details = evaluate_kernel_condition(
        feature_map, arrays, "ideal", None, 1, seed, list(qcfg["qsvc"]["C_candidates"]), matrix_dir
    )
    metric_rows.extend(rows); confusion_frames.extend(confs); prediction_rows.extend(preds); execution_rows.append({"model": "QSVC", **details})
    for shots in qcfg["finite_shots"]["shots"]:
        for repetition in range(1, int(qcfg["finite_shots"]["repetitions"]) + 1):
            run_seed = seed + int(shots) + repetition
            rows, confs, preds, _, details = evaluate_kernel_condition(
                feature_map, arrays, "finite_shots", int(shots), repetition, run_seed,
                list(qcfg["qsvc"]["C_candidates"]), matrix_dir, reference=ideal_matrices,
            )
            metric_rows.extend(rows); confusion_frames.extend(confs); prediction_rows.extend(preds); execution_rows.append({"model": "QSVC", **details})
    for repetition in range(1, int(qcfg["noisy"]["repetitions"]) + 1):
        run_seed = seed + 10000 + repetition
        rows, confs, preds, _, details = evaluate_kernel_condition(
            feature_map, arrays, "noisy", int(qcfg["noisy"]["shots"]), repetition, run_seed,
            list(qcfg["qsvc"]["C_candidates"]), matrix_dir, fake_backend=fake_backend, reference=ideal_matrices,
        )
        metric_rows.extend(rows); confusion_frames.extend(confs); prediction_rows.extend(preds); execution_rows.append({"model": "QSVC", **details})

    rows, confs, preds, convergence, details = evaluate_vqc_condition(
        feature_map, ansatz, arrays, "ideal", None, 1, seed,
        int(qcfg["vqc"]["ideal_maxiter"]),
    )
    metric_rows.extend(rows); confusion_frames.extend(confs); prediction_rows.extend(preds); convergence_frames.append(convergence); execution_rows.append({"model": "VQC", **details})
    for shots in qcfg["finite_shots"]["shots"]:
        for repetition in range(1, int(qcfg["finite_shots"]["repetitions"]) + 1):
            run_seed = seed + int(shots) + repetition
            rows, confs, preds, convergence, details = evaluate_vqc_condition(
                feature_map, ansatz, arrays, "finite_shots", int(shots), repetition, run_seed,
                int(qcfg["vqc"]["sampled_maxiter"]),
            )
            metric_rows.extend(rows); confusion_frames.extend(confs); prediction_rows.extend(preds); convergence_frames.append(convergence); execution_rows.append({"model": "VQC", **details})
    for repetition in range(1, int(qcfg["noisy"]["repetitions"]) + 1):
        run_seed = seed + 10000 + repetition
        rows, confs, preds, convergence, details = evaluate_vqc_condition(
            feature_map, ansatz, arrays, "noisy", int(qcfg["noisy"]["shots"]), repetition,
            run_seed, int(qcfg["vqc"]["sampled_maxiter"]), fake_backend=fake_backend,
        )
        metric_rows.extend(rows); confusion_frames.extend(confs); prediction_rows.extend(preds); convergence_frames.append(convergence); execution_rows.append({"model": "VQC", **details})

    unsupervised = label_free_sample(frame, int(qcfg["kernel_clustering"]["samples"]))
    unsupervised_encoded = scaler.transform(unsupervised[features])
    clustering, clustering_metrics = quantum_kernel_clustering(feature_map, unsupervised, unsupervised_encoded, seed, matrix_dir)

    metrics = pd.DataFrame(metric_rows)
    confusions = pd.concat(confusion_frames, ignore_index=True)
    predictions = pd.DataFrame(prediction_rows)
    convergence = pd.concat(convergence_frames, ignore_index=True)
    execution = pd.DataFrame(execution_rows)
    plot_outputs(metrics, confusions, convergence, ideal_matrices["train"], plots_dir)

    split_membership = frame[["segment_id", "experiment_id", "station_id", "session_id", "segment_start_utc", "segment_end_utc", "inferred_state", "target", "split", "supervised_eligible"]].copy()
    selected_ids = set(qml["segment_id"]); split_membership["qml_selected"] = split_membership["segment_id"].isin(selected_ids)
    tables = {
        "split_membership": split_membership, "qml_sample": qml,
        "selected_feature_diagnostics": feature_report, "selected_feature_correlations": correlations,
        "scaling_parameters": scaling, "metrics": metrics, "confusion_matrices": confusions,
        "predictions": predictions, "circuit_diagnostics": circuits, "execution_details": execution,
        "vqc_convergence": convergence, "quantum_kernel_clustering": clustering,
        "quantum_kernel_clustering_metrics": pd.DataFrame([clustering_metrics]),
    }
    output_db = root / config["outputs"]["sqlite"]
    if output_db.exists(): output_db.unlink()
    with sqlite3.connect(output_db) as connection:
        for name, table in tables.items(): table.to_sql(name, connection, index=False, if_exists="replace")
    for name, table in tables.items():
        table.to_csv(results_dir / f"{name}.csv.gz", index=False, compression="gzip")
    splits.to_csv(results_dir / "split_summary.csv", index=False)
    (results_dir / "quantum_kernel_clustering_summary.json").write_text(json.dumps(clustering_metrics, indent=2), encoding="utf-8")

    frozen_hashes_after = verify_frozen_inputs(root, config)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "version": config["version"], "seed": seed,
        "python": platform.python_version(), "qiskit": qiskit.__version__,
        "qiskit_machine_learning": qiskit_machine_learning.__version__, "qiskit_aer": qiskit_aer.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "frozen_inputs_unchanged": frozen_hashes == frozen_hashes_after,
        "selected_features": features, "target_is_pseudo_label": True,
        "split_strategy": config["split"]["strategy"],
        "qml_samples": {name: len(values[1]) for name, values in arrays.items()},
        "feature_map": {**qcfg["feature_map"], "qubits": len(features)},
        "ansatz": {"name": qcfg["vqc"]["ansatz"], "reps": qcfg["vqc"]["ansatz_reps"], "parameters": ansatz.num_parameters},
        "ideal_local_completed": True, "finite_shots_completed": True, "noisy_local_completed": True,
        "fake_backend": qcfg["noisy"]["fake_backend"], "ibm_runtime_service_used": False,
        "real_backend_used": False, "qpu_used": False,
        "quantum_kernel_clustering": clustering_metrics,
        "output_sqlite": str(output_db.relative_to(root)), "output_sqlite_sha256": sha256_file(output_db),
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
