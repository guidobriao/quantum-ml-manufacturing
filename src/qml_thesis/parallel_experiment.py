"""Parallel experiment: recover machine states from energy-only signals.

2x2 design, one axis of variation per comparison:
  - unsupervised row: classical k-means  vs  quantum kernel k-means
  - supervised row:    SVC/RF/XGBoost    vs  QSVC/VQC
All models use the same 1-second windows, the same six energy features,
the same samples, and are evaluated post-hoc against the PLC-derived
ground truth (never used as input). Station 50 is included but reported
separately (power flat at ~11.8 W: expected unrecoverable, documented).
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .common_io import assign_temporal_split
from .qml_common import label_free_sample, supervised_qml_sample

CLASS_MERGE = {
    "idle_ready": "idle_ready",
    "processing": "processing",
    "waiting_carrier": "waiting_carrier",
    "waiting_carrier_ready": "waiting_carrier",
    "entering": "transfer_transient",
    "exiting": "transfer_transient",
    "transfer": "transfer_transient",
}
CLASSES = ["idle_ready", "waiting_carrier", "transfer_transient", "processing"]


def load_windows(ground_truth_sqlite: Path, feature_columns: list[str]) -> pd.DataFrame:
    with sqlite3.connect(f"file:{ground_truth_sqlite}?mode=ro", uri=True) as connection:
        features = pd.read_sql_query(
            f"SELECT * FROM gt_features_1s", connection)
        labels = pd.read_sql_query(
            "SELECT segment_id, inferred_state FROM gt_windows WHERE window_seconds = 1",
            connection)
    frame = features.merge(labels, on="segment_id", validate="one_to_one")
    frame = frame[frame["inferred_state"].isin(CLASS_MERGE)]
    frame["state_4"] = frame["inferred_state"].map(CLASS_MERGE)
    # frame["station_id"] = frame["segment_id"].str.split(":").str[2].str[1:].astype(int)
    # frame["experiment_id"] = frame["segment_id"].str.split(":").str[1]
    # experiment_id / station_id already exist as columns in gt_features_1s
    return frame.reset_index(drop=True)


# ------------------------------------------------------------- unsupervised

def kernel_kmeans(kernel_matrix: np.ndarray, k: int, seed: int,
                  n_init: int = 10, max_iter: int = 100) -> np.ndarray:
    """Kernel k-means on a precomputed kernel matrix (standard KKMeans)."""
    rng = np.random.default_rng(seed)
    n = len(kernel_matrix)
    diag = np.diag(kernel_matrix)
    best_labels, best_objective = None, np.inf
    for _ in range(n_init):
        labels = rng.integers(0, k, size=n)
        if len(np.unique(labels)) < k:  # re-draw rare empty init
            labels[:k] = np.arange(k)
        for _ in range(max_iter):
            one_hot = np.zeros((n, k))
            one_hot[np.arange(n), labels] = 1.0
            counts = one_hot.sum(axis=0)                     # n_j
            counts[counts == 0] = 1.0
            # ||phi(x)-c_j||^2 = K_ii - 2/n_j sum_l K_il + 1/n_j^2 sum_lm K_lm
            cross = kernel_matrix @ one_hot                   # sum_l in j K_il
            inner = one_hot.T @ (kernel_matrix @ one_hot)     # sum_lm K_lm
            distances = (diag[:, None] - 2.0 * cross / counts
                         + inner.diagonal()[None, :] / counts ** 2)
            new_labels = distances.argmin(axis=1)
            if len(np.unique(new_labels)) < k:
                break
            if (new_labels == labels).all():
                labels = new_labels
                break
            labels = new_labels
        objective = float(np.sum(diag - 2.0 * (kernel_matrix @ np.eye(k)[labels]
                     / np.maximum(np.bincount(labels, minlength=k), 1)[labels])))
        if objective < best_objective:
            best_objective, best_labels = objective, labels.copy()
    return best_labels


def hungarian_accuracy(truth: np.ndarray, clusters: np.ndarray) -> tuple[float, dict]:
    """Optimal cluster->state mapping via Hungarian algorithm on the confusion matrix."""
    states = np.unique(truth)
    matrix = np.zeros((len(states), len(np.unique(clusters))))
    for i, state in enumerate(states):
        for j, cluster in enumerate(np.unique(clusters)):
            matrix[i, j] = np.sum((truth == state) & (clusters == cluster))
    rows, columns = linear_sum_assignment(-matrix)
    mapping = {int(np.unique(clusters)[j]): states[i] for i, j in zip(rows, columns)}
    mapped = np.array([mapping[int(c)] for c in clusters])
    return float(accuracy_score(truth, mapped)), mapping


def unsupervised_row(config: dict, seed: int) -> list[dict]:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.utils import algorithm_globals
    from .qml_common import make_sampler, quantum_kernel

    sqlite_path = Path(config["source"]["ground_truth_sqlite"])
    frame = load_windows(sqlite_path, config["features"]["energy"])
    sample = label_free_sample(frame, config["sampling"]["unsupervised_n"])

    matrix = StandardScaler().fit_transform(
        sample[config["features"]["energy"]].to_numpy())
    truth = sample["state_4"].to_numpy()

    rows = []
    k_list = config["clustering"]["k_candidates"]
    for k in k_list:
        from sklearn.cluster import KMeans
        labels_km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(matrix)
        accuracy, _ = hungarian_accuracy(truth, labels_km)
        rows.append({"row": "unsupervised", "model": "kmeans", "k": k,
                     "ari": adjusted_rand_score(truth, labels_km),
                     "nmi": normalized_mutual_info_score(truth, labels_km),
                     "accuracy_mapped": accuracy, "n": int(len(sample))})

    algorithm_globals.random_seed = seed
    features = config["features"]["energy"]
    feature_map = ZZFeatureMap(feature_dimension=len(features), reps=2, entanglement="linear")
    encoded = MinMaxScaler(feature_range=(0.0, np.pi)).fit_transform(
        sample[features].to_numpy())
    kernel = quantum_kernel(feature_map, "ideal", None, seed)
    kernel = kernel[0] if isinstance(kernel, tuple) else kernel
    gram = kernel.evaluate(encoded)
    for k in k_list:
        labels_q = kernel_kmeans(gram, k, seed)
        accuracy, _ = hungarian_accuracy(truth, labels_q)
        rows.append({"row": "unsupervised", "model": "quantum_kernel_kmeans", "k": k,
                     "ari": adjusted_rand_score(truth, labels_q),
                     "nmi": normalized_mutual_info_score(truth, labels_q),
                     "accuracy_mapped": accuracy, "n": int(len(sample))})
    return rows


# --------------------------------------------------------------- supervised

def assign_split_by_experiment(frame: pd.DataFrame, split_config: dict) -> pd.DataFrame:
    """Map experiment_id to train/validation/test from the config (phase-2)."""
    mapping: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        for experiment in split_config[f"{split}_experiments"]:
            mapping[experiment] = split
    frame = frame.copy()
    frame["split"] = frame["experiment_id"].map(mapping).fillna("unknown")
    return frame


def supervised_row(config: dict, seed: int) -> list[dict]:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.optimizers import COBYLA
    from qiskit_machine_learning.algorithms import QSVC, VQC
    from qiskit_machine_learning.utils import algorithm_globals
    from .qml_common import quantum_kernel

    frame = load_windows(Path(config["source"]["ground_truth_sqlite"]),
                         config["features"]["energy"])
    frame = assign_temporal_split(frame, config["split"])
    from .qml_common import LABEL_MAP
    frame["target"] = frame["state_4"].map(LABEL_MAP)
    counts = config["sampling"]["supervised_per_class"]
    sample = supervised_qml_sample(frame, counts)
    features = config["features"]["energy"]

    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    train = sample[sample["split"] == "train"]
    scaler.fit(train[features].to_numpy())
    encoded = {split: scaler.transform(part[features].to_numpy())
               for split, part in sample.groupby("split")}
    truth = {split: part["state_4"].to_numpy() for split, part in sample.groupby("split")}

    classical = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "svc_rbf": SVC(class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                                class_weight="balanced", random_state=seed),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 tree_method="hist", device="cpu", random_state=seed),
    }
    rows = []
    for name, model in classical.items():
        model.fit(scaler.transform(train[features].to_numpy()), truth["train"])
        for split in ["validation", "test"]:
            prediction = model.predict(encoded[split])
            rows.append({"row": "supervised", "model": name, "split": split,
                         "n": int(len(truth[split])),
                         "accuracy": accuracy_score(truth[split], prediction),
                         "f1_macro": f1_score(truth[split], prediction, average="macro")})

    algorithm_globals.random_seed = seed
    feature_map = ZZFeatureMap(feature_dimension=len(features), reps=2, entanglement="linear")
    kernel = quantum_kernel(feature_map, "ideal", None, seed)
    kernel = kernel[0] if isinstance(kernel, tuple) else kernel
    qsvc = QSVC(kernel=kernel)
    qsvc.fit(encoded["train"], truth["train"])
    for split in ["validation", "test"]:
        prediction = qsvc.predict(encoded[split])
        rows.append({"row": "supervised", "model": "QSVC", "split": split,
                     "n": int(len(truth[split])),
                     "accuracy": accuracy_score(truth[split], prediction),
                     "f1_macro": f1_score(truth[split], prediction, average="macro")})

    ansatz = RealAmplitudes(len(features), reps=1, entanglement="linear")
    vqc = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=COBYLA(maxiter=30),
              loss="cross_entropy")
    vqc.fit(encoded["train"], truth["train"])
    for split in ["validation", "test"]:
        prediction = vqc.predict(encoded[split])
        rows.append({"row": "supervised", "model": "VQC", "split": split,
                     "n": int(len(truth[split])),
                     "accuracy": accuracy_score(truth[split], prediction),
                     "f1_macro": f1_score(truth[split], prediction, average="macro")})
    return rows


def run(config_path: Path) -> dict[str, object]:
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seed = config["seed"]
    rows = unsupervised_row(config, seed) + supervised_row(config, seed)
    metrics = pd.DataFrame(rows)
    output = Path(config["outputs"]["sqlite"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(output) as connection:
        metrics.to_sql("parallel_metrics", connection, index=False)
    return {"metric_rows": int(len(metrics))}


if __name__ == "__main__":
    print(run(Path("configs/parallel_experiment.yaml")))