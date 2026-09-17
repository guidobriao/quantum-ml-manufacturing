from __future__ import annotations

import pandas as pd

from qml_thesis.qml_experiments import (
    LABEL_MAP, assign_temporal_split, label_free_sample, supervised_qml_sample,
)


def _segments() -> pd.DataFrame:
    rows = []
    for experiment, start in (("E1", 0), ("E2", 100000), ("E3", 200000)):
        for station in (10, 20):
            for state in LABEL_MAP:
                for index in range(10):
                    rows.append({
                        "segment_id": f"{experiment}-{station}-{state}-{index}",
                        "experiment_id": experiment, "station_id": station, "session_id": f"{experiment}-S{station}",
                        "segment_start_epoch_ms": start + index * 5000, "inferred_state": state,
                    })
    return pd.DataFrame(rows)


def test_chronological_experiment_split_is_disjoint() -> None:
    frame = assign_temporal_split(_segments(), {
        "train_experiments": ["E1"], "validation_experiments": ["E2"], "test_experiments": ["E3"],
    })
    assert set(frame.loc[frame.split == "train", "experiment_id"]) == {"E1"}
    assert set(frame.loc[frame.split == "validation", "experiment_id"]) == {"E2"}
    assert set(frame.loc[frame.split == "test", "experiment_id"]) == {"E3"}


def test_supervised_sample_is_class_balanced_and_station_distributed() -> None:
    frame = assign_temporal_split(_segments(), {
        "train_experiments": ["E1"], "validation_experiments": ["E2"], "test_experiments": ["E3"],
    })
    sample = supervised_qml_sample(frame, {"train": 6, "validation": 4, "test": 2})
    counts = sample.groupby(["split", "target"]).size()
    assert counts[("train", 0)] == counts[("train", 1)] == 6
    assert sample.groupby(["split", "target"])["station_id"].nunique().min() == 2


def test_unsupervised_sampling_does_not_require_target() -> None:
    frame = _segments().drop(columns="inferred_state")
    sample = label_free_sample(frame, 8)
    assert len(sample) == 8
    assert sample["station_id"].nunique() == 2
