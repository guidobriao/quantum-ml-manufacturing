#!/usr/bin/env python3
"""Read-only verification of the Prompt C QML resource report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_DIR = ROOT / "results/qml/resources"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    summary = json.loads((RESOURCE_DIR / "resource_summary.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    for item in summary["inputs"].values():
        if isinstance(item, dict) and "sha256" in item:
            checks[f"hash_{Path(item['path']).name}"] = sha256(ROOT / item["path"]) == item["sha256"]

    with (RESOURCE_DIR / "resource_counts.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    checks["all_models_present"] = {row["model"] for row in rows} == {
        "QSVC", "VQC", "quantum_kernel_clustering"
    }
    checks["six_features_and_qubits"] = all(
        row["selected_features"] == "6" and row["logical_qubits"] == "6" for row in rows
    )

    qsvc = next(row for row in rows if row["model"] == "QSVC" and row["condition"] == "ideal")
    checks["qsvc_arithmetic"] = (
        int(qsvc["kernel_stored_elements"]) == 40 * 40 + 16 * 40 + 24 * 40
        and int(qsvc["kernel_unique_evaluations"]) == 40 * 39 // 2 + 16 * 40 + 24 * 40
        and int(qsvc["kernel_memory_bytes"]) == 8 * (40 * 40 + 16 * 40 + 24 * 40)
    )
    cluster = next(row for row in rows if row["model"] == "quantum_kernel_clustering")
    checks["clustering_arithmetic"] = (
        int(cluster["kernel_stored_elements"]) == 56 * 56
        and int(cluster["kernel_unique_evaluations"]) == 56 * 55 // 2
        and int(cluster["kernel_memory_bytes"]) == 8 * 56 * 56
    )

    with (RESOURCE_DIR / "qpu_shot_estimates.csv").open(newline="", encoding="utf-8") as stream:
        estimates = list(csv.DictReader(stream))
    checks["shot_grid_complete"] = len(estimates) == 9 and {
        int(row["shot"]) for row in estimates
    } == {256, 512, 1024}
    checks["estimates_are_analytical"] = all(
        row["overhead_modeled"] == "false"
        and row["queue_time_estimated"] == "false"
        and row["billing_or_runtime_determined"] == "false"
        for row in estimates
    )
    checks["no_ibm_access_reported"] = (
        not summary["qpu_estimation"]["ibm_service_accessed"]
        and not summary["qpu_estimation"]["api_key_or_crn_used"]
        and not summary["qpu_estimation"]["job_submitted"]
    )
    required = ["README.md", "final_resource_comparison.csv", "resource_counts.csv", "resource_summary.json", "formulas.md"]
    checks["required_outputs_present"] = all((RESOURCE_DIR / name).is_file() for name in required)
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
