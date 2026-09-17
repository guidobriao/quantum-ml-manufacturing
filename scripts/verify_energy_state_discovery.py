#!/usr/bin/env python3
"""Independent read-only checks for classical energy-state discovery outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from qml_thesis.energy_state_discovery import (
    BOOLEAN_PLC, DISCOVERY_FEATURES, IDENTIFIERS, MES_FIELDS, sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/common_pipeline.sqlite"
OUTPUT = ROOT / "data/processed/energy_state_discovery.sqlite"
SUMMARY = ROOT / "results/energy_state_discovery/run_summary.json"
RAW_MANIFEST = ROOT / "data/raw_manifest.csv"


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def main() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    checks["source_sqlite_unchanged"] = sha256_file(SOURCE) == summary["source_sqlite_sha256"]
    checks["raw_manifest_unchanged"] = sha256_file(RAW_MANIFEST) == summary["raw_manifest_sha256"]
    checks["no_forbidden_steps"] = summary["forbidden_steps_performed"] == []
    with sqlite3.connect(f"file:{SOURCE}?mode=ro", uri=True) as source, sqlite3.connect(f"file:{OUTPUT}?mode=ro", uri=True) as output:
        source_rows = source.execute("SELECT COUNT(*) FROM power_operation_fused WHERE session_id IS NOT NULL").fetchone()[0]
        sample_rows = output.execute("SELECT COUNT(*) FROM segmented_power_samples").fetchone()[0]
        primary_segments = output.execute("SELECT COUNT(*) FROM segments_with_inferred_state").fetchone()[0]
        x_rows = output.execute("SELECT COUNT(*) FROM X_discovery_energy_sensor").fetchone()[0]
        z_rows = output.execute("SELECT COUNT(*) FROM Z_plc_mes_validation").fetchone()[0]
        checks["all_in_session_power_rows_segmented"] = source_rows == sample_rows
        checks["sample_row_identity_unique"] = output.execute(
            "SELECT COUNT(*) FROM (SELECT canonical_row_id FROM segmented_power_samples GROUP BY canonical_row_id HAVING COUNT(*)<>1)"
        ).fetchone()[0] == 0
        checks["segments_do_not_cross_groups"] = output.execute(
            "SELECT COUNT(*) FROM (SELECT segment_id FROM segmented_power_samples GROUP BY segment_id HAVING COUNT(DISTINCT experiment_id)<>1 OR COUNT(DISTINCT station_id)<>1 OR COUNT(DISTINCT session_id)<>1)"
        ).fetchone()[0] == 0
        checks["fixed_5s_span_respected"] = output.execute(
            "SELECT COUNT(*) FROM segments_with_inferred_state WHERE duration_seconds>5.0 OR segmentation_config<>'fixed_5s'"
        ).fetchone()[0] == 0
        checks["required_features_complete"] = output.execute(
            "SELECT COUNT(*) FROM segments_with_inferred_state WHERE duration_seconds IS NULL OR energy_joule IS NULL OR power_mean IS NULL"
        ).fetchone()[0] == 0
        x_columns = columns(output, "X_discovery_energy_sensor")
        checks["x_has_exact_nonsemantic_schema"] = x_columns == set(IDENTIFIERS + DISCOVERY_FEATURES)
        checks["x_excludes_plc_mes"] = not (x_columns & set(BOOLEAN_PLC + MES_FIELDS))
        checks["x_z_aligned"] = x_rows == z_rows and output.execute(
            "SELECT COUNT(*) FROM X_discovery_energy_sensor x LEFT JOIN Z_plc_mes_validation z USING(segment_id) WHERE z.segment_id IS NULL"
        ).fetchone()[0] == 0
        checks["inferred_state_complete"] = output.execute(
            "SELECT COUNT(*) FROM segments_with_inferred_state WHERE inferred_state IS NULL OR inferred_state_is_pseudo_label<>1"
        ).fetchone()[0] == 0
        checks["unclusterable_segments_are_unknown"] = output.execute(
            "SELECT COUNT(*) FROM segments_with_inferred_state WHERE cluster IS NULL AND inferred_state<>'unknown'"
        ).fetchone()[0] == 0
        checks["cluster_mapping_explicit"] = output.execute("SELECT COUNT(*) FROM cluster_state_mapping").fetchone()[0] == summary["selected_cluster_count"]
        details.update(source_rows=source_rows, segmented_rows=sample_rows, primary_segments=primary_segments, x_rows=x_rows, z_rows=z_rows)
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "details": details}
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
