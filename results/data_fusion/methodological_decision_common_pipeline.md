# Methodological decision: definitive common pipeline

Date: 2026-09-11  
Decision: **accepted and frozen**

The read-only audit found no substantive anomaly in the seven synchronized PLC silences or in
valid matches with `state_age_seconds` from 600 s through the observed maximum of 1205.278 s.
The common preprocessing/data-fusion pipeline is therefore the definitive shared version. Its
fusion logic must not be changed without a new explicit review and approval.

SQLite (`data/processed/common_pipeline.sqlite`) is the canonical dataset. The gzip CSV files are
companion exports only. Sessions are derived exclusively from `tblOperationLog`. The parameters
3 candidate gaps per station, 60% minimum support of active stations, and 20% minimum duration
relative to the longest synchronized candidate of the same experiment are explicit algorithm
parameters, not absolute thresholds in seconds.

No pipeline regeneration and no windowing, feature extraction, clustering, `inferred_state`,
classification, or QML operation was performed during the audit. Full evidence, sample sequences,
power profiles, commands, versions, and hashes are recorded in
`results/data_fusion/common_pipeline_audit_2026-09-11.md`.
