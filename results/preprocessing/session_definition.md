# Session definition

Session boundaries are derived independently for each `experiment_id + station_id` using only
unique timestamps from `tblOperationLog`. For every group, the algorithm examines the unique
positive gaps in the empirical upper tail (from the configured 95th percentile), finds its largest
multiplicative jump and, when the two sides are separated by at least the configured ratio, sets
the group-specific threshold to their geometric mean. The quantile, separation criterion and
formula are explicit in `configs/common_pipeline.yaml`; no fixed global duration threshold is used.

A second PLC-only level detects synchronized silences. Every station contributes its configured
number of longest empirical gaps; their temporal overlaps are retained only when supported by the
configured majority of active stations. Contiguous consensus segments are merged and, to prevent
over-segmentation by brief coincident pauses, a segment must exceed the configured fraction of the
longest synchronized duration observed in that experiment. These boundaries augment, but
never remove, the per-station boundaries. All counts, support stations and source gaps are written
to `results/preprocessing/synchronized_plc_silences.csv`.

In configuration version 1 these explicit parameters are: **3 candidate gaps per station**,
**at least 60% of the active stations** (subject also to the explicit minimum count of 2), and a
**minimum relative duration of 20% of the longest synchronized candidate in the same experiment**.
They are parameters of the sessionization algorithm and are not absolute temporal thresholds in
seconds. In particular, 20% is a within-experiment dimensionless comparison; it does not mean a
fixed number of seconds.

The reported synchronized `core` is the interval in which the station consensus holds. A support
station need not be silent at every instant of the merged core: its actual session boundary is tied
to each listed `supporting_station_gaps` interval, whose endpoints are two consecutive
`tblOperationLog` timestamps for that station. “Power inside a synchronized silence is unmatched”
therefore means power strictly inside the relevant station-specific supporting gap. It does not
mean that every station named in the union of support stations has no PLC event throughout the
whole merged consensus core.

Both the local and synchronized levels use only `tblOperationLog`. `tblPowerLog` is assigned after
the PLC sessions have been fixed and cannot introduce, remove, or move a boundary.

Each session is valid only from its first through its last PLC record, inclusively. A power row is
assigned a `session_id` only inside one of those intervals. PLC state is never propagated outside
an interval or across experiment, station, or PLC-session boundaries.

Status: this definition is frozen as the definitive common pipeline after the read-only audit of
2026-09-11. See `results/data_fusion/common_pipeline_audit_2026-09-11.md` and
`results/data_fusion/methodological_decision_common_pipeline.md`.
