# Verified common-pipeline report

- Raw archive: 45 files, 163719816 bytes; before/after SHA-256 check passed.
- Canonical SQL dumps: 15.
- Selected snapshot rows: 955596.
- Canonical power rows: 773846.
- Canonical operation rows: 34257.
- Deduplicated complete-row occurrences: 147493.
- Same ResourceID/timestamp conflict groups: 0.
- Sessions: 86.
- Power rows outside PLC session intervals: 173343.
- Fused rows: 773846 (exactly one per canonical power row).
- Join coverage: 77.600%.
- Synchronized PLC silences: 7.
- Matches with state age above 1800 s: 0.
- Empty tables verified: {'tblMachineReport': 0, 'tblSensorsLog': 0}.
- Always-NULL PLC/MES fields verified: {'DoneWorkingxBG9': True, 'WorkPlanNo': True, 'StepNo': True}.
- Canonical storage format: SQLite. Compressed CSV files are companion exports.
- Parquet available locally: False.
- Forbidden downstream steps performed: none.
