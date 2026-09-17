"""Idle-duration residual regression pipeline (TODO phase-2).

Trigger: entry into an idle state (plc_state_machine.IDLE_STATES).
Target: seconds until the next state transition.
Inputs: energy/sensor + PLC/MES features, backward as-of at trigger time.
Models: SVR / RF regressor / XGBoost regressor + QSVR / VQR,
naive baselines (per-station historical median, previous idle).
Imports will come from common_io and qml_common.
"""
