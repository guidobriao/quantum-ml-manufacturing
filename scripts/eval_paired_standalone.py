"""Re-evaluate paired models (MLP, SVR-paired) on the cached QML subsample
without re-running the full idle pipeline. Patches the official tables."""
import sys, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from qml_thesis.idle_duration import regression_metrics, decision_metrics

config = yaml.safe_load((ROOT / "configs" / "idle_duration.yaml").read_text())
seed = config["seed"]

connection = sqlite3.connect(ROOT / "data/processed/idle_duration.sqlite")
subsample = pd.read_sql("SELECT * FROM idle_qml_subsample", connection)

features = config["features"]["quantum_selected"]
x = {s: g[features].to_numpy() for s, g in subsample.groupby("split")}
y = {s: g["target_log1p"].to_numpy() for s, g in subsample.groupby("split")}

scaler = StandardScaler().fit(x["train"])
mlp = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=3000, random_state=seed)
mlp.fit(scaler.transform(x["train"]), y["train"])

metric_rows, prediction_rows = [], []
for split in ["validation", "test"]:
    part = subsample[subsample["split"] == split]
    pred_log = mlp.predict(scaler.transform(x[split]))
    metric_rows.append({"model": "mlp_paired_6feat", "condition": "classical",
                        "split": split, "n": int(len(pred_log)),
                        **regression_metrics(y[split], pred_log)})
    for row, p in zip(part.itertuples(index=False), np.expm1(pred_log)):
        prediction_rows.append({"model": "mlp_paired_6feat", "condition": "classical",
                                "split": split, "station_id": int(row.station_id),
                                "start_ms": int(row.start_ms),
                                "duration_seconds": float(row.duration_seconds),
                                "prediction": float(p)})

metrics = pd.DataFrame(metric_rows)
decision = decision_metrics(pd.DataFrame(prediction_rows),
                            config["decision"]["thresholds_seconds"])
print(metrics.to_string())
print(decision[decision["tt_seconds"] == 28].to_string())

for table in ["idle_metrics", "idle_predictions", "idle_decision_metrics"]:
    connection.execute(f"DELETE FROM {table} WHERE model='mlp_paired_6feat'")
metrics.to_sql("idle_metrics", connection, if_exists="append", index=False)
pd.DataFrame(prediction_rows).to_sql("idle_predictions", connection,
                                      if_exists="append", index=False)
decision.to_sql("idle_decision_metrics", connection, if_exists="append", index=False)
connection.commit()
connection.close()
print("tables patched")
