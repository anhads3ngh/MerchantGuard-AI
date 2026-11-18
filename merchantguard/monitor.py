"""Population Stability Index based monitoring for numeric features."""
import json
from pathlib import Path
import numpy as np
from .config import paths
from .model import load_metadata
from .store import get_features, record_run


def psi(expected: np.ndarray, observed: np.ndarray) -> float:
    """PSI against pre-bucketed normalized histograms; epsilon guards empty bins."""
    a = np.maximum(np.asarray(expected,dtype=float), 1e-6)
    b = np.maximum(np.asarray(observed,dtype=float), 1e-6)
    a /= a.sum(); b /= b.sum()
    return float(np.sum((b-a)*np.log(b/a)))


def monitor(conn, simulate_shift: bool = False, metadata_path: Path | None = None) -> dict:
    meta = load_metadata(metadata_path)
    data = get_features(conn)
    if data.empty:
        raise ValueError("Build features before monitoring")
    recent = data.iloc[int(len(data)*.85):].copy()
    issues = []
    measures = {}
    for col, base in meta["training_baseline"].items():
        values = recent[col].astype(float).to_numpy().copy()
        if simulate_shift and col in ("amount", "merchant_txns_24h"):
            values = values*5 + (300 if col == "amount" else 15)
        cuts = np.array([-np.inf]+base["edges"]+[np.inf])
        observed = np.histogram(values,bins=cuts)[0]
        observed = observed / max(observed.sum(), 1)
        score = round(psi(np.array(base["shares"]), observed),5)
        severity = "high" if score >= .25 else "moderate" if score >= .1 else "normal"
        measures[col] = {"psi":score,"severity":severity}
        if severity != "normal":
            issues.append({"feature":col,"psi":score,"severity":severity,
                           "message":"Distribution shifted; investigate data quality before retraining."})
    with conn:
        for issue in issues:
            conn.execute("INSERT INTO alerts(feature_name,psi,severity,details_json) VALUES (?,?,?,?)",
                         (issue["feature"],issue["psi"],issue["severity"],json.dumps(issue)))
    report = {"sample_size":len(recent),"simulated_shift":simulate_shift,
              "measures":measures,"alerts":issues,"alerts_count":len(issues)}
    record_run(conn,"monitor",{"alert_count":len(issues),"simulated_shift":simulate_shift})
    return report
