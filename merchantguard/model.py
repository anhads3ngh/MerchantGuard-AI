"""Chronological risk modeling with PR-AUC and fixed review-capacity evaluation."""
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from .config import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, paths
from .store import get_features, record_run


def model_pipeline():
    num_pipe = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    prep = ColumnTransformer([("num", num_pipe, NUMERIC_FEATURES),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    return Pipeline([("transform", prep), ("classifier", LogisticRegression(max_iter=650, class_weight="balanced", random_state=2027))])


def distribution_baseline(df: pd.DataFrame) -> dict:
    baseline = {}
    for col in ["amount", "account_age_days", "merchant_txns_24h", "customer_txns_24h", "merchant_declines_24h"]:
        values = df[col].astype(float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11))).tolist()
        if len(edges) < 2:
            edges = [float(values.min()) - 0.5, float(values.max()) + 0.5]
        # -inf/+inf capture new values outside training ranges.
        hist = np.histogram(values, bins=np.array([-np.inf] + edges[1:-1] + [np.inf]))[0]
        baseline[col] = {"edges": edges[1:-1], "shares": (hist / hist.sum()).tolist()}
    return baseline


def _metric(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    reviewed = scores >= threshold
    positives = (y == 1)
    return {
        "samples": len(y), "fraud_prevalence": round(float(positives.mean()), 5),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4) if len(np.unique(y)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y, scores)), 4) if positives.any() else None,
        "review_rate": round(float(reviewed.mean()), 4),
        "precision_at_capacity": round(float(positives[reviewed].mean()), 4) if reviewed.any() else 0,
        "recall_at_capacity": round(float(reviewed[positives].mean()), 4) if positives.any() else 0,
    }


def train(conn, model_path: Path | None = None, metadata_path: Path | None = None) -> dict:
    data = get_features(conn)
    if len(data) < 100:
        raise ValueError("At least 100 feature rows required")
    p = paths()
    model_path = Path(model_path) if model_path else p["model"]
    metadata_path = Path(metadata_path) if metadata_path else p["metadata"]
    n = len(data)
    train_df = data.iloc[:int(n*.7)]
    valid_df = data.iloc[int(n*.7):int(n*.85)]
    test_df = data.iloc[int(n*.85):]
    if min(train_df.fraud_label.sum(), valid_df.fraud_label.sum(), test_df.fraud_label.sum()) == 0:
        raise ValueError("Each chronological split requires positive examples; generate a larger dataset")
    pipeline = model_pipeline()
    pipeline.fit(train_df[FEATURES], train_df.fraud_label)
    val_scores = pipeline.predict_proba(valid_df[FEATURES])[:,1]
    test_scores = pipeline.predict_proba(test_df[FEATURES])[:,1]
    threshold = float(np.quantile(val_scores, .98))
    version = f"merchantguard-logreg-{datetime.now(timezone.utc).strftime('%Y%m%d')}-v1"
    metadata = {
        "model_version": version, "training_rows": len(train_df), "validation_rows":len(valid_df), "test_rows":len(test_df),
        "split_strategy":"chronological 70/15/15; validation sets 2% review capacity; held-out test",
        "review_threshold": threshold, "review_capacity": .02,
        "validation_metrics":_metric(valid_df.fraud_label.to_numpy(), val_scores, threshold),
        "test_metrics":_metric(test_df.fraud_label.to_numpy(), test_scores, threshold),
        "training_baseline": distribution_baseline(train_df),
        "test_date_range": [str(test_df.event_time.min()), str(test_df.event_time.max())],
        "disclaimer":"Synthetic labels and metrics; not a deployed fraud-prevention system"
    }
    model_path.parent.mkdir(parents=True,exist_ok=True)
    metadata_path.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata,indent=2))
    with conn:
        conn.execute("DELETE FROM predictions")
        conn.executemany("INSERT INTO predictions(tx_id,score,needs_review,model_version) VALUES (?,?,?,?)",
                         [(str(tx), float(score), int(score >= threshold),version)
                          for tx,score in zip(test_df.tx_id, test_scores)])
    record_run(conn, "train", {k: metadata[k] for k in ["model_version", "training_rows", "test_metrics"]})
    return metadata


def load_model(model_path: Path | None = None):
    file = Path(model_path) if model_path else paths()["model"]
    if not file.is_file():
        raise FileNotFoundError(f"No trained model at {file}. Run: merchantguard demo")
    return joblib.load(file)


def load_metadata(metadata_path: Path | None = None):
    file = Path(metadata_path) if metadata_path else paths()["metadata"]
    if not file.is_file():
        raise FileNotFoundError(f"No model metadata at {file}")
    return json.loads(file.read_text())
