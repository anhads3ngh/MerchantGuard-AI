"""Explicit runtime paths; override MERCHANTGUARD_HOME for isolation/deployment."""
from pathlib import Path
import os


def data_home() -> Path:
    return Path(os.environ.get("MERCHANTGUARD_HOME", Path.cwd() / ".merchantguard")).expanduser().resolve()


def paths() -> dict[str, Path]:
    home = data_home()
    home.mkdir(parents=True, exist_ok=True)
    return {"home": home, "db": home / "payments.db", "model": home / "risk_model.joblib", "metadata": home / "model_metadata.json", "report": home / "run_report.json"}

NUMERIC_FEATURES = [
    "amount", "hour", "day_of_week", "card_present", "international",
    "avs_match", "cvv_match", "account_age_days", "merchant_age_days",
    "merchant_txns_24h", "customer_txns_24h", "merchant_declines_24h",
]
CATEGORICAL_FEATURES = ["merchant_category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
