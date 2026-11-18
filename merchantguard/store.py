"""Idempotent SQLite ingestion, auditable pipeline steps, and curated feature store."""
import sqlite3
import json
from pathlib import Path
import pandas as pd
from .config import paths

REQUIRED = ["tx_id", "event_time", "merchant_id", "customer_id", "merchant_category", "amount", "card_present", "international", "avs_match", "cvv_match", "account_age_days", "merchant_age_days", "declined", "fraud_label"]


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db = Path(db_path) if db_path else paths()["db"]
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            tx_id TEXT PRIMARY KEY, event_time TEXT NOT NULL, merchant_id TEXT NOT NULL,
            customer_id TEXT NOT NULL, merchant_category TEXT NOT NULL, amount REAL NOT NULL,
            card_present INTEGER NOT NULL, international INTEGER NOT NULL, avs_match INTEGER NOT NULL,
            cvv_match INTEGER NOT NULL, account_age_days INTEGER NOT NULL,
            merchant_age_days INTEGER NOT NULL, declined INTEGER NOT NULL,
            fraud_label INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_transactions_time ON transactions(event_time);
        CREATE INDEX IF NOT EXISTS idx_transactions_merchant ON transactions(merchant_id, event_time);
        CREATE TABLE IF NOT EXISTS features (
            tx_id TEXT PRIMARY KEY REFERENCES transactions(tx_id), event_time TEXT NOT NULL,
            merchant_id TEXT NOT NULL, merchant_category TEXT NOT NULL, amount REAL NOT NULL,
            hour INTEGER NOT NULL, day_of_week INTEGER NOT NULL, card_present INTEGER NOT NULL,
            international INTEGER NOT NULL, avs_match INTEGER NOT NULL, cvv_match INTEGER NOT NULL,
            account_age_days INTEGER NOT NULL, merchant_age_days INTEGER NOT NULL,
            merchant_txns_24h INTEGER NOT NULL, customer_txns_24h INTEGER NOT NULL,
            merchant_declines_24h INTEGER NOT NULL, fraud_label INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_features_time ON features(event_time);
        CREATE TABLE IF NOT EXISTS predictions (
            tx_id TEXT PRIMARY KEY REFERENCES transactions(tx_id), score REAL NOT NULL,
            needs_review INTEGER NOT NULL, model_version TEXT NOT NULL, scored_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP, details_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            feature_name TEXT NOT NULL, psi REAL NOT NULL, severity TEXT NOT NULL, details_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS investigations (
            investigation_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            requested_alert_id INTEGER, findings_json TEXT NOT NULL, decision TEXT NOT NULL);
    """)
    conn.commit()
    return conn


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = set(REQUIRED).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = df[REQUIRED].copy()
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data["event_time"] = pd.to_datetime(data["event_time"], errors="coerce", utc=True)
    bad = (data["tx_id"].isna() | data["tx_id"].duplicated(keep="first")
           | data["amount"].isna() | (data["amount"] <= 0)
           | data["event_time"].isna()
           | data["merchant_id"].isna() | data["customer_id"].isna()
           | data["merchant_category"].isna())
    for binary in ["card_present", "international", "avs_match", "cvv_match", "declined", "fraud_label"]:
        bad |= ~data[binary].isin([0, 1])
    for col in ["account_age_days", "merchant_age_days"]:
        bad |= pd.to_numeric(data[col], errors="coerce").isna() | (pd.to_numeric(data[col], errors="coerce") < 0)
    valid = data.loc[~bad].copy()
    valid["event_time"] = valid["event_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return valid, {"input_rows": int(len(df)), "valid_rows": int(len(valid)), "quarantined_rows": int(bad.sum()), "duplicate_ids_in_batch": int(data["tx_id"].duplicated().sum())}


def record_run(conn: sqlite3.Connection, stage: str, details: dict):
    conn.execute("INSERT INTO pipeline_runs(stage,details_json) VALUES (?,?)", (stage, json.dumps(details, sort_keys=True)))
    conn.commit()


def ingest(conn: sqlite3.Connection, df: pd.DataFrame) -> dict:
    valid, report = validate(df)
    if valid.empty:
        raise ValueError("No valid rows to ingest")
    before = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    placeholders = ",".join(["?"] * len(REQUIRED))
    conn.executemany(f"INSERT OR IGNORE INTO transactions({','.join(REQUIRED)}) VALUES ({placeholders})", valid[REQUIRED].itertuples(index=False, name=None))
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    report["inserted_rows"] = after - before
    report["skipped_existing_rows"] = len(valid) - report["inserted_rows"]
    record_run(conn, "ingest", report)
    return report


def get_transactions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM transactions ORDER BY event_time, tx_id", conn)


def get_features(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM features ORDER BY event_time, tx_id", conn)


def merchant_overview(conn: sqlite3.Connection, limit=8) -> list[dict]:
    sql = """SELECT t.merchant_id, COUNT(*) AS transactions, ROUND(SUM(t.amount),2) AS volume,
        ROUND(AVG(t.fraud_label)*100,2) AS synthetic_fraud_pct,
        ROUND(AVG(COALESCE(p.score,0)),4) AS avg_model_score,
        SUM(COALESCE(p.needs_review,0)) AS reviews
        FROM transactions t LEFT JOIN predictions p ON t.tx_id=p.tx_id
        GROUP BY t.merchant_id ORDER BY volume DESC LIMIT ?"""
    return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]
