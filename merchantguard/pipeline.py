"""Point-in-time feature construction: current event is excluded from historical counts."""
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import pandas as pd
from .store import get_transactions, record_run

FEATURE_COLUMNS = ["tx_id", "event_time", "merchant_id", "merchant_category", "amount", "hour", "day_of_week", "card_present", "international", "avs_match", "cvv_match", "account_age_days", "merchant_age_days", "merchant_txns_24h", "customer_txns_24h", "merchant_declines_24h", "fraud_label"]


def materialize_features(conn) -> dict:
    raw = get_transactions(conn)
    if raw.empty:
        raise ValueError("Ingest transactions before building features")
    merchant_hist = defaultdict(deque)
    customer_hist = defaultdict(deque)
    rows = []
    for tx in raw.itertuples(index=False):
        current = datetime.fromisoformat(tx.event_time.replace("Z", "+00:00"))
        earliest = current - timedelta(hours=24)
        mh = merchant_hist[tx.merchant_id]
        ch = customer_hist[tx.customer_id]
        while mh and mh[0][0] < earliest:
            mh.popleft()
        while ch and ch[0] < earliest:
            ch.popleft()
        rows.append((tx.tx_id, tx.event_time, tx.merchant_id, tx.merchant_category,
                     tx.amount, current.hour, current.weekday(), tx.card_present,
                     tx.international, tx.avs_match, tx.cvv_match, tx.account_age_days,
                     tx.merchant_age_days, len(mh), len(ch), sum(x[1] for x in mh), tx.fraud_label))
        # Add current event only *after* calculating pre-event features.
        mh.append((current, tx.declined))
        ch.append(current)
    placeholders = ",".join(["?"]*len(FEATURE_COLUMNS))
    # Full rebuild is intentional: late arrivals cannot silently corrupt history.
    with conn:
        conn.execute("DELETE FROM predictions")
        conn.execute("DELETE FROM features")
        conn.executemany(f"INSERT INTO features ({','.join(FEATURE_COLUMNS)}) VALUES ({placeholders})", rows)
    result = {"materialized_rows":len(rows), "feature_count":13, "point_in_time_safe":True}
    record_run(conn, "features", result)
    return result
