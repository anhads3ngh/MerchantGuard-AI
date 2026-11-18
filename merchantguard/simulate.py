"""Deterministic, explicitly synthetic payment-event generator; no personal/card data."""
import numpy as np
import pandas as pd

CATEGORIES = ["restaurants", "retail", "professional_services", "ecommerce", "health_wellness", "home_services"]


def make_transactions(n: int = 12000, seed: int = 2027, merchants: int = 120) -> pd.DataFrame:
    if n < 100 or merchants < 2:
        raise ValueError("At least 100 transactions and 2 merchants are required")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    # Low-frequency merchants and high-volume merchants create realistic bursts.
    merchant_ids = rng.choice(merchants, size=n, p=rng.dirichlet(np.ones(merchants) * 2.0))
    categories = rng.choice(CATEGORIES, merchants, p=[.20, .23, .14, .22, .11, .10])
    merchant_age = rng.integers(60, 2200, size=merchants)
    customer_ids = rng.integers(0, max(500, n // 4), n)
    # Some customers return frequently (nonuniform activity).
    repeat = rng.random(n) < .30
    customer_ids[repeat] = rng.integers(0, 90, repeat.sum())
    # Event timestamp distributed over 70 days with targeted activity peaks.
    time_s = rng.uniform(0, 70 * 86400, n)
    # Correctly generate bursts with one mask per assignment.
    burst = rng.random(n) < .13
    time_s[burst] = rng.uniform(34 * 86400, 35 * 86400, burst.sum())
    amount = np.round(np.clip(rng.lognormal(3.65, 1.08, n), 1.0, 6000), 2)
    card_present = (rng.random(n) < .55).astype(int)
    international = (rng.random(n) < np.where(card_present == 1, .04, .15)).astype(int)
    avs_match = (rng.random(n) < np.where(international == 1, .72, .94)).astype(int)
    cvv_match = (rng.random(n) < np.where(international == 1, .83, .975)).astype(int)
    account_age = rng.integers(1, 1400, n)
    new_mask = rng.random(n) < .20
    account_age[new_mask] = rng.integers(0, 12, new_mask.sum())
    # Hidden simulated high-risk merchant trait produces nontrivial data variation.
    merchant_risk = rng.beta(1.5, 10, size=merchants)
    logit = (-5.5 + .95 * international + 1.5 * (1-avs_match)
             + 1.4 * (1-cvv_match) + .75 * (1-card_present)
             + .70 * (amount > 300) + .55 * (account_age < 10)
             + 3.8 * merchant_risk[merchant_ids])
    risk = 1 / (1 + np.exp(-logit))
    fraud = (rng.random(n) < risk).astype(int)
    decline_prob = np.clip(.035 + .25 * risk + .07 * (1-cvv_match), 0, .8)
    declined = (rng.random(n) < decline_prob).astype(int)
    df = pd.DataFrame({
        "tx_id": [f"txn_{seed}_{i:07d}" for i in range(n)],
        "event_time": [(start + pd.to_timedelta(float(s), unit="s")).strftime("%Y-%m-%dT%H:%M:%SZ") for s in time_s],
        "merchant_id": [f"m_{i:04d}" for i in merchant_ids],
        "customer_id": [f"cust_{i:05d}" for i in customer_ids],
        "merchant_category": categories[merchant_ids],
        "amount": amount,
        "card_present": card_present,
        "international": international,
        "avs_match": avs_match,
        "cvv_match": cvv_match,
        "account_age_days": account_age,
        "merchant_age_days": merchant_age[merchant_ids],
        "declined": declined,
        "fraud_label": fraud,
    })
    return df.sort_values(["event_time", "tx_id"], kind="stable").reset_index(drop=True)
