"""FastAPI read-only scoring API; synthetic demo only, not PCI-compliant."""
from contextlib import contextmanager
from typing import Literal
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .config import FEATURES
from .store import connect, merchant_overview
from .model import load_model, load_metadata
from .agent import investigate

app = FastAPI(title="MerchantGuard API", version="1.0.0", description="Synthetic transaction-risk prototype (human review only)")


class TransactionFeatures(BaseModel):
    amount: float = Field(gt=0)
    hour: int = Field(ge=0,le=23)
    day_of_week: int = Field(ge=0,le=6)
    card_present: Literal[0,1]
    international: Literal[0,1]
    avs_match: Literal[0,1]
    cvv_match: Literal[0,1]
    account_age_days: int = Field(ge=0)
    merchant_age_days: int = Field(ge=0)
    merchant_txns_24h: int = Field(ge=0)
    customer_txns_24h: int = Field(ge=0)
    merchant_declines_24h: int = Field(ge=0)
    merchant_category: str = Field(min_length=1,max_length=64)


@app.get("/health")
def health():
    return {"status":"ok","system":"MerchantGuard synthetic demo"}


@app.post("/predict")
def predict(item: TransactionFeatures):
    try:
        model, meta = load_model(), load_metadata()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    x = pd.DataFrame([item.model_dump()])[FEATURES]
    score = float(model.predict_proba(x)[0,1])
    return {"risk_score":round(score,6),"needs_manual_review":score >= meta["review_threshold"],
            "model_version":meta["model_version"],"notice":"A score is not a fraud verdict. Synthetic-data demonstration only."}


@app.get("/model/metrics")
def model_metrics():
    try:
        meta = load_metadata()
        return {"model_version":meta["model_version"],"test_metrics":meta["test_metrics"],"split_strategy":meta["split_strategy"]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503,detail=str(e)) from e


@app.get("/merchants/overview")
def merchants(limit: int = 8):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="Limit must be 1..100")
    with connect() as conn:
        return merchant_overview(conn,limit)


@app.get("/alerts")
def alerts(limit: int = 10):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422,detail="Limit must be 1..100")
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT alert_id,created_at,feature_name,psi,severity FROM alerts ORDER BY alert_id DESC LIMIT ?",(limit,))]


@app.post("/investigate/latest")
def investigate_latest():
    with connect() as conn:
        return investigate(conn)
