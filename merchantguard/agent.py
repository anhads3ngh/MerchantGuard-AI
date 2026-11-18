"""Deterministic multi-tool investigation agent; all consequential actions require humans."""
import json
from .store import merchant_overview


def investigate(conn, alert_id: int | None = None) -> dict:
    """Tool-oriented observe -> plan -> investigate -> recommend, with full trace.

    This is intentionally NOT an LLM-powered autonomous financial decision-maker.
    Its policy is explicit, auditable, reproducible and never blocks a payment.
    """
    trace = []
    trace.append({"step":"plan","tool":"lookup_alert","reason":"Retrieve alert and supporting context"})
    if alert_id is None:
        alert = conn.execute("SELECT * FROM alerts ORDER BY alert_id DESC LIMIT 1").fetchone()
    else:
        alert = conn.execute("SELECT * FROM alerts WHERE alert_id=?",(alert_id,)).fetchone()
    if alert is None:
        return {"status":"no_alert","trace":trace,"human_review_required":False}
    alert = dict(alert)
    trace.append({"step":"observe","tool":"lookup_alert","evidence":{"id":alert["alert_id"],"feature":alert["feature_name"],"psi":alert["psi"]}})
    trace.append({"step":"plan","tool":"data_quality_checks","reason":"Separate upstream ingestion problems from distribution change"})
    q = conn.execute("SELECT COUNT(*) AS n, SUM(CASE WHEN amount<=0 THEN 1 ELSE 0 END) AS bad_amounts, COUNT(DISTINCT tx_id) AS unique_ids FROM transactions").fetchone()
    quality = dict(q)
    trace.append({"step":"observe","tool":"data_quality_checks","evidence":quality})
    trace.append({"step":"plan","tool":"merchant_risk_summary","reason":"Inspect affected merchants without exposing customer/card data"})
    merchants = merchant_overview(conn,5)
    trace.append({"step":"observe","tool":"merchant_risk_summary","merchant_count":len(merchants)})
    reason = ("Severe input-feature distribution change" if alert["psi"] >= .25 else "Moderate input-feature distribution change")
    next_actions = ["Validate input schema and late-arriving data", "Compare transaction mix across merchant categories", "Review labels and calibration before any retraining", "Obtain analyst approval for remediation"]
    if quality["bad_amounts"]:
        next_actions.insert(0,"Investigate invalid payment amounts")
    outcome = {"status":"needs_human_review","alert_id":alert["alert_id"],"risk_signal":reason,
               "feature":alert["feature_name"],"psi":alert["psi"],"data_quality":quality,
               "merchant_snapshot":merchants,"next_actions":next_actions,
               "human_review_required":True,"no_automatic_payment_decision":True,"trace":trace}
    with conn:
        conn.execute("INSERT INTO investigations(requested_alert_id,findings_json,decision) VALUES (?,?,?)",
                     (alert["alert_id"],json.dumps(outcome),"human_review_pending"))
    return outcome
