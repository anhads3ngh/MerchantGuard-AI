"""CLI demonstration: reproducible simulate -> ingest -> features -> train -> monitor -> triage."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .config import paths
from .simulate import make_transactions
from .store import connect,ingest
from .pipeline import materialize_features
from .model import train
from .monitor import monitor
from .agent import investigate


def demo(n=12000, seed=2027, simulate_shift=True):
    p = paths()
    if p["db"].exists():
        p["db"].unlink()  # Reset only the explicitly configured local demo database.
    raw = make_transactions(n=n, seed=seed)
    raw.to_csv(p["home"] / "synthetic_payments.csv",index=False)
    with connect() as conn:
        ingest_report = ingest(conn,raw)
        pipeline_report = materialize_features(conn)
        model_report = train(conn)
        monitoring = monitor(conn,simulate_shift=simulate_shift)
        triage = investigate(conn)
    report = {"ingestion":ingest_report,"feature_pipeline":pipeline_report,
              "model":{"version":model_report["model_version"],"test_metrics":model_report["test_metrics"]},
              "monitor":{"alerts_count":monitoring["alerts_count"],"simulated_shift":simulate_shift},
              "agent":{"status":triage["status"],"human_review_required":triage["human_review_required"]}}
    p["report"].write_text(json.dumps(report,indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="MerchantGuard synthetic payments data platform")
    sub = parser.add_subparsers(dest="command",required=True)
    d = sub.add_parser("demo",help="Run complete reproducible workflow")
    d.add_argument("--rows",type=int,default=12000)
    d.add_argument("--seed",type=int,default=2027)
    d.add_argument("--no-shift",action="store_true",help="Disable showcase distribution shift")
    i = sub.add_parser("ingest",help="Ingest user-provided CSV with schema validation")
    i.add_argument("csv",type=Path)
    sub.add_parser("features",help="Rebuild point-in-time feature store")
    sub.add_parser("train",help="Chronological train/validation/test ML workflow")
    m = sub.add_parser("monitor",help="Monitor PSI and optionally show an injected shift")
    m.add_argument("--simulate-shift",action="store_true")
    sub.add_parser("triage",help="Run auditable human-in-loop tool-based investigation")
    args = parser.parse_args()
    if args.command == "demo":
        result = demo(args.rows,args.seed,not args.no_shift)
    else:
        with connect() as conn:
            if args.command == "ingest":result=ingest(conn,pd.read_csv(args.csv))
            elif args.command == "features":result=materialize_features(conn)
            elif args.command == "train":result=train(conn)
            elif args.command == "monitor":result=monitor(conn,args.simulate_shift)
            elif args.command == "triage":result=investigate(conn)
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
