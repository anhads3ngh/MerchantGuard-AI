import numpy as np
from merchantguard.store import ingest
from merchantguard.simulate import make_transactions
from merchantguard.pipeline import materialize_features
from merchantguard.model import train
from merchantguard.monitor import monitor,psi
from merchantguard.agent import investigate


def test_psi_zero_for_identical():
    assert abs(psi(np.array([.5,.5]),np.array([.5,.5])))<1e-10


def test_psi_detects_shift():
    assert psi(np.array([.5,.5]),np.array([.05,.95]))>.5


def test_train_monitor_agent(db,tmp_path):
    ingest(db,make_transactions(4000))
    materialize_features(db)
    meta=train(db,tmp_path/'model.joblib',tmp_path/'meta.json')
    assert meta['test_metrics']['samples']>100
    assert 0<=meta['test_metrics']['pr_auc']<=1
    result=monitor(db,True,tmp_path/'meta.json')
    assert result['alerts_count']>=1
    investigation=investigate(db)
    assert investigation['human_review_required'] is True
    assert investigation['no_automatic_payment_decision'] is True
    assert len(investigation['trace'])>=4


def test_no_alert_agent(db):
    assert investigate(db)['status']=='no_alert'
