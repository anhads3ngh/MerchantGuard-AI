import pandas as pd
import pytest
from merchantguard.store import ingest,validate,get_features
from merchantguard.pipeline import materialize_features
from merchantguard.simulate import make_transactions


def test_idempotent_ingestion(db):
    d=make_transactions(180)
    a=ingest(db,d);b=ingest(db,d)
    assert a['inserted_rows']==180
    assert b['inserted_rows']==0
    assert b['skipped_existing_rows']==180


def test_validation_rejects_bad_amount(db):
    d=make_transactions(100)
    d.loc[0,'amount']=-12
    report=ingest(db,d)
    assert report['quarantined_rows']==1
    assert report['inserted_rows']==99


def test_validation_rejects_missing_column():
    with pytest.raises(ValueError):validate(pd.DataFrame({'tx_id':['a']}))


def test_pit_excludes_current_event(db):
    d=make_transactions(200)
    d['merchant_id']='m_one'
    d['customer_id']='c_one'
    d['declined']=0
    ingest(db,d)
    materialize_features(db)
    f=get_features(db)
    assert f.iloc[0].merchant_txns_24h==0
    assert f.iloc[0].customer_txns_24h==0
    assert f.iloc[0].merchant_declines_24h==0
    assert f.iloc[1].merchant_txns_24h>=0
    assert len(f)==len(d)


def test_feature_rebuild_is_idempotent(db):
    ingest(db,make_transactions(150))
    a=materialize_features(db);b=materialize_features(db)
    assert a['materialized_rows']==b['materialized_rows']==150
