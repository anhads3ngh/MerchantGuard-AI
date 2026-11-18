import pandas as pd
import pytest
from merchantguard.simulate import make_transactions


def test_reproducible_and_sorted():
    first=make_transactions(300,5)
    second=make_transactions(300,5)
    pd.testing.assert_frame_equal(first,second)
    assert first.event_time.is_monotonic_increasing


def test_nontrivial_labels():
    f=make_transactions(5000,2027)
    assert .005 < f.fraud_label.mean() < .2
    assert f.tx_id.is_unique
    assert (f.amount>0).all()


def test_bad_generator_args():
    with pytest.raises(ValueError):make_transactions(9,3)
