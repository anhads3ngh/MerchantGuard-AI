import pytest
from merchantguard.store import connect

@pytest.fixture
def db(tmp_path):
    c = connect(tmp_path / "fixture.sqlite")
    yield c
    c.close()
