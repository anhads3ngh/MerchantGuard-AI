from fastapi.testclient import TestClient
from merchantguard.api import app

client=TestClient(app)


def test_health():
    r=client.get('/health')
    assert r.status_code==200
    assert r.json()['status']=='ok'


def test_invalid_payload():
    r=client.post('/predict',json={'amount':-2})
    assert r.status_code==422


def test_limits():
    r=client.get('/alerts?limit=101')
    assert r.status_code==422
