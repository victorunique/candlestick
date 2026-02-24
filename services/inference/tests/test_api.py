import pytest
from fastapi.testclient import TestClient
from src.main import app
import time

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_inference_endpoint(client):
    start = time.perf_counter()
    response = client.get("/api/v1/predict/current?symbol=BTC/USD")
    latency = time.perf_counter() - start
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["UP", "DOWN", "SAME", "UNKNOWN"]
    assert "confidence" in data
    assert latency < 1.0 
