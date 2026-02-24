import pytest
import pandas as pd
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "generated"))

import trading_pb2 as trading_pb2
import trading_pb2_grpc as trading_pb2_grpc
from src.main import app, DataServiceServicer, get_data_manager
from src.data_manager import DataManager

@pytest.fixture
def mock_db_manager():
    manager = DataManager(":memory:")
    manager.setup_schema()
    dates = pd.date_range("2026-01-01", periods=1500, freq="1min")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "BTC/USD",
        "open": [1.0] * 1500,
        "high": [1.1] * 1500,
        "low": [0.9] * 1500,
        "close": [1.0] * 1500,
        "volume": [100.0] * 1500
    })
    manager.ingest_dataframe(df)
    return manager

@pytest.fixture
def test_client(mock_db_manager):
    app.dependency_overrides[get_data_manager] = lambda: mock_db_manager
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

def test_historical_download_api(test_client):
    response = test_client.get("/api/v1/data/historical?symbol=ETH/USD&granularity=1min")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

@pytest.mark.asyncio
async def test_grpc_get_sliding_window(mock_db_manager):
    servicer = DataServiceServicer(mock_db_manager)
    # The dataframe has dates up to early Jan 2nd
    request = trading_pb2.SlidingWindowRequest(
        symbol="BTC/USD",
        end_timestamp="2026-01-02 00:00:00",
        window_size=1000
    )
    
    class MockContext:
        def abort(self, code, details):
            raise Exception(details)

    response = await servicer.GetSlidingWindow(request, MockContext())
    assert len(response.open) == 1000
    assert response.close[0] == 1.0
