import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "generated"))

import asyncio
import grpc
from fastapi import FastAPI, Depends
from loguru import logger
from contextlib import asynccontextmanager
import uvicorn
import pandas as pd
import numpy as np
import datetime

import src.generated.trading_pb2 as trading_pb2
import src.generated.trading_pb2_grpc as trading_pb2_grpc
from src.data_manager import DataManager

# Global DataManager instance
data_manager = DataManager()

def get_data_manager():
    return data_manager

class DataServiceServicer(trading_pb2_grpc.DataServiceServicer):
    def __init__(self, manager: DataManager):
        self.manager = manager
        
    async def GetSlidingWindow(self, request, context):
        try:
            timestamp = pd.to_datetime(request.end_timestamp)
            df = self.manager.get_sliding_window(request.symbol, timestamp, request.window_size)
            
            return trading_pb2.OHLCVMatrix(
                open=df["open"].tolist(),
                high=df["high"].tolist(),
                low=df["low"].tolist(),
                close=df["close"].tolist(),
                volume=df["volume"].tolist()
            )
        except Exception as e:
            logger.error(f"Error serving GetSlidingWindow: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))

async def serve_grpc():
    server = grpc.aio.server()
    trading_pb2_grpc.add_DataServiceServicer_to_server(DataServiceServicer(data_manager), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    logger.info("gRPC server started on port 50051")
    await server.wait_for_termination()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup duckdb schema
    data_manager.setup_schema()
    
    # Run gRPC server in background
    grpc_task = asyncio.create_task(serve_grpc())
    yield
    grpc_task.cancel()

app = FastAPI(lifespan=lifespan)

@app.get("/api/v1/data/historical")
async def fetch_historical(symbol: str, granularity: str, dm: DataManager = Depends(get_data_manager)):
    # Mocking fetching historical data
    # In reality, you'd fetch from Binance/Alpaca
    logger.info(f"Fetching historical data for {symbol} with resolution {granularity}")
    dates = pd.date_range("2026-01-01", periods=2000, freq=granularity)
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": symbol,
        "open": np.random.rand(2000),
        "high": np.random.rand(2000),
        "low": np.random.rand(2000),
        "close": np.random.rand(2000),
        "volume": np.random.rand(2000) * 100
    })
    dm.ingest_dataframe(df)
    return {"status": "success", "message": f"Ingested 2000 rows for {symbol}"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8001, reload=True)
