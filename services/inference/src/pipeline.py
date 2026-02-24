import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "generated"))

import grpc
import src.generated.trading_pb2 as trading_pb2
import src.generated.trading_pb2_grpc as trading_pb2_grpc
from loguru import logger
import numpy as np
import datetime

class InferencePipeline:
    def __init__(self, data_addr="localhost:50051", chart_addr="localhost:50052", vision_addr="localhost:50053"):
        self.data_channel = grpc.insecure_channel(data_addr)
        self.chart_channel = grpc.insecure_channel(chart_addr)
        self.vision_channel = grpc.insecure_channel(vision_addr)
        
        self.data_stub = trading_pb2_grpc.DataServiceStub(self.data_channel)
        self.chart_stub = trading_pb2_grpc.ChartServiceStub(self.chart_channel)
        self.vision_stub = trading_pb2_grpc.VisionServiceStub(self.vision_channel)

    def get_current_features(self, symbol: str) -> np.ndarray:
        try:
            now = datetime.datetime.utcnow().isoformat()
            
            req_data = trading_pb2.SlidingWindowRequest(symbol=symbol, end_timestamp=now, window_size=1000)
            ohlcv = self.data_stub.GetSlidingWindow(req_data)
            chart = self.chart_stub.GenerateChart(ohlcv)
            feature_resp = self.vision_stub.ExtractFeatures(chart)
            
            return np.array(feature_resp.features, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Inference pipeline fallback (expected if backends are offline): {e}")
            return np.zeros(256, dtype=np.float32)
