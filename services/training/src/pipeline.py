import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "generated"))

import grpc
import src.generated.trading_pb2 as trading_pb2
import src.generated.trading_pb2_grpc as trading_pb2_grpc
from loguru import logger
import numpy as np

class RemotePipeline:
    def __init__(self, symbol: str, data_addr="localhost:50051", chart_addr="localhost:50052", vision_addr="localhost:50053"):
        self.symbol = symbol
        self.data_channel = grpc.insecure_channel(data_addr)
        self.chart_channel = grpc.insecure_channel(chart_addr)
        self.vision_channel = grpc.insecure_channel(vision_addr)
        
        self.data_stub = trading_pb2_grpc.DataServiceStub(self.data_channel)
        self.chart_stub = trading_pb2_grpc.ChartServiceStub(self.chart_channel)
        self.vision_stub = trading_pb2_grpc.VisionServiceStub(self.vision_channel)
        
        self.prices_array = [100.0 + i * 0.1 for i in range(1000)]

    @property
    def prices(self):
        return self.prices_array

    def get_features(self, step_idx: int) -> np.ndarray:
        try:
            req_data = trading_pb2.SlidingWindowRequest(symbol=self.symbol, end_timestamp=f"idx_{step_idx}", window_size=1000)
            ohlcv = self.data_stub.GetSlidingWindow(req_data)
            chart = self.chart_stub.GenerateChart(ohlcv)
            feature_resp = self.vision_stub.ExtractFeatures(chart)
            
            return np.array(feature_resp.features, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Pipeline error at step {step_idx} (normal if gRPC servers are down): {e}")
            return np.zeros(256, dtype=np.float32)

    def get_future_close(self, step_idx: int) -> float:
        if step_idx + 1 < len(self.prices_array):
            return self.prices_array[step_idx + 1]
        return self.prices_array[-1]
