import pytest
import io
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "generated"))

import trading_pb2 as trading_pb2
import trading_pb2_grpc as trading_pb2_grpc
from src.main import ChartServiceServicer

@pytest.mark.asyncio
async def test_generate_chart_grpc():
    servicer = ChartServiceServicer()
    
    # Create request
    request = trading_pb2.OHLCVMatrix(
        open=[100.0] * 10,
        high=[110.0] * 10,
        low=[90.0] * 10,
        close=[105.0] * 10,
        volume=[1000.0] * 10
    )
    
    class MockContext:
        def abort(self, code, details):
            raise Exception(details)
            
    response = await servicer.GenerateChart(request, MockContext())
    
    assert response.width == 10
    assert response.height == 64
    
    # We serialize the tensor with torch.save, so we can check it
    buffer = io.BytesIO(response.image_data)
    tensor = torch.load(buffer, map_location="cpu")
    
    assert tensor.shape == (1, 64, 10)
