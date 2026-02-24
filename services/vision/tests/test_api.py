import pytest
import io
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "generated"))

import trading_pb2 as trading_pb2
import trading_pb2_grpc as trading_pb2_grpc
from src.main import VisionServiceServicer

@pytest.mark.asyncio
async def test_extract_features_grpc():
    servicer = VisionServiceServicer()
    
    dummy_tensor = torch.rand(1, 64, 1000)
    buffer = io.BytesIO()
    torch.save(dummy_tensor, buffer)
    image_bytes = buffer.getvalue()
    
    # Create request
    request = trading_pb2.ChartImage(
        image_data=image_bytes,
        width=1000,
        height=64
    )
    
    class MockContext:
        def abort(self, code, details):
            raise Exception(details)
            
    response = await servicer.ExtractFeatures(request, MockContext())
    
    assert len(response.features) == 256
