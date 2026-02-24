import pytest
import time
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "generated"))

from src.vision_model import VisionPipeline

def test_vision_pipeline_shape_and_device():
    pipeline = VisionPipeline(device="cpu")
    dummy_input = torch.rand(1, 64, 1000)
    output = pipeline.extract_features(dummy_input)
    assert output.dim() == 2
    assert output.shape[1] == 256

def test_vision_inference_speed():
    pipeline = VisionPipeline()
    dummy_input = torch.rand(1, 64, 1000)
    
    # Warmup
    for _ in range(5):
        _ = pipeline.extract_features(dummy_input)
        
    # Benchmark
    start = time.perf_counter()
    for _ in range(10):
        _ = pipeline.extract_features(dummy_input)
    end = time.perf_counter()
    
    avg_latency = (end - start) / 10.0
    print(f"Average latency: {avg_latency*1000:.2f}ms")
    
    assert avg_latency < 0.05
