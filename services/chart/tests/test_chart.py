import pytest
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "generated"))

from src.chart_generator import ChartGenerator

def test_chart_generation_shapes():
    window_size = 1000
    open_prices = np.random.uniform(100, 200, window_size)
    high_prices = open_prices + np.random.uniform(0, 10, window_size)
    low_prices = open_prices - np.random.uniform(0, 10, window_size)
    close_prices = open_prices + np.random.uniform(-5, 5, window_size)
    volume = np.random.uniform(100, 1000, window_size)
    
    generator = ChartGenerator(width=1000, height=64)
    tensor = generator.generate_chart(open_prices, high_prices, low_prices, close_prices, volume)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 64, 1000)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

def test_deterministic_rendering():
    window_size = 10
    open_p = np.full(window_size, 100.0)
    high_p = np.full(window_size, 110.0)
    low_p = np.full(window_size, 90.0)
    close_p = np.full(window_size, 105.0)
    vol = np.full(window_size, 1000.0)
    
    generator = ChartGenerator(width=10, height=64)
    
    t1 = generator.generate_chart(open_p, high_p, low_p, close_p, vol)
    t2 = generator.generate_chart(open_p, high_p, low_p, close_p, vol)
    
    assert torch.allclose(t1, t2)
