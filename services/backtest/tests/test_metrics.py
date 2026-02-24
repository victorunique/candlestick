import pytest
import numpy as np
from src.metrics import calculate_max_drawdown, calculate_sharpe_ratio, calculate_accuracy

def test_calculate_max_drawdown():
    returns = np.array([0.1, -0.1, -0.1, 0.1223])
    # Trough relative to peak (110 -> 89.1) = ~19%
    dd = calculate_max_drawdown(returns)
    assert pytest.approx(dd, 0.01) == 0.19

def test_calculate_sharpe_ratio():
    returns = np.array([0.01, 0.02, -0.01, 0.01])
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe > 0

def test_calculate_accuracy():
    preds = ["UP", "DOWN", "SAME", "UP"]
    actuals = ["UP", "UP", "DOWN", "UP"]
    acc = calculate_accuracy(preds, actuals)
    assert acc == 0.5
