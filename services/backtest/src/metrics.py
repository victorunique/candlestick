import numpy as np

def calculate_max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0: return 0.0
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (peak - cum_returns) / peak
    return float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods=252*24*60) -> float:
    if len(returns) == 0: return 0.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0: return 0.0
    return float((mean_ret - risk_free_rate) / std_ret * np.sqrt(periods))

def calculate_accuracy(predictions: list, actuals: list) -> float:
    if not len(predictions) or len(predictions) != len(actuals): return 0.0
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    return float(correct / len(predictions))
