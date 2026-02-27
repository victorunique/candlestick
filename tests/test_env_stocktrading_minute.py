import pytest
import pandas as pd
import numpy as np
import gymnasium as gym
from src.env_stocktrading_minute import StockTradingEnvMinute

@pytest.fixture
def mock_stock_data():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    data = []
    
    # Simple deterministic prices to test math
    for i, date in enumerate(dates):
        # AAPL goes up, MSFT is flat
        aapl_price = 100 + i * 10
        msft_price = 200
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "tic": "AAPL",
            "open": aapl_price,
            "high": aapl_price + 5,
            "low": aapl_price - 5,
            "close": aapl_price,
            "volume": 1000,
            "macd": 0, "rsi_30": 50, "cci_30": 0, "dx_30": 20
        })
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "tic": "MSFT",
            "open": msft_price,
            "high": msft_price + 5,
            "low": msft_price - 5,
            "close": msft_price,
            "volume": 1000,
            "macd": 0, "rsi_30": 50, "cci_30": 0, "dx_30": 20
        })
        
    df = pd.DataFrame(data)
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    return df

@pytest.fixture
def env_kwargs():
    return {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "print_verbosity": 10,
        "discrete_actions": False,
        "daily_information_cols": ["open", "close", "high", "low", "volume", "macd", "rsi_30", "cci_30", "dx_30"],
        "stoploss_penalty": 0.9,
        "profit_loss_ratio": 2,
        "cash_penalty_proportion": 0.1,
        "patient": True,
        "random_start": False,
        "episode_length": -1
    }

def test_env_init(mock_stock_data, env_kwargs):
    env = StockTradingEnvMinute(df=mock_stock_data, **env_kwargs)
    assert len(env.assets) == 2
    assert env.initial_amount == 1000000
    assert env.action_space.shape[0] == 4  # 2 for trading actions, 2 for stoploss ratios
    
def test_env_reset(mock_stock_data, env_kwargs):
    env = StockTradingEnvMinute(df=mock_stock_data, **env_kwargs)
    state, info = env.reset()
    assert isinstance(state, list) or isinstance(state, np.ndarray)
    
    # State length: cash (1) + holdings (2) + features (2 assets * 9 cols = 18) = 21
    assert len(state) == 21 
    assert state[0] == 1000000  # initial cash
    assert state[1] == 0  # AAPL holdings
    assert state[2] == 0  # MSFT holdings

def test_env_step_buy(mock_stock_data, env_kwargs):
    env = StockTradingEnvMinute(df=mock_stock_data, **env_kwargs)
    state, _ = env.reset()
    
    # Action: Buy 1 * 100 hmax AAPL, Buy 0 MSFT. Stoploss ratios 0.9, 0.9
    # AAPL price at day 0 is 100
    action = np.array([1.0, 0.0, 0.9, 0.9])
    
    next_state, reward, done, truncated, info = env.step(action)
    
    cash = next_state[0]
    holdings = next_state[1:3]
    
    assert holdings[0] > 0
    assert holdings[1] == 0
    assert cash < 1000000

def test_env_stoploss_trigger(mock_stock_data, env_kwargs):
    # This specifically tests the stop-loss intervention mechanism
    env = StockTradingEnvMinute(df=mock_stock_data, **env_kwargs)
    env.reset()
    
    # Buy a lot of AAPL on day 0
    env.step(np.array([1.0, 0.0, 0.9, 0.9]))
    
    # Force the avg_buy_price artificially high to guarantee a stop loss drop on day 1
    # Day 1 Low will be 110 - 5 = 105. Let's make average buy price 200, stop loss ratio 0.9 -> threshold 180
    env.avg_buy_price = np.array([200.0, 0.0])
    
    # Try to hold AAPL and do nothing. The environment should intercept and sell it all anyway.
    action = np.array([0.0, 0.0, 0.9, 0.9])
    next_state, reward, done, truncated, info = env.step(action)
    
    holdings = next_state[1:3]
    # Stoploss should have fired and sold all AAPL
    assert holdings[0] == 0
