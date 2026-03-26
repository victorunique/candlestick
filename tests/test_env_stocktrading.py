import pytest
import pandas as pd
import numpy as np
from src.env_stocktrading import StockTradingEnv

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
        "feature_columns": ["open", "close", "high", "low", "volume", "macd", "rsi_30", "cci_30", "dx_30"],
        "stoploss_penalty": 0.9,
        "profit_loss_ratio": 2,
        "cash_penalty_proportion": 0.1,
        "patient": True,
        "random_start": False,
        "episode_length": -1,
        "window_size": 10
    }

def test_env_init(mock_stock_data, env_kwargs):
    env = StockTradingEnv(df=mock_stock_data, **env_kwargs)
    assert len(env.assets) == 2
    assert env.initial_amount == 1000000
    assert env.action_space.shape[0] == 4  # 2 for trading actions, 2 for stoploss ratios
    
def test_env_reset(mock_stock_data, env_kwargs):
    env = StockTradingEnv(df=mock_stock_data, **env_kwargs)
    state, info = env.reset()
    assert isinstance(state, np.ndarray)
    
    # State shape should be (window_size, state_space)
    # state_space length: cash (1) + holdings (2) + features (2 assets * 9 cols = 18) = 21
    assert state.shape == (10, 21)
    
    # Check the most recent step (last row of the 2D array)
    last_step = state[-1]
    assert last_step[0] == 1000000  # initial cash
    assert last_step[1] == 0  # AAPL holdings
    assert last_step[2] == 0  # MSFT holdings

def test_env_step_buy(mock_stock_data, env_kwargs):
    env = StockTradingEnv(df=mock_stock_data, **env_kwargs)
    state, _ = env.reset()
    
    # Action: Buy 1 * 100 hmax AAPL, Buy 0 MSFT. Stoploss ratios 0.9, 0.9
    # AAPL price at day 0 is 100
    action = np.array([1.0, 0.0, 0.9, 0.9])
    
    next_state, reward, done, truncated, info = env.step(action)
    
    assert next_state.shape == (10, 21)
    
    last_step = next_state[-1]
    cash = last_step[0]
    holdings = last_step[1:3]
    
    assert holdings[0] > 0
    assert holdings[1] == 0
    assert cash < 1000000

def test_env_stoploss_trigger(mock_stock_data, env_kwargs):
    # This specifically tests the stop-loss intervention mechanism
    env = StockTradingEnv(df=mock_stock_data, **env_kwargs)
    env.reset()
    
    # Buy a lot of AAPL on day 0
    env.step(np.array([1.0, 0.0, 0.9, 0.9]))
    
    # Force the avg_buy_price artificially high to guarantee a stop loss drop on day 1
    # Day 1 Low will be 110 - 5 = 105. Let's make average buy price 200, stop loss ratio 0.9 -> threshold 180
    env.avg_buy_price = np.array([200.0, 0.0])
    
    # Try to hold AAPL and do nothing. The environment should intercept and sell it all anyway.
    action = np.array([0.0, 0.0, 0.9, 0.9])
    next_state, reward, done, truncated, info = env.step(action)
    
    last_step = next_state[-1]
    holdings = last_step[1:3]
    # Stoploss should have fired and sold all AAPL
    assert holdings[0] == 0


def test_stoploss_executes_at_threshold_price(mock_stock_data, env_kwargs):
    """
    When stop-loss fires intraday (open > threshold > low), proceeds should be calculated
    at the SL threshold price. If it gaps down (open < threshold), it should execute
    at the open price. This tests the intraday non-gapping case.
    """
    env = StockTradingEnv(df=mock_stock_data, **env_kwargs)
    env.reset()

    # Step 0: Buy AAPL. Action [1.0, 0.0] * hmax=100 -> 100 dollar-units.
    # AAPL close at step 0 = 100, so shares = 100/100 = 1.0 share.
    env.step(np.array([1.0, 0.0, 0.9, 0.9]))
    aapl_holdings = env.state_memory[-1][1]
    cash_after_buy = env.state_memory[-1][0]

    # Stoploss action is 0.9, which maps to ratio = 0.75 + 0.9 * 0.25 = 0.975
    # Force avg_buy_price to 112.0 so SL threshold = 112.0 * 0.975 = 109.2.
    # Step 1: AAPL open = 110, low = 105.
    # Since low (105) < 109.2, SL triggers.
    # Since open (110) > 109.2, execution price is 109.2 (no gap down).
    env.avg_buy_price = np.array([112.0, 0.0])

    action = np.array([0.0, 0.0, 0.9, 0.9])
    next_state, reward, done, truncated, info = env.step(action)

    last_step = next_state[-1]
    cash_after_sl = last_step[0]
    
    # 0.9 SL action maps to 0.975 ratio
    sl_threshold_price = 112.0 * 0.975  # = 109.2
    expected_gross_proceeds = aapl_holdings * sl_threshold_price
    sell_cost = expected_gross_proceeds * env.sell_cost_pct

    # Cash should reflect SL threshold price exactly (109.2), NOT close price (110)
    expected_cash = cash_after_buy + expected_gross_proceeds - sell_cost
    assert abs(cash_after_sl - expected_cash) < 0.01, (
        f"SL should sell at threshold ${sl_threshold_price}, "
        f"expected cash ${expected_cash:.2f}, got ${cash_after_sl:.2f}"
    )


def test_stoploss_no_trigger_when_low_above_threshold(mock_stock_data, env_kwargs):
    """
    If the bar's low stays above the SL threshold, the stop-loss should NOT
    fire and holdings should be preserved.
    """
    env = StockTradingEnv(df=mock_stock_data, **env_kwargs)
    env.reset()

    # Step 0: Buy AAPL
    env.step(np.array([1.0, 0.0, 0.9, 0.9]))
    aapl_holdings_before = env.state_memory[-1][1]
    assert aapl_holdings_before > 0

    # avg_buy_price after buying at close=100 should be 100.
    # SL threshold = 100 * 0.9 = 90.
    # Step 1: AAPL low = 105 which is > 90, so SL should NOT trigger.
    action = np.array([0.0, 0.0, 0.9, 0.9])  # hold
    next_state, _, _, _, _ = env.step(action)

    last_step = next_state[-1]
    aapl_holdings_after = last_step[1]
    assert aapl_holdings_after == aapl_holdings_before, (
        f"SL should not trigger when low (105) > threshold (90), "
        f"but holdings changed from {aapl_holdings_before} to {aapl_holdings_after}"
    )


def test_action_memory_includes_stoploss_mask(mock_stock_data, env_kwargs):
    """save_action_memory() must include a 'stoploss_mask' column that marks
    which steps had a stop-loss triggered (per ticker)."""
    env = StockTradingEnv(df=mock_stock_data, **env_kwargs)
    env.reset()

    # Step 0: Buy AAPL
    env.step(np.array([1.0, 0.0, 0.9, 0.9]))

    # Step 1: Hold — no stop-loss (low=105 > threshold=90)
    env.step(np.array([0.0, 0.0, 0.9, 0.9]))

    # Force avg_buy_price high to guarantee stop-loss on step 2
    env.avg_buy_price = np.array([200.0, 0.0])

    # Step 2: SL should fire (day-2 low = 115 < threshold 180)
    env.step(np.array([0.0, 0.0, 0.9, 0.9]))

    df_actions = env.save_action_memory()
    assert df_actions is not None, "save_action_memory() returned None"
    assert "stoploss_mask" in df_actions.columns, (
        f"Expected 'stoploss_mask' column, got columns: {list(df_actions.columns)}"
    )

    # Parse the stringified arrays
    from src.plot_backtest import _parse_array_str
    masks = df_actions["stoploss_mask"].apply(
        lambda s: _parse_array_str(s) if isinstance(s, str) else np.array(s, dtype=float)
    )

    # Step 0 (buy) and Step 1 (hold, no SL): mask should be all zeros
    np.testing.assert_array_equal(masks.iloc[0], [0.0, 0.0],
                                  err_msg="Step 0 should have no stop-loss")
    np.testing.assert_array_equal(masks.iloc[1], [0.0, 0.0],
                                  err_msg="Step 1 should have no stop-loss")

    # Step 2 (SL fired on AAPL): mask should be [1.0, 0.0]
    np.testing.assert_array_equal(masks.iloc[2], [1.0, 0.0],
                                  err_msg="Step 2 should have stop-loss on AAPL only")
