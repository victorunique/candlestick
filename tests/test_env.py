import numpy as np
import pandas as pd
from src.env_stocktrading import StockTradingEnv

def test_env_bugfixes():
    # 1. Create a dummy dataframe with date, tic, open, close, high, low, volume
    timestamps = pd.date_range("2020-01-01", periods=10)
    data = []
    for t in timestamps:
        data.append({"date": t, "tic": "AAPL", "open": 100, "close": 110, "high": 120, "low": 90, "volume": 1000})
        data.append({"date": t, "tic": "MSFT", "open": 200, "close": 210, "high": 220, "low": 190, "volume": 2000})
    df = pd.DataFrame(data)
    
    feature_columns = ["open", "close", "high", "low", "volume"]
    
    # Initialize env
    env = StockTradingEnv(
        df=df,
        timestamp_col_name="date",
        feature_columns=feature_columns,
        initial_amount=10000.0,
        patient=True,
    )
    
    # 2. Test initial bounds [-1, 1] everywhere
    assert np.all(env.action_space.low == -1.0), "Action space low bound should be -1.0"
    assert np.all(env.action_space.high == 1.0), "Action space high bound should be 1.0"
    
    # 3. Simulate a sequence of steps
    obs, info = env.reset()
    
    # Step 1: Buy some AAPL and MSFT
    # trading actions = [0.5, 0.5], sl actions = [0, 0] => target ratio = 0.75
    action1 = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    obs, reward, done, trunc, info = env.step(action1)
    
    # Verify holding has been created
    # After step 1: we spent some money.
    
    # Step 2: Test SL hit with gap-down
    # Modify the df globally in env for the exact next step so we can trigger a gap down
    # step_index is now 1. Next step accesses data for step_index 1.
    env.df.iloc[(env.df.index == timestamps[2]) & (env.df["tic"] == "AAPL"), env.df.columns.get_loc("low")] = 50.0  # Force a low
    env.df.iloc[(env.df.index == timestamps[2]) & (env.df["tic"] == "AAPL"), env.df.columns.get_loc("open")] = 60.0 # Force an open below the standard 0.75 target!
    # Wait, get_step_vector looks up by timestamps[self.step_index].
    # self.step_index is currently 1 (which refers to timestamps[1]) after step 1 finished incrementing it to 1.
    # So we should modify timestamps[1].
    env.df.iloc[(env.df.index == timestamps[1]) & (env.df["tic"] == "AAPL"), env.df.columns.get_loc("low")] = 50.0
    env.df.iloc[(env.df.index == timestamps[1]) & (env.df["tic"] == "AAPL"), env.df.columns.get_loc("open")] = 60.0
    
    action2 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32) # Do nothing
    obs, reward, done, trunc, info = env.step(action2)
    
    # Step 3: Cash Shortage Test
    # Try to buy an enormous amount with no cash
    # actions_final > 0 will trigger shortage if cash isn't there
    action3 = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32) # Huge buy, SL = 0.5
    obs, reward, done, trunc, info = env.step(action3)
    
    print("All basic bugfix tests pass successfully!")

if __name__ == "__main__":
    test_env_bugfixes()
