import os
import pandas as pd
import tempfile
import tempfile
from src.run_pipeline import build_commands, load_local_history

def test_build_commands_with_local_history():
    combo = {
        "train_start": "2026-02-06",
        "train_end": "2026-02-13",
        "test_start": "2026-02-13",
        "test_end": "2026-02-14",
        "tickers": ["AAPL", "MSFT"],
        "total_timesteps": 10000,
        "reward_weight_pnl": 1.0,
        "reward_weight_drawdown": 0.5,
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "gamma": 0.99,
        "episode_length": 1000,
    }
    cmds = build_commands(combo, "/tmp/test", use_local_history=True, local_history_dir="./history")
    
    # 0th command should fetch train data from local
    train_cmd = " ".join(cmds[0])
    assert "src.run_pipeline" in train_cmd
    assert "local_load" in train_cmd
    assert "AAPL,MSFT" in train_cmd
    assert "2026-02-06" in train_cmd
    assert "2026-02-13" in train_cmd
    assert "./history" in train_cmd

    # 2nd command should fetch test data from local
    test_cmd = " ".join(cmds[2])
    assert "src.run_pipeline" in test_cmd
    assert "local_load" in test_cmd
    assert "AAPL,MSFT" in test_cmd
    assert "2026-02-13" in test_cmd
    assert "2026-02-14" in test_cmd

def test_load_local_history():
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        # Create dummy csv files
        df1 = pd.DataFrame({"date": ["2026-02-06 09:30:00-04:00"], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1], "tic": ["aapl"]})
        df1.to_csv(os.path.join(data_dir, "AAPL_2026-02-06_1m.csv"), index=False)
        
        df2 = pd.DataFrame({"date": ["2026-02-07 09:30:00-04:00"], "open": [2], "high": [2], "low": [2], "close": [2], "volume": [2], "tic": ["aapl"]})
        df2.to_csv(os.path.join(data_dir, "AAPL_2026-02-07_1m.csv"), index=False)
        
        output_path = os.path.join(out_dir, "out.csv")
        load_local_history(["AAPL"], "2026-02-06", "2026-02-08", data_dir, output_path)
        
        res = pd.read_csv(output_path)
        assert len(res) == 2
        assert res["close"].iloc[0] == 1
        assert res["close"].iloc[1] == 2
