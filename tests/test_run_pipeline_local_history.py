from src.run_pipeline import build_commands

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
        "cash_penalty_proportion": 0.05,
        "upside_pnl_multiplier": 1.0,
        "stoploss_min": 0.90,
        "stoploss_max": 0.95,
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "gamma": 0.99,
        "episode_length": 1000,
    }
    cmds = build_commands(combo, "/tmp/test", use_local_history=True, local_history_dir="./history")
    
    # 0th command should fetch train data from local using data_loader
    train_cmd = " ".join(cmds[0])
    assert "src.data_loader" in train_cmd
    assert "--start_date 2026-02-01" in train_cmd
    assert "--end_date 2026-02-13" in train_cmd
    assert "--ticker_list AAPL MSFT" in train_cmd
    assert "--data_dir ./history" in train_cmd

    # 2nd command should fetch test data from local using data_loader
    test_cmd = " ".join(cmds[2])
    assert "src.data_loader" in test_cmd
    assert "--start_date 2026-02-08" in test_cmd
    assert "--end_date 2026-02-14" in test_cmd
    assert "--ticker_list AAPL MSFT" in test_cmd
    assert "--data_dir ./history" in test_cmd
