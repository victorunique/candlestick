"""Tests for generate_report module."""

import os
import pandas as pd
import pytest

from src.generate_report import generate_report

@pytest.fixture
def synthetic_csvs(tmp_path):
    dates = pd.date_range("2024-01-02", periods=5, freq="B", tz="US/Eastern")
    
    # account
    acct_rows = []
    for i, d in enumerate(dates):
        acct_rows.append({
            "timestamp": d, "cash": 900_000, "asset_value": 100_000,
            "total_assets": 1_000_000 + i * 5_000, "reward": 0.01 * i,
        })
    df_acct = pd.DataFrame(acct_rows)
    acct_path = str(tmp_path / "account_history.csv")
    df_acct.to_csv(acct_path, index=False)

    # action
    act_rows = []
    for i, d in enumerate(dates):
        act_rows.append({
            "timestamp": d, "actions": "[0 0]", "transactions": "[10 -5]"
        })
    df_act = pd.DataFrame(act_rows)
    act_path = str(tmp_path / "action_history.csv")
    df_act.to_csv(act_path, index=False)

    # data
    data_rows = []
    for d in dates:
        for tic, base in [("AAPL", 150.0), ("MSFT", 380.0)]:
            data_rows.append({
                "date": d, "tic": tic, "open": base, "high": base + 3,
                "low": base - 3, "close": base, "volume": 1_000_000
            })
    df_data = pd.DataFrame(data_rows)
    data_path = str(tmp_path / "data.csv")
    df_data.to_csv(data_path, index=False)
    
    # log
    log_rows = [{"timesteps": i * 100, "ep_rew_mean": i * 0.5, "policy_loss": 0.1, "value_loss": 0.1, "entropy_loss": 0.1, "approx_kl": 0.1} for i in range(1, 6)]
    df_log = pd.DataFrame(log_rows)
    log_path = str(tmp_path / "training_log.csv")
    df_log.to_csv(log_path, index=False)

    # baseline
    bl_rows = [{"date": d, "total_assets": 1_000_000 + i * 2_000} for i, d in enumerate(dates)]
    df_bl = pd.DataFrame(bl_rows)
    bl_path = str(tmp_path / "baseline.csv")
    df_bl.to_csv(bl_path, index=False)

    return acct_path, act_path, data_path, log_path, bl_path

def test_generate_report_creates_html(synthetic_csvs, tmp_path):
    acct_path, act_path, data_path, log_path, baseline_path = synthetic_csvs
    out_path = str(tmp_path / "report.html")

    class Args:
        def __init__(self, account_path, action_path, log_path, train_data_path, test_data_path, baseline_path, fixed_sl_path, output_path):
            self.account_path = account_path
            self.action_path = action_path
            self.log_path = log_path
            self.train_data_path = train_data_path
            self.test_data_path = test_data_path
            self.train_initial_capital = 1000000.0
            self.baseline_path = baseline_path
            self.fixed_sl_path = fixed_sl_path
            self.fixed_stoploss_ratio = 0.95
            self.output_path = output_path
            self.test_start = "2024-01-04"

    args = Args(acct_path, act_path, log_path, data_path, data_path, baseline_path, acct_path, out_path)
    generate_report(args)

    assert os.path.exists(out_path)
    with open(out_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "Strategy Evaluation Report" in html
    assert "AAPL, MSFT" in html
    assert '<img src="data:image/png;base64,' in html
    
    # Check that it contains 6 images (3 for backtest + 3 for training)
    assert html.count('data:image/png;base64,') == 6
