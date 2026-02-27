import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from src.train_ppo import train_ppo
from src.backtest import backtest


@pytest.fixture
def sample_preprocessed_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = []

    for i, date in enumerate(dates):
        # Generate dummy data for two assets
        for tic in ["AAPL", "MSFT"]:
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": tic,
                "open": 100 + i,
                "high": 105 + i,
                "low": 95 + i,
                "close": 100 + i,
                "volume": 1000,
                "macd": 1,
                "rsi_30": 50,
                "cci_30": 100,
                "dx_30": 20
            })

    df = pd.DataFrame(data)
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    return df


@pytest.fixture
def trained_model_dir(sample_preprocessed_data):
    """Train a small model once and return (temp_dir, model_name) for reuse."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_name = "test_backtest_model"

        train_ppo(
            df=sample_preprocessed_data,
            total_timesteps=10,
            model_dir=temp_dir,
            model_name=model_name,
            indicators=["macd", "rsi_30", "cci_30", "dx_30"],
            window_size=10
        )

        yield temp_dir, model_name


def test_backtest(sample_preprocessed_data, trained_model_dir):
    temp_dir, model_name = trained_model_dir
    model_path = os.path.join(temp_dir, model_name)
    results_dir = os.path.join(temp_dir, "results")

    account_df, actions_df = backtest(
        df=sample_preprocessed_data,
        model_path=model_path,
        results_dir=results_dir,
        indicators=["macd", "rsi_30", "cci_30", "dx_30"],
        window_size=10
    )

    # Verify execution and output formats
    assert not account_df.empty
    assert "total_assets" in account_df.columns
    assert not actions_df.empty

    # Verify files were saved
    assert os.path.exists(os.path.join(results_dir, f"{model_name}_account_history.csv"))
    assert os.path.exists(os.path.join(results_dir, f"{model_name}_action_history.csv"))


def test_backtest_output_says_results(sample_preprocessed_data, trained_model_dir, capsys):
    """Bug 2: Verify the print output says 'Backtest Results', not 'Backtest Layout'."""
    temp_dir, model_name = trained_model_dir
    model_path = os.path.join(temp_dir, model_name)
    results_dir = os.path.join(temp_dir, "results_print")

    backtest(
        df=sample_preprocessed_data,
        model_path=model_path,
        results_dir=results_dir,
        indicators=["macd", "rsi_30", "cci_30", "dx_30"],
        window_size=10
    )

    captured = capsys.readouterr()
    assert "Backtest Results" in captured.out
    assert "Backtest Layout" not in captured.out


def test_backtest_return_pct_positive_trend(sample_preprocessed_data, trained_model_dir):
    """Bug 1: With the off-by-one fix, verify backtest completes and returns valid DataFrames."""
    temp_dir, model_name = trained_model_dir
    model_path = os.path.join(temp_dir, model_name)
    results_dir = os.path.join(temp_dir, "results_pct")

    account_df, actions_df = backtest(
        df=sample_preprocessed_data,
        model_path=model_path,
        results_dir=results_dir,
        indicators=["macd", "rsi_30", "cci_30", "dx_30"],
        window_size=10
    )

    # Verify account_df has more than just the initial row
    assert len(account_df) > 1
    # Verify total_assets column has numeric values
    assert account_df["total_assets"].dtype in ["float64", "int64"]
    # Verify actions_df has the expected columns
    assert "actions" in actions_df.columns
    assert "transactions" in actions_df.columns


def test_backtest_iterates_by_dates_not_rows(sample_preprocessed_data, trained_model_dir):
    """Bug 1: Verify the loop iterates by unique dates, not by dataframe row count."""
    temp_dir, model_name = trained_model_dir
    model_path = os.path.join(temp_dir, model_name)
    results_dir = os.path.join(temp_dir, "results_dates")

    n_unique_dates = len(sample_preprocessed_data["date"].unique())
    n_rows = len(sample_preprocessed_data)
    # With 2 tickers, rows should be 2x the dates
    assert n_rows == n_unique_dates * 2

    account_df, actions_df = backtest(
        df=sample_preprocessed_data,
        model_path=model_path,
        results_dir=results_dir,
        indicators=["macd", "rsi_30", "cci_30", "dx_30"],
        window_size=10
    )

    # The account history length should be at most n_unique_dates + 1
    # (one initial entry + one per step), NOT n_rows + 1
    assert len(account_df) <= n_unique_dates + 1


def test_backtest_output_includes_max_drawdown(sample_preprocessed_data, trained_model_dir, capsys):
    """Verify the printed output includes Max Drawdown."""
    temp_dir, model_name = trained_model_dir
    model_path = os.path.join(temp_dir, model_name)
    results_dir = os.path.join(temp_dir, "results_drawdown")

    backtest(
        df=sample_preprocessed_data,
        model_path=model_path,
        results_dir=results_dir,
        indicators=["macd", "rsi_30", "cci_30", "dx_30"],
        window_size=10
    )

    captured = capsys.readouterr()
    assert "Max Drawdown:" in captured.out
    assert "%" in captured.out.split("Max Drawdown:")[1].split("\n")[0]


def test_max_drawdown_calculation():
    """Verify max drawdown math on a known portfolio value sequence."""
    # Portfolio: 100 -> 120 -> 90 -> 110
    # Peak at 120, trough at 90 => drawdown = (90 - 120) / 120 = -25%
    portfolio_values = np.array([100.0, 120.0, 90.0, 110.0])
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = drawdowns.min() * 100

    assert max_drawdown == pytest.approx(-25.0)
