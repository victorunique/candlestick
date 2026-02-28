import os
import numpy as np
import pandas as pd
import argparse

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from src.env_stocktrading import StockTradingEnv
from src.feature_engineer import INDICATORS

def baseline_buy_and_hold(
    df: pd.DataFrame,
    initial_amount: float = 1_000_000,
    buy_cost_pct: float = 0.0001,
) -> pd.DataFrame:
    """Equal-weight buy-and-hold baseline over the same data period.

    Allocates *initial_amount* equally across all tickers on the first
    date (deducting transaction costs), then holds until the end.
    Returns a DataFrame with columns ["date", "total_assets"].
    """
    tickers = sorted(df["tic"].unique())
    dates = sorted(df["date"].unique())
    n_tickers = len(tickers)

    # Build a date×ticker close-price matrix
    price_df = df.pivot_table(index="date", columns="tic", values="close")
    price_df = price_df[tickers].loc[dates]

    # Day-0 purchase: split cash equally, buy at first close
    alloc_per_ticker = initial_amount / n_tickers
    first_closes = price_df.iloc[0].values
    # Shares bought (fractional) after deducting cost
    shares = (alloc_per_ticker * (1 - buy_cost_pct)) / first_closes
    leftover_cash = initial_amount - np.sum(shares * first_closes) \
        - np.sum(shares * first_closes) * buy_cost_pct
    # Simplify: leftover is the rounding dust from cost deduction
    leftover_cash = initial_amount - np.sum(shares * first_closes * (1 + buy_cost_pct))

    # Daily portfolio value
    daily_values = price_df.values @ shares + leftover_cash

    return pd.DataFrame({"date": dates, "total_assets": daily_values})


def backtest(
    df: pd.DataFrame,
    model_path: str,
    results_dir: str,
    indicators: list,
    window_size: int = 60,
    hmax: int = 100000,
    stoploss_penalty: float = 0.9,
    profit_loss_ratio: float = 1.5,
    cash_penalty: float = 0.05
):
    os.makedirs(results_dir, exist_ok=True)
    
    # Needs matching parameters to training env shape
    env_test_kwargs = {
        "hmax": hmax,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.0001,
        "sell_cost_pct": 0.0001,
        "print_verbosity": 500,
        "discrete_actions": True,
        "feature_columns": ["open", "close", "high", "low", "volume"] + indicators,
        "stoploss_penalty": stoploss_penalty,
        "profit_loss_ratio": profit_loss_ratio,
        "cash_penalty_proportion": cash_penalty,
        "patient": True,
        "episode_length": -1,  # Run to end
        "random_start": False  # Start at beginning
    }
    
    e_trade_gym = DummyVecEnv([lambda: StockTradingEnv(df=df, **env_test_kwargs)])
    e_trade_stacked = VecFrameStack(e_trade_gym, n_stack=window_size)
    
    vec_normalize_path = f"{model_path}_vecnormalize.pkl"
    if not os.path.exists(vec_normalize_path):
        raise FileNotFoundError(f"Normalization statistics not found at {vec_normalize_path}")
        
    e_trade_normalized = VecNormalize.load(vec_normalize_path, e_trade_stacked)
    e_trade_normalized.training = False
    e_trade_normalized.norm_reward = False
    
    print(f"Loading model from {model_path}.zip...")
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    trained_ppo = PPO.load(model_path, device=device)
    
    print("Running backtest...")
    obs = e_trade_normalized.reset()
    max_steps = len(df["date"].unique()) - 1
    
    for i in range(len(df["date"].unique())):
        action, _states = trained_ppo.predict(obs, deterministic=True)
        obs, rewards, dones, info = e_trade_normalized.step(action)
        
        if i >= max_steps or dones[0]:
            print("Hit end of data!")
            break
            
    # Access inner unwrapped environment for state logs
    actual_env = e_trade_normalized.venv.venv.envs[0]
    df_account_value = actual_env.save_asset_memory()
    df_actions = actual_env.save_action_memory()

    if df_account_value is None or df_actions is None:
        raise RuntimeError(
            "Backtest produced no results. The data may be too short for the given window_size."
        )
    
    # Calculate simple performance
    final_value = df_account_value.iloc[-1]['total_assets']
    initial_value = env_test_kwargs['initial_amount']
    return_pct = ((final_value - initial_value) / initial_value) * 100

    # Calculate max drawdown
    portfolio_values = np.array(df_account_value['total_assets'])
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    max_drawdown = drawdowns.min() * 100  # as percentage

    # --- Buy-and-Hold baseline ---
    df_baseline = baseline_buy_and_hold(
        df, initial_amount=initial_value, buy_cost_pct=env_test_kwargs['buy_cost_pct']
    )
    bl_final = df_baseline["total_assets"].iloc[-1]
    bl_return = ((bl_final - initial_value) / initial_value) * 100
    bl_values = np.array(df_baseline["total_assets"])
    bl_running_max = np.maximum.accumulate(bl_values)
    bl_drawdowns = (bl_values - bl_running_max) / bl_running_max
    bl_max_dd = bl_drawdowns.min() * 100

    print("\n--- Backtest Results ---")
    print(f"Initial Portfolio Value:  {initial_value}")
    print("")
    print("  PPO Agent")
    print(f"  Final Portfolio Value:  {final_value:.2f}")
    print(f"  Total Return:           {return_pct:.2f}%")
    print(f"  Max Drawdown:           {max_drawdown:.2f}%")
    print("")
    print("  Buy & Hold Baseline")
    print(f"  Final Portfolio Value:  {bl_final:.2f}")
    print(f"  Total Return:           {bl_return:.2f}%")
    print(f"  Max Drawdown:           {bl_max_dd:.2f}%")
    
    model_name = os.path.basename(model_path)
    df_account_value.to_csv(os.path.join(results_dir, f"{model_name}_account_history.csv"), index=False)
    df_actions.to_csv(os.path.join(results_dir, f"{model_name}_action_history.csv"), index=False)
    df_baseline.to_csv(os.path.join(results_dir, "baseline_buy_and_hold_account_history.csv"), index=False)
    
    return df_account_value, df_actions, df_baseline

def main():
    parser = argparse.ArgumentParser(description="Backtest a trained PPO agent.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed CSV backtest data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (without .zip)")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save backtest CSVs")
    parser.add_argument("--indicators", type=str, nargs="+", default=INDICATORS, help="List of indicators used in data")
    parser.add_argument("--window_size", type=int, default=60, help="CNN1D Window size")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file {args.data_path} not found.")
        
    df = pd.read_csv(args.data_path)
    
    _account, _actions, _baseline = backtest(
        df=df,
        model_path=args.model_path,
        results_dir=args.results_dir,
        indicators=args.indicators,
        window_size=args.window_size
    )
    
    print(f"Backtest full results saved to {args.results_dir}")

if __name__ == "__main__":
    main()
