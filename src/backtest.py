import os
import pandas as pd
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from src.env_stocktrading import StockTradingEnv

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
    trained_ppo = PPO.load(model_path)
    
    print("Running backtest...")
    obs = e_trade_normalized.reset()
    max_steps = len(df.index.unique()) - 1
    
    for i in range(len(df.index.unique())):
        action, _states = trained_ppo.predict(obs, deterministic=True)
        obs, rewards, dones, info = e_trade_normalized.step(action)
        
        if i >= max_steps or dones[0]:
            print("Hit end of data!")
            break
            
    # Access inner unwrapped environment for state logs
    actual_env = e_trade_normalized.venv.venv.envs[0]
    df_account_value = actual_env.save_asset_memory()
    df_actions = actual_env.save_action_memory()
    
    # Calculate simple performance
    final_value = df_account_value.iloc[-1]['total_assets']
    initial_value = env_test_kwargs['initial_amount']
    return_pct = ((final_value - initial_value) / initial_value) * 100

    print("\n--- Backtest Layout ---")
    print(f"Initial Portfolio Value: {initial_value}")
    print(f"Final Portfolio Value:   {final_value:.2f}")
    print(f"Total Return:            {return_pct:.2f}%")
    
    model_name = os.path.basename(model_path)
    df_account_value.to_csv(os.path.join(results_dir, f"{model_name}_account_history.csv"), index=False)
    df_actions.to_csv(os.path.join(results_dir, f"{model_name}_action_history.csv"), index=False)
    
    return df_account_value, df_actions

def main():
    parser = argparse.ArgumentParser(description="Backtest a trained PPO agent.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed CSV backtest data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (without .zip)")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save backtest CSVs")
    parser.add_argument("--indicators", type=str, nargs="+", default=["macd", "rsi_30", "cci_30", "dx_30"], help="List of indicators used in data")
    parser.add_argument("--window_size", type=int, default=60, help="CNN1D Window size")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file {args.data_path} not found.")
        
    df = pd.read_csv(args.data_path)
    
    backtest(
        df=df,
        model_path=args.model_path,
        results_dir=args.results_dir,
        indicators=args.indicators,
        window_size=args.window_size
    )
    
    print(f"Backtest full results saved to {args.results_dir}")

if __name__ == "__main__":
    main()
