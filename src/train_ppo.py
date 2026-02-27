import os
import pandas as pd
import argparse
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

from src.env_stocktrading_minute import StockTradingEnvMinute
from src.custom_models import CNN1DFeaturesExtractor

def train_ppo(
    df: pd.DataFrame, 
    total_timesteps: int, 
    model_dir: str, 
    model_name: str,
    indicators: list,
    window_size: int = 60,
    ent_coef: float = 0.01,
    learning_rate: float = 0.00025,
    gamma: float = 0.99,
    hmax: int = 100000,
    stoploss_penalty: float = 0.9,
    profit_loss_ratio: float = 1.5,
    cash_penalty: float = 0.05,
    episode_length: int = 1000
):
    os.makedirs(model_dir, exist_ok=True)
    
    env_train_kwargs = {
        "hmax": hmax,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.0001,
        "sell_cost_pct": 0.0001,
        "print_verbosity": 500,
        "discrete_actions": True,
        "daily_information_cols": ["open", "close", "high", "low", "volume"] + indicators,
        "stoploss_penalty": stoploss_penalty,
        "profit_loss_ratio": profit_loss_ratio,
        "cash_penalty_proportion": cash_penalty,
        "patient": True,
        "episode_length": episode_length,
        "random_start": True
    }
    
    e_train_gym = DummyVecEnv([lambda: StockTradingEnvMinute(df=df, **env_train_kwargs)])
    e_train_normalized = VecNormalize(e_train_gym, norm_obs=True, norm_reward=True, clip_obs=10.0)
    e_train_stacked = VecFrameStack(e_train_normalized, n_stack=window_size)
    
    device = "cpu"  # Force CPU as PPO with small networks is usually faster on CPU
    
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "batch_size": 128,
        "gamma": gamma,
        "device": device,
    }
    
    POLICY_KWARGS = {
        "features_extractor_class": CNN1DFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128, "n_stack": window_size},
    }
    
    model = PPO(
        "MlpPolicy",
        env=e_train_stacked,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        **PPO_PARAMS
    )
    
    model.learn(total_timesteps=total_timesteps, tb_log_name="ppo_stock_trading")
    
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    e_train_normalized.save(f"{model_path}_vecnormalize.pkl")
    
    return model, e_train_normalized

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent with StockTradingEnvMinute")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed CSV data")
    parser.add_argument("--model_dir", type=str, default="./trained_models", help="Directory to save the trained model")
    parser.add_argument("--model_name", type=str, default="ppo_trading_agent", help="Name of the saved model")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total training timesteps")
    parser.add_argument("--indicators", type=str, nargs="+", default=["macd", "rsi_30", "cci_30", "dx_30"], help="List of indicators used in data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Train Hyperparameters
    parser.add_argument("--window_size", type=int, default=60, help="CNN1D Window size")
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--episode_length", type=int, default=1000)
    
    args = parser.parse_args()
    
    set_seeds(args.seed)
    
    print(f"Loading data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file {args.data_path} not found.")
        
    df = pd.read_csv(args.data_path)
    
    print(f"Starting training for {args.total_timesteps} timesteps...")
    
    model, env_normalized = train_ppo(
        df=df,
        total_timesteps=args.total_timesteps,
        model_dir=args.model_dir,
        model_name=args.model_name,
        indicators=args.indicators,
        window_size=args.window_size,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        episode_length=args.episode_length
    )
    
    print(f"Training finished! Model saved to {os.path.join(args.model_dir, args.model_name)}")

if __name__ == "__main__":
    main()
