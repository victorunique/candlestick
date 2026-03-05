import os
import pandas as pd
import argparse
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from src.env_stocktrading import StockTradingEnv
from src.custom_models import CNN1DFeaturesExtractor
from src.feature_engineer import INDICATORS


class RewardLoggingCallback(BaseCallback):
    """Logs per-rollout training metrics to a CSV for learning-curve visualization."""

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.rows: list[dict] = []

    def _on_rollout_end(self) -> None:
        logger = self.model.logger.name_to_value
        
        # Extract episode rewards directly from the model's info buffer
        ep_info_buffer = self.model.ep_info_buffer
        if ep_info_buffer and len(ep_info_buffer) > 0:
            ep_rew_mean = np.mean([ep_info["r"] for ep_info in ep_info_buffer])
            ep_len_mean = np.mean([ep_info["l"] for ep_info in ep_info_buffer])
        else:
            ep_rew_mean = float("nan")
            ep_len_mean = float("nan")
            
        self.rows.append({
            "timesteps": self.num_timesteps,
            "ep_rew_mean": ep_rew_mean,
            "ep_len_mean": ep_len_mean,
            "policy_loss": logger.get("train/policy_gradient_loss", float("nan")),
            "value_loss": logger.get("train/value_loss", float("nan")),
            "entropy_loss": logger.get("train/entropy_loss", float("nan")),
            "approx_kl": logger.get("train/approx_kl", float("nan")),
            "clip_fraction": logger.get("train/clip_fraction", float("nan")),
        })

    def _on_training_end(self) -> None:
        pd.DataFrame(self.rows).to_csv(self.log_path, index=False)
        if self.verbose:
            print(f"Training log saved to {self.log_path}")

    def _on_step(self) -> bool:
        return True


def train_ppo(
    df: pd.DataFrame, 
    total_timesteps: int, 
    model_dir: str, 
    model_name: str,
    indicators: list,
    window_size: int = 60,
    n_steps: int = 2048,
    ent_coef: float = 0.01,
    learning_rate: float = 0.00025,
    gamma: float = 0.99,
    hmax: int = 100000,
    stoploss_penalty: float = 0.9,
    profit_loss_ratio: float = 1.5,
    cash_penalty: float = 0.05,
    reward_weight_pnl: float = 1.0,
    reward_weight_drawdown: float = 0.5,
    incremental_drawdown_penalty: bool = True,
    episode_length: int = 1000,
    seed: int = None
):
    os.makedirs(model_dir, exist_ok=True)
    
    env_train_kwargs = {
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
        "reward_weight_pnl": reward_weight_pnl,
        "reward_weight_drawdown": reward_weight_drawdown,
        "incremental_drawdown_penalty": incremental_drawdown_penalty,
        "patient": True,
        "episode_length": episode_length,
        "random_start": True
    }
    
    e_train_gym = DummyVecEnv([lambda: Monitor(StockTradingEnv(df=df, **env_train_kwargs))])
    e_train_stacked = VecFrameStack(e_train_gym, n_stack=window_size)
    e_train_normalized = VecNormalize(e_train_stacked, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal Performance Shaders) for training.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for training.")
    else:
        device = "cpu"
        print("Using CPU for training.")
    
    PPO_PARAMS = {
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        "learning_rate": learning_rate,
        "batch_size": 128,
        "gamma": gamma,
        "device": device,
        "seed": seed,
    }
    
    POLICY_KWARGS = {
        "features_extractor_class": CNN1DFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128, "n_stack": window_size},
    }
    
    model = PPO(
        "MlpPolicy",
        env=e_train_normalized,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        **PPO_PARAMS
    )
    
    log_path = os.path.join(model_dir, f"{model_name}_training_log.csv")
    reward_callback = RewardLoggingCallback(log_path=log_path, verbose=1)
    
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="ppo_stock_trading",
        callback=reward_callback,
    )
    
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
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent with StockTradingEnv")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed CSV data")
    parser.add_argument("--model_dir", type=str, default="./trained_models", help="Directory to save the trained model")
    parser.add_argument("--model_name", type=str, default="ppo_trading_agent", help="Name of the saved model")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total training timesteps")
    parser.add_argument("--indicators", type=str, nargs="+", default=INDICATORS, help="List of indicators used in data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Train Hyperparameters
    parser.add_argument("--window_size", type=int, default=60, help="CNN1D Window size")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO rollout buffer size per update")
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--episode_length", type=int, default=1000)
    
    # Environment Hyperparameters
    parser.add_argument("--hmax", type=int, default=100000, help="Max number of shares to trade")
    parser.add_argument("--stoploss_penalty", type=float, default=0.9, help="Stop-loss penalty ratio")
    parser.add_argument("--profit_loss_ratio", type=float, default=1.5, help="Profit-to-loss ratio")
    parser.add_argument("--cash_penalty", type=float, default=0.05, help="Cash penalty proportion")
    parser.add_argument("--reward_weight_pnl", type=float, default=1.0, help="Reward weight for PnL")
    parser.add_argument("--reward_weight_drawdown", type=float, default=0.5, help="Reward weight for drawdown penalty")
    parser.add_argument("--continuous_drawdown_penalty", action="store_true", help="Use continuous instead of incremental drawdown penalty")
    
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
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        episode_length=args.episode_length,
        hmax=args.hmax,
        stoploss_penalty=args.stoploss_penalty,
        profit_loss_ratio=args.profit_loss_ratio,
        cash_penalty=args.cash_penalty,
        reward_weight_pnl=args.reward_weight_pnl,
        reward_weight_drawdown=args.reward_weight_drawdown,
        incremental_drawdown_penalty=not args.continuous_drawdown_penalty,
        seed=args.seed
    )
    
    print(f"Training finished! Model saved to {os.path.join(args.model_dir, args.model_name)}")

if __name__ == "__main__":
    main()
