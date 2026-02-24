from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.trading_env import TradingEnv
from src.pipeline import RemotePipeline
import redis
import json
from loguru import logger
import os

class RedisCallback(BaseCallback):
    def __init__(self, redis_client):
        super().__init__()
        self.r = redis_client
        
    def _on_step(self) -> bool:
        if "rewards" in getattr(self, "locals", {}):
            rwd = float(self.locals["rewards"][0])
        else:
            rwd = 0.0
            
        metrics = {
            "step": self.num_timesteps,
            "reward": rwd
        }
        try:
            self.r.publish("training_metrics", json.dumps(metrics))
        except BaseException:
            pass
        return True

def main():
    logger.info("Starting RL Training Service")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    r = redis.Redis.from_url(redis_url)
    
    pipeline = RemotePipeline("BTC/USD", 
        data_addr=os.environ.get("DATA_ADDR", "localhost:50051"),
        chart_addr=os.environ.get("CHART_ADDR", "localhost:50052"),
        vision_addr=os.environ.get("VISION_ADDR", "localhost:50053")
    )
    
    env = TradingEnv(pipeline, max_steps=100)
    
    # Using PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    
    logger.info("Starting PPO Training loop...")
    try:
        # Using 10 timesteps for the sake of checking without hanging
        model.learn(total_timesteps=10, callback=RedisCallback(r))
        model.save("ppo_trading_policy")
        logger.info("Model saved.")
    except Exception as e:
        logger.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()
