import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, pipeline, max_steps=1000):
        super(TradingEnv, self).__init__()
        self.pipeline = pipeline
        self.max_steps = max_steps
        self.current_step = 0
        
        # Action space: 0 (SAME), 1 (DOWN), 2 (UP) # Directional classification
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.pipeline.get_features(self.current_step)
        return obs, {}

    def step(self, action):
        current_price = self.pipeline.prices[self.current_step]
        future_price = self.pipeline.get_future_close(self.current_step)
        
        if future_price > current_price:
            correct_action = 2 
        elif future_price < current_price:
            correct_action = 1 
        else:
            correct_action = 0 
            
        reward = 1.0 if action == correct_action else -1.0
        
        self.current_step += 1
        
        # If we hit max_steps or ran out of future predictions
        terminated = self.current_step >= self.max_steps - 1
        
        if not terminated:
            obs = self.pipeline.get_features(self.current_step)
        else:
            obs = np.zeros(256, dtype=np.float32)
            
        return obs, reward, terminated, False, {}
