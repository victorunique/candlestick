from __future__ import annotations
import random
import time
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    """
    A multi-timeframe trading environment (supports 1m, 5m, 1h, 1d, etc.).
    It focuses on immediate PnL (Profit and Loss) rather than long-term accumulated average return.
    The timestamp column should contain ISO-format timestamps at whatever interval the data uses.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        buy_cost_pct=3e-3,
        sell_cost_pct=3e-3,
        timestamp_col_name="date",
        hmax=10,
        discrete_actions=False,
        shares_increment=1,
        stoploss_penalty=0.9,
        profit_loss_ratio=2,
        print_verbosity=10,
        initial_amount=1e6,
        feature_columns=["open", "close", "high", "low", "volume"],
        cache_indicator_data=True,
        cash_penalty_proportion=0.1,
        random_start=True,
        patient=False,
        currency="$",
        episode_length=-1,
        window_size=1,
        warmup_steps=0,
        reward_weight_pnl=1.0,
        reward_weight_drawdown=0.5,
        incremental_drawdown_penalty=True,
    ):
        self.df = df
        self.incremental_drawdown_penalty = incremental_drawdown_penalty
        self.stock_col = "tic"
        self.assets = sorted(df[self.stock_col].unique())
        
        all_timestamps = df[timestamp_col_name].sort_values().unique()
        self.warmup_steps = warmup_steps
        if self.warmup_steps > 0:
            self.timestamps = all_timestamps[self.warmup_steps:]
        else:
            self.timestamps = all_timestamps
            
        self.random_start = random_start
        self.episode_length = episode_length
        self.window_size = window_size
        self.discrete_actions = discrete_actions
        self.patient = patient
        self.currency = currency
        self.df = self.df.set_index(timestamp_col_name)
        self.shares_increment = shares_increment
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.stoploss_penalty = stoploss_penalty
        # NOTE: min_profit_penalty is NOT currently used in the reward function.
        # It is retained for potential future reward shaping, e.g. penalising
        # the agent for closing trades that do not meet a minimum
        # risk-to-reward ratio relative to the stop-loss distance.
        self.min_profit_penalty = 1 + profit_loss_ratio * (1 - self.stoploss_penalty)
        self.feature_columns = feature_columns
        self.state_space = (
            1 + len(self.assets) + len(self.assets) * len(self.feature_columns)
        )
        
        # Action Space: composite of trading actions and stoploss ratios
        # Flattened for PPO compatibility: [Trading Actions (n) | Stoploss Ratios (n)]
        n_assets = len(self.assets)
        action_dim = 2 * n_assets
        
        # Define bounds
        # Trading actions: [-1, 1]
        # Stoploss ratios: [0.5, 1.0]
        # Note: SB3 will automatically rescale actions to these bounds
        low = np.concatenate([np.full(n_assets, -1.0, dtype=np.float32), np.full(n_assets, 0.5, dtype=np.float32)])
        high = np.concatenate([np.full(n_assets, 1.0, dtype=np.float32), np.full(n_assets, 1.0, dtype=np.float32)])
        
        self.action_space = spaces.Box(low=low, high=high, shape=(action_dim,), dtype=np.float32)
        
        if self.window_size > 1:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.window_size, self.state_space), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
            )

        self.episode = -1 
        self.episode_history = []
        self.printed_header = False
        self.cache_indicator_data = cache_indicator_data
        self.cached_data = None
        # NOTE: cash_penalty_proportion is NOT currently used in the reward
        # function. It is retained for potential future reward shaping, e.g.
        # penalising the agent for holding an excessive proportion of the
        # portfolio in idle cash instead of deploying it into positions.
        self.cash_penalty_proportion = cash_penalty_proportion
        self.reward_weight_pnl = reward_weight_pnl
        self.reward_weight_drawdown = reward_weight_drawdown
        
        if self.cache_indicator_data:
            # print("caching data...")
            temp_df = self.df.reset_index()
            pivot_df = temp_df.pivot(index='date', columns='tic', values=self.feature_columns)
            pivot_df.columns = pivot_df.columns.swaplevel(0, 1)
            
            expected_cols = pd.MultiIndex.from_product(
                [self.assets, self.feature_columns],
                names=['tic', 'feature']
            )
            pivot_df = pivot_df.reindex(columns=expected_cols)

            # Note: cached_data stores ALL timestamps including warmup.
            n_steps_all = len(all_timestamps)
            n_assets = len(self.assets)
            n_features = len(self.feature_columns)
            
            self.cached_data = pivot_df.values.reshape(n_steps_all, n_assets, n_features)
            self.col_map = {col: i for i, col in enumerate(self.feature_columns)}
            # print(f"data cached! Shape: {self.cached_data.shape}")
            
        self.final_asset_memory = None
        self.final_action_memory = None

    def seed(self, seed=None):
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    @property
    def current_step(self):
        return self.step_index - self.starting_point

    def reset(self, *, seed=None, options=None):
        self.avg_buy_price = np.zeros(len(self.assets))
        if self.random_start:
            if self.episode_length > 0:
                max_start = len(self.timestamps) - self.episode_length
                if max_start <= 0:
                    starting_point = 0
                else:
                    starting_point = random.choice(range(max_start))
            else:
                starting_point = random.choice(range(int(len(self.timestamps) * 0.5)))
            self.starting_point = starting_point
        else:
            self.starting_point = 0
        
        self.step_index = self.starting_point
        self.step_in_episode = 0

        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.stoploss_memory = []
        self.state_memory = []
        self.account_information = {
            "cash": [self.initial_amount],
            "asset_value": [0],
            "total_assets": [self.initial_amount],
            "reward": [0],
        }
        self.peak_total_assets = self.initial_amount
        self.prev_drawdown = 0.0
        init_state = np.array(
            [self.initial_amount]
            + [0] * len(self.assets)
            + self.get_step_vector(self.step_index), dtype=np.float32
        )
        self.state_memory.append(init_state)
        return self.get_full_state(), {}

    def get_full_state(self):
        """Builds a 2D array of the past `window_size` steps. Uses actual historical data for history before starting_point."""
        if self.window_size <= 1:
            return np.array(self.state_memory[-1], dtype=np.float32)
            
        start_idx = self.step_index - self.window_size + 1
        state_vectors = []
        for i in range(start_idx, self.step_index + 1):
            if i < self.starting_point:
                # Actual historical data exists because i is relative to self.timestamps, 
                # meaning i = -1 is the last warmup step.
                actual_i = i + self.warmup_steps
                features = self.get_step_vector_by_absolute_index(max(0, actual_i))
                vec = np.array([self.initial_amount] + [0] * len(self.assets) + features, dtype=np.float32)
                state_vectors.append(vec)
            else:
                mem_idx = i - self.starting_point
                state_vectors.append(np.array(self.state_memory[mem_idx], dtype=np.float32))
                
        return np.vstack(state_vectors)

    def get_step_vector_by_absolute_index(self, abs_step, cols=None):
        if self.cached_data is not None:
            if cols is None:
                return self.cached_data[abs_step].flatten().tolist()
            else:
                col_indices = [self.col_map[c] for c in cols]
                return self.cached_data[abs_step, :, col_indices].flatten().tolist()
        else:
            raise NotImplementedError("Fetching without cache by absolute index not implemented")

    def get_step_vector(self, step, cols=None):
        return self.get_step_vector_by_absolute_index(step + self.warmup_steps, cols)

    def return_terminal(self, reason="Last Date", reward=0):
        state = self.get_full_state()
        self.log_step(reason=reason, terminal_reward=reward)
        self.final_asset_memory = self.save_asset_memory()
        self.final_action_memory = self.save_action_memory()
        return state, reward, True, False, {}

    def log_step(self, reason, terminal_reward=None):
        should_force_print = terminal_reward is not None
        if terminal_reward is None:
            if len(self.account_information["reward"]) > 0:
                terminal_reward = self.account_information["reward"][-1]
            else:
                terminal_reward = 0
        cash_pct = (
            self.account_information["cash"][-1]
            / self.account_information["total_assets"][-1]
        )
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        rec = [
            self.episode,
            self.step_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['total_assets'][-1]))}",
            f"{terminal_reward * 100:0.5f}%",
            f"{(gl_pct - 1) * 100:0.5f}%",
            f"{cash_pct * 100:0.2f}%",
        ]
        self.episode_history.append(rec)
        if (self.current_step + 1) % self.print_verbosity == 0 or should_force_print:
            pass # print(self.template.format(*rec))

    def log_header(self):
        self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}|{7:10}"
        self.printed_header = True

    def step(self, actions):
        begin_cash = self.state_memory[-1][0]
        holdings = np.array(self.state_memory[-1][1:len(self.assets) + 1])

        current_closings = np.array(self.get_step_vector(self.step_index, cols=["close"]))
        prev_total_assets = begin_cash + np.dot(holdings, current_closings)

        if isinstance(actions, dict):
            trading_actions = actions["trading_actions"].astype(np.float32)
            stoploss_ratios = actions["stoploss_ratios"].astype(np.float32)
        else:
            if isinstance(actions, np.ndarray) and actions.shape[0] == 2 * len(self.assets):
                trading_actions = actions[:len(self.assets)].astype(np.float32)
                stoploss_ratios = actions[len(self.assets):].astype(np.float32)
            else:
                trading_actions = actions
                if hasattr(trading_actions, "astype"):
                    trading_actions = trading_actions.astype(np.float32)
                stoploss_ratios = np.ones(len(self.assets), dtype=np.float32) * self.stoploss_penalty

        stoploss_ratios = np.clip(stoploss_ratios, 0.5, 1.0)

        if self.printed_header is False:
            self.log_header()
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")

        if self.step_index == len(self.timestamps) - 1:
            return self.return_terminal(reward=0)

        current_lows = np.array(self.get_step_vector(self.step_index, cols=["low"]))
        sl_thresholds = self.avg_buy_price * stoploss_ratios
        sl_hit_mask = (current_lows < sl_thresholds) & (holdings > 0)
        
        self.stoploss_memory.append(sl_hit_mask.astype(float))

        if np.any(sl_hit_mask):
            self.log_step(reason="STOP LOSS TRIGGERED")

        trading_actions = trading_actions * self.hmax
        self.actions_memory.append(trading_actions)

        if self.discrete_actions:
            trading_actions = np.where(current_closings > 0, trading_actions // current_closings, 0)
            trading_actions = trading_actions.astype(int)
            trading_actions = np.where(
                trading_actions >= 0,
                (trading_actions // self.shares_increment) * self.shares_increment,
                ((trading_actions + self.shares_increment) // self.shares_increment)
                * self.shares_increment,
            )
        else:
            trading_actions = np.where(current_closings > 0, trading_actions / current_closings, 0)

        actions_final = np.where(sl_hit_mask, -np.array(holdings), trading_actions)
        actions_final = np.maximum(actions_final, -np.array(holdings))

        sells = -np.clip(actions_final, -np.inf, 0)
        # Use SL threshold price for stop-loss sells, close price for normal sells
        sell_prices = np.where(sl_hit_mask, sl_thresholds, current_closings)
        proceeds = np.dot(sells, sell_prices)
        costs = proceeds * self.sell_cost_pct
        coh = begin_cash + proceeds

        buys = np.clip(actions_final, 0, np.inf)
        spend = np.dot(buys, current_closings)
        costs += spend * self.buy_cost_pct

        if (spend + costs) > coh:
            if self.patient:
                self.log_step(reason="CASH SHORTAGE")
                actions_final = np.where(actions_final > 0, 0, actions_final)
                spend = 0
                sells = -np.clip(actions_final, -np.inf, 0)
                proceeds = np.dot(sells, current_closings)
                costs = proceeds * self.sell_cost_pct
                coh = begin_cash + proceeds
            else:
                self.transaction_memory.append(actions_final)
                return self.return_terminal(reason="CASH SHORTAGE", reward=0)

        self.transaction_memory.append(actions_final)
        
        coh = coh - spend - costs
        holdings_updated = holdings + actions_final

        buys_mask = actions_final > 0
        if np.any(buys_mask):
            new_cost_basis = buys * current_closings 
            numerator = (self.avg_buy_price * holdings) + new_cost_basis
            denominator = holdings_updated
            safe_denom = np.where(denominator == 0, 1, denominator) 
            new_waps = numerator / safe_denom
            self.avg_buy_price = np.where(buys_mask, new_waps, self.avg_buy_price)

        self.avg_buy_price = np.where(holdings_updated == 0, 0, self.avg_buy_price)

        self.step_index += 1
        self.step_in_episode += 1

        truncated = False
        if self.episode_length > 0 and self.step_in_episode >= self.episode_length:
            truncated = True

        new_closings = np.array(self.get_step_vector(self.step_index, cols=["close"]))
        new_asset_value = np.dot(holdings_updated, new_closings)
        new_total_assets = coh + new_asset_value

        pnl_reward = ((new_total_assets - prev_total_assets) / prev_total_assets) * 1000
        
        # Update peak and calculate current drawdown
        self.peak_total_assets = max(self.peak_total_assets, new_total_assets)
        current_drawdown = (new_total_assets - self.peak_total_assets) / self.peak_total_assets
        
        if self.incremental_drawdown_penalty:
            # Calculate incremental drawdown penalty (only punish when drawdown deepens)
            if current_drawdown < self.prev_drawdown:
                penalized_drawdown = current_drawdown - self.prev_drawdown
            else:
                penalized_drawdown = 0.0
        else:
            # Continuous drawdown penalty
            penalized_drawdown = current_drawdown
            
        self.prev_drawdown = current_drawdown
        drawdown_penalty = penalized_drawdown * 1000 
        
        reward = (self.reward_weight_pnl * pnl_reward) + (self.reward_weight_drawdown * drawdown_penalty)

        self.account_information["cash"].append(coh)
        self.account_information["asset_value"].append(new_asset_value)
        self.account_information["total_assets"].append(new_total_assets)
        self.account_information["reward"].append(reward)

        next_state = np.array(
            [coh] + list(holdings_updated) + self.get_step_vector(self.step_index), dtype=np.float32
        )
        self.state_memory.append(next_state)

        return self.get_full_state(), reward, False, truncated, {}

    def save_asset_memory(self):
        if self.current_step == 0:
            if hasattr(self, "final_asset_memory") and self.final_asset_memory is not None:
                return self.final_asset_memory
            return None
        else:
            self.account_information["timestamp"] = self.timestamps[
                self.starting_point:self.starting_point + len(self.account_information["cash"])
            ]
            return pd.DataFrame(self.account_information)

    def save_action_memory(self):
        if self.current_step == 0:
            if hasattr(self, "final_action_memory") and self.final_action_memory is not None:
                return self.final_action_memory
            return None
        else:
            return pd.DataFrame(
                {
                    "timestamp": self.timestamps[self.starting_point:self.starting_point + len(self.actions_memory)],
                    "actions": self.actions_memory,
                    "transactions": self.transaction_memory,
                    "stoploss_mask": self.stoploss_memory,
                }
            )

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Test Minute-Level Trading Environment")
    parser.add_argument("--data_path", type=str, required=True, help="Path to preprocessed CSV data")
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}")
    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} not found.")
        exit(1)
        
    df = pd.read_csv(args.data_path)
    
    env_kwargs = {
        "hmax": 100000,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "discrete_actions": False,
        "feature_columns": ["open", "close", "high", "low", "volume", "macd", "rsi_30", "cci_30", "dx_30"],
        "episode_length": 100,
        "random_start": True
    }
    
    # Filter columns to what we actually have in df
    available_cols = [c for c in env_kwargs["feature_columns"] if c in df.columns]
    env_kwargs["feature_columns"] = available_cols
    
    print(f"Initializing Environment... using columns: {available_cols}")
    env = StockTradingEnv(df=df, **env_kwargs)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    print("Testing random steps...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            break
            
    print(f"Completed 10 random steps. Cumulative reward: {total_reward}")
