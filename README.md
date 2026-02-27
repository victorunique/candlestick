# Candlestick - FinRL TDD Setup

This project implements a stock trading agent using Deep Reinforcement Learning (PPO), inspired by FinRL's rapid trading strategy with a hard stop-loss. 

All components have been rewritten following Test-Driven Development (TDD) principles to be fully self-contained, reading and writing entirely from local files.

## Project Structure
```
├── README.md
├── pyproject.toml
├── src
│   ├── data_fetcher.py               # Fetches Yahoo Finance data
│   ├── feature_engineer.py           # Calculates 23 technical indicators (ATR, MACD, RSI, etc.)
│   ├── env_stocktrading.py           # Custom Gym environment with dynamic stop-loss  
│   ├── custom_models.py              # CNN1DFeaturesExtractor PyTorch model
│   ├── train_ppo.py                  # Trains the PPO agent
│   ├── plot_training_curve.py        # Visualizes the RL learning curve
│   ├── backtest.py                   # Tests the agent on unseen data
│   └── plot_backtest.py              # Visualizes portfolio value, drawdown, and buy/sell actions
├── tests                             # Pytest unit tests for every component
│   ├── test_data_fetcher.py
│   ├── test_feature_engineer.py
│   ├── test_env_stocktrading.py
│   ├── test_custom_models.py
│   ├── test_train_ppo.py
│   └── test_backtest.py
```

## Setup & Running Tests
This project uses `uv` for dependency management and execution.

1. **Install Dependencies**
```bash
uv sync
```

2. **Run All Tests**
```bash
uv run pytest tests/
```

## End-to-End Pipeline Usage

Every script provides a `--help` interface. You can string them together as follows:

### 1. Fetch Data
Downloads raw stock price data and formats it correctly.
```bash
uv run python -m src.data_fetcher --start_date 2024-01-01 --end_date 2024-06-01 --ticker_list AAPL MSFT --output_path ./data/raw_data.csv
```

### 2. Feature Engineering
Processes the raw data and calculates technical indicators. Default indicators include `atr`, `macd`, `macds`, `macdh`, `boll_ub`, `boll_lb`, `boll`, `rsi_14`, `rsi_6`, `adx`, `dx`, `close_12_ema`, `close_26_ema`, `close_50_ema`, `close_30_sma`, `close_60_sma`, `kdjk`, `kdjd`, `kdjj`, `cci`, `cci_20`, `tr`, and `atr_20`.
```bash
uv run python -m src.feature_engineer --input_path ./data/raw_data.csv --output_path ./data/processed_data.csv
```

### 3. Train the Agent
Initializes the trading environment and trains the PPO agent with 1D CNN extraction. Saves the `.zip` model and the `_vecnormalize.pkl` stats.
```bash
uv run python -m src.train_ppo --data_path ./data/processed_data.csv --model_dir ./trained_models --model_name my_ppo_bot --total_timesteps 10000 --seed 42
```
Additional environment hyperparameters: `--hmax`, `--stoploss_penalty`, `--profit_loss_ratio`, `--cash_penalty`. Run `--help` for details.

### 4. Plot Training Curve
Visualize the learning curve to verify that the agent is actually learning (e.g., reward trending up) by reading the training-log CSV.
```bash
uv run python -m src.plot_training_curve --log_path trained_models/my_ppo_bot_training_log.csv
```

### 5. Backtesting
Loads the trained model and normalizer, predicting actions on the dataset and calculating PnL.
```bash
# In practice, you should fetch new data for backtesting (e.g. 2024-06-01 to 2024-09-01) 
# and process it through feature_engineer.py first.
uv run python -m src.backtest --data_path ./data/processed_data.csv --model_path ./trained_models/my_ppo_bot --results_dir ./results
```
Additional environment hyperparameters: `--hmax`, `--stoploss_penalty`, `--profit_loss_ratio`, `--cash_penalty`, `--window_size`. Run `--help` for details.

The backtest will output `my_ppo_bot_account_history.csv` (portfolio value over time) and `my_ppo_bot_action_history.csv` (trade execution log) in the specified results directory.

### 6. Visualize Backtest Results
Plot the portfolio total assets over time, maximum drawdown, and asset value with overlaid buy/sell signals.
```bash
uv run python -m src.plot_backtest \
    --account_path results/my_ppo_bot_account_history.csv \
    --action_path results/my_ppo_bot_action_history.csv
```
