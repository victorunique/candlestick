# Candlestick - FinRL TDD Setup

This project implements a stock trading agent using Deep Reinforcement Learning (PPO), inspired by FinRL's rapid trading strategy with a hard stop-loss. 

All components have been rewritten following Test-Driven Development (TDD) principles to be fully self-contained, reading and writing entirely from local files.

## Project Structure
```
├── README.md
├── pyproject.toml
├── src
│   ├── data_fetcher.py               # Fetches Yahoo Finance data
│   ├── feature_engineer.py           # Calculates MACD, RSI, CCI, DX
│   ├── env_stocktrading_minute.py    # Custom Gym environment with dynamic stop-loss  
│   ├── custom_models.py              # CNN1DFeaturesExtractor PyTorch model
│   ├── train_ppo.py                  # Trains the PPO agent
│   └── backtest.py                   # Tests the agent on unseen data
├── tests                             # Pytest unit tests for every component
│   ├── test_data_fetcher.py
│   ├── test_feature_engineer.py
│   ├── test_env_stocktrading_minute.py
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
uv run src/data_fetcher.py --start_date 2024-01-01 --end_date 2024-06-01 --ticker_list AAPL MSFT --output_path ./data/raw_data.csv
```

### 2. Feature Engineering
Processes the raw data and calculates technical indicators (default: `macd`, `rsi_30`, `cci_30`, `dx_30`).
```bash
uv run src/feature_engineer.py --input_path ./data/raw_data.csv --output_path ./data/processed_data.csv
```

### 3. Train the Agent
Initializes the minute-level trading environment and trains the PPO agent with 1D CNN extraction. Saves the `.zip` model and the `_vecnormalize.pkl` stats.
```bash
uv run src/train_ppo.py --data_path ./data/processed_data.csv --model_dir ./trained_models --model_name my_ppo_bot --total_timesteps 10000
```

### 4. Backtesting
Loads the trained model and normalizer, predicting actions on the dataset and calculating PnL.
```bash
# In practice, you should fetch new data for backtesting (e.g. 2024-06-01 to 2024-09-01) 
# and process it through feature_engineer.py first.
uv run src/backtest.py --data_path ./data/processed_data.csv --model_path ./trained_models/my_ppo_bot --results_dir ./results
```
The backtest will output `my_ppo_bot_account_history.csv` (portfolio value over time) and `my_ppo_bot_action_history.csv` (trade execution log) in the specified results directory.
