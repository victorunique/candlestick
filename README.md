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
│   ├── run_pipeline.py               # Hyperparameter tuning pipeline (grid search)
│   ├── plot_backtest.py              # Visualizes portfolio value, drawdown, and buy/sell actions
│   └── generate_report.py            # Generates a static HTML report with dataset info and graphs
├── tests                             # Pytest unit tests for every component
│   ├── test_data_fetcher.py
│   ├── test_feature_engineer.py
│   ├── test_env_stocktrading.py
│   ├── test_custom_models.py
│   ├── test_train_ppo.py
│   ├── test_backtest.py
│   ├── test_run_pipeline.py
│   ├── test_plot_backtest.py
│   └── test_generate_report.py
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
uv run python -m src.data_fetcher --start_date 2020-01-01 --end_date 2025-06-30 --ticker_list AAPL MSFT TSLA META AMZN GOOGL --output_path ./data/data_training.csv --interval 1d
```

### 2. Feature Engineering
Processes the raw data and calculates technical indicators. Default indicators include `atr`, `macd`, `macds`, `macdh`, `boll_ub`, `boll_lb`, `boll`, `rsi_14`, `rsi_6`, `adx`, `dx`, `close_12_ema`, `close_26_ema`, `close_50_ema`, `close_30_sma`, `close_60_sma`, `kdjk`, `kdjd`, `kdjj`, `cci`, `cci_20`, `tr`, and `atr_20`.
```bash
uv run python -m src.feature_engineer --input_path ./data/data_training.csv --output_path ./data/data_training_preprocessed.csv
```

### 3. Train the Agent
Initializes the trading environment and trains the PPO agent with 1D CNN extraction. Saves the `.zip` model and the `_vecnormalize.pkl` stats.
```bash
uv run python -m src.train_ppo --data_path ./data/data_training_preprocessed.csv --model_dir ./trained_models --model_name my_ppo_bot --total_timesteps 10000 --seed 42
```
Additional environment hyperparameters: `--hmax`, `--stoploss_penalty`, `--profit_loss_ratio`, `--cash_penalty`, `--reward_weight_pnl`, `--reward_weight_drawdown`. Run `--help` for details.

### 4. Plot Training Curve (Optional)
Visualize the learning curve to verify that the agent is actually learning (e.g., reward trending up) by reading the training-log CSV.
```bash
uv run python -m src.plot_training_curve --log_path trained_models/my_ppo_bot_training_log.csv
```

### 5. Backtesting
Loads the trained model and normalizer, predicting actions on the dataset and calculating PnL.
```bash
# In practice, you should fetch new data for backtesting (e.g. 2024-06-01 to 2024-09-01) 
# and process it through feature_engineer.py first.
uv run python -m src.backtest --data_path ./data/data_testing_preprocessed.csv --model_path ./trained_models/my_ppo_bot --results_dir ./results
```
Additional environment hyperparameters: `--hmax`, `--stoploss_penalty`, `--profit_loss_ratio`, `--cash_penalty`, `--window_size`, `--fixed_stoploss_ratio`. Run `--help` for details.

The backtest will output `my_ppo_bot_account_history.csv` (portfolio value over time), `my_ppo_bot_action_history.csv` (trade execution log), `baseline_buy_and_hold_account_history.csv` (buy-and-hold baseline), and `fixed_stoploss_account_history.csv` (PPO with fixed stop-loss) in the specified results directory.

#### Machine-readable output (`--plaintext`)

Add `--plaintext` to emit a single comma-separated line of decimal values (no labels, no `%` signs) for scripting and hyperparameter tuning:

```bash
uv run python -m src.backtest --data_path ./data/data_testing_preprocessed.csv --model_path ./trained_models/my_ppo_bot --results_dir ./results --plaintext
```

Output format: `<ppo_return>,<ppo_max_dd>,<fsl_return>,<fsl_max_dd>,<bh_return>,<bh_max_dd>`

Example: `-0.0016,-0.0069,-0.0163,-0.0228,-0.0089,-0.0158`

The printed output includes a three-way comparison:
1. **PPO Agent** — the trained RL model with its learned dynamic stop-loss ratios.
2. **Buy & Hold Baseline** — equal-weight allocation across all tickers from day 1.
3. **PPO + Fixed SL** — the same RL model but with the stop-loss ratio overridden to a fixed value (default 95%, configurable via `--fixed_stoploss_ratio`).

### 6. Visualize Backtest Results (Optional)
Plot the portfolio total assets over time, maximum drawdown, and per-ticker close prices with overlaid buy/sell signals.
```bash
uv run python -m src.plot_backtest \
    --account_path results/my_ppo_bot_account_history.csv \
    --action_path  results/my_ppo_bot_action_history.csv \
    --baseline_path results/baseline_buy_and_hold_account_history.csv \
    --fixed_sl_path results/fixed_stoploss_account_history.csv \
    --data_path data/data_testing_preprocessed.csv
```

The `--baseline_path` and `--fixed_sl_path` flags are optional; when provided, their strategy's portfolio value and max drawdown are overlaid on the first two charts for direct comparison.

The `--data_path` flag is optional; when provided, the third graph plots each ticker's actual close price with per-ticker buy/sell arrows. Without it, the graph falls back to plotting the aggregate asset value (market exposure).

### 7. Generate Static HTML Report
Create a comprehensive, academic-style static HTML report containing dataset information, performance metrics, and embedded visualizations from backtesting and training.

```bash
uv run python -m src.generate_report \
    --account_path results/my_ppo_bot_account_history.csv \
    --action_path results/my_ppo_bot_action_history.csv \
    --log_path trained_models/my_ppo_bot_training_log.csv \
    --train_data_path data/data_training_preprocessed.csv \
    --test_data_path data/data_testing_preprocessed.csv \
    --baseline_path results/baseline_buy_and_hold_account_history.csv \
    --fixed_sl_path results/fixed_stoploss_account_history.csv \
    --output_path results/report.html
```

### 8. Hyperparameter Tuning Pipeline

Automates the entire 4-step pipeline (fetch → feature-engineer → train → backtest) across a grid of hyperparameters, accumulating results into a single CSV.

#### Configure the grid

Edit the `*_LIST` constants at the top of `src/run_pipeline.py`:

| Variable | Description |
|---|---|
| `DATE_RANGES` | List of `(train_start, train_end, test_start, test_end)` tuples |
| `TICKER_LISTS` | List of ticker lists (shared for train & test) |
| `TOTAL_TIMESTEPS_LIST` | Training timesteps |
| `REWARD_WEIGHT_PNL_LIST` | Reward weight for PnL |
| `REWARD_WEIGHT_DRAWDOWN_LIST` | Reward weight for drawdown penalty |
| `N_STEPS_LIST`, `ENT_COEF_LIST`, `LEARNING_RATE_LIST`, `GAMMA_LIST`, `EPISODE_LENGTH_LIST` | PPO hyperparameters |

Fixed values (not part of the grid): `interval=1m`, `seed=42`, `fixed_stoploss_ratio=0.95`, `initial_amount=1000000`.

#### Preview combinations (dry run)
```bash
uv run python -m src.run_pipeline --dry-run
```

#### Run the full grid
```bash
uv run python -m src.run_pipeline --output ./results/hparam_results.csv
```

#### Resume from a specific combo
```bash
uv run python -m src.run_pipeline --start-from 5 --output ./results/hparam_results.csv
```

#### Output CSV format

```
combo_id,train_start,train_end,test_start,test_end,tickers,total_timesteps,
reward_weight_pnl,reward_weight_drawdown,n_steps,ent_coef,learning_rate,gamma,
episode_length,ppo_return,ppo_max_dd,fsl_return,fsl_max_dd,bh_return,bh_max_dd
```

Each combo runs in an isolated temp directory that is cleaned up afterwards, ensuring no state leakage between runs.
