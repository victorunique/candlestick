# Candlestick - FinRL TDD Setup

This project implements a stock trading agent using Deep Reinforcement Learning (PPO), inspired by FinRL's rapid trading strategy with a hard stop-loss. 

All components have been rewritten following Test-Driven Development (TDD) principles to be fully self-contained, reading and writing entirely from local files.

## Project Structure
```
├── README.md
├── pyproject.toml
├── src
│   ├── data_fetcher.py               # Fetches Yahoo Finance data
│   ├── history_collector.py          # Periodic 1-min OHLCV collector (cron-scheduled)
│   ├── feature_engineer.py           # Calculates 23 technical indicators (ATR, MACD, RSI, etc.)
│   ├── env_stocktrading.py           # Custom Gym environment with dynamic stop-loss  
│   ├── custom_models.py              # CNN1DFeaturesExtractor PyTorch model
│   ├── train_ppo.py                  # Trains the PPO agent
│   ├── plot_training_curve.py        # Visualizes the RL learning curve
│   ├── backtest.py                   # Tests the agent on unseen data
│   ├── run_pipeline.py               # Hyperparameter tuning pipeline (grid search)
│   ├── plot_backtest.py              # Visualizes portfolio value, drawdown, and buy/sell actions
│   ├── generate_report.py            # Generates a static HTML report with dataset info and graphs
│   └── analyse_pareto.py             # Pareto front analysis and Sharpe/Sortino cross-strategy comparison
├── tests                             # Pytest unit tests for every component
│   ├── test_data_fetcher.py
│   ├── test_history_collector.py
│   ├── test_feature_engineer.py
│   ├── test_env_stocktrading.py
│   ├── test_custom_models.py
│   ├── test_train_ppo.py
│   ├── test_backtest.py
│   ├── test_run_pipeline.py
│   ├── test_plot_backtest.py
│   ├── test_generate_report.py
│   └── test_analyse_pareto.py
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
uv run python -m src.data_fetcher --start_date 2020-01-01 --end_date 2025-06-30 --ticker_list AAPL MSFT TSLA META AMZN GOOGL --output_path ./data/data_training.csv --interval 1m
```

### 1b. Load Local Data (Alternative)
Manually fetches 1-minute historical data collected from the local `history_collector.py` dataset, bypassing Yahoo Finance requirements constraints.
```bash
uv run python -m src.data_loader --start_date 2026-02-09 --end_date 2026-02-28 --ticker_list AAPL MSFT TSLAMETA AMZN GOOGL --data_dir ./history --output_path ./data/data_training.csv
```

### 2. Feature Engineering
Processes the raw data and calculates technical indicators. Default indicators include `atr`, `macd`, `macds`, `macdh`, `boll_ub`, `boll_lb`, `boll`, `rsi_14`, `rsi_6`, `adx`, `dx`, `close_12_ema`, `close_26_ema`, `close_50_ema`, `close_30_sma`, `close_60_sma`, `kdjk`, `kdjd`, `kdjj`, `cci`, `cci_20`, `tr`, and `atr_20`.
```bash
uv run python -m src.feature_engineer --input_path ./data/data_training.csv --output_path ./data/data_training_preprocessed.csv
```
You can optionally supply `--indicator_list` to override the defaults. (*Note: The `--start_date` parameter was removed; calculating indicators no longer clips the start of the data.*)

### 3. Train the Agent
Initializes the trading environment and trains the PPO agent with 1D CNN extraction. Saves the `.zip` model and the `_vecnormalize.pkl` stats. Note the `--train_start` parameter, which is used to tell the agent exactly which date to start trading from (it will look back `window_size - 1` days natively for the CNN).
```bash
uv run python -m src.train_ppo --data_path ./data/data_training_preprocessed.csv --model_dir ./trained_models --model_name my_ppo_bot --train_start 2020-01-01 --total_timesteps 10000 --seed 42
```
Additional environment hyperparameters: `--hmax`, `--stoploss_min`, `--stoploss_max`, `--cash_penalty`, `--upside_pnl_multiplier`, `--reward_weight_pnl`, `--reward_weight_drawdown`, `--continuous_drawdown_penalty`. Run `--help` for details.

### 4. Plot Training Curve (Optional)
Visualize the learning curve to verify that the agent is actually learning (e.g., reward trending up) by reading the training-log CSV.
```bash
uv run python -m src.plot_training_curve --log_path trained_models/my_ppo_bot_training_log.csv
```

### 5. Backtesting
Loads the trained model and normalizer, predicting actions on the dataset and calculating PnL. Like `--train_start`, use `--test_start` to tell the environment the date to begin its first trading step.
```bash
# In practice, you should fetch new data for backtesting (e.g. 2024-06-01 to 2024-09-01) 
uv run python -m src.backtest --data_path ./data/data_testing_preprocessed.csv --model_path ./trained_models/my_ppo_bot --results_dir ./results --test_start 2024-06-01
```
Additional environment hyperparameters: `--hmax`, `--stoploss_min`, `--stoploss_max`, `--cash_penalty`, `--upside_pnl_multiplier`, `--window_size`, `--fixed_stoploss_ratio`. Run `--help` for details.

The backtest will output `my_ppo_bot_account_history.csv` (portfolio value over time), `my_ppo_bot_action_history.csv` (trade execution log), `baseline_buy_and_hold_account_history.csv` (buy-and-hold baseline), and `fixed_stoploss_account_history.csv` (PPO with fixed stop-loss) in the specified results directory.

#### Machine-readable output (`--plaintext`)

Add `--plaintext` to emit a single comma-separated line of decimal values (no labels, no `%` signs) for scripting and hyperparameter tuning:

```bash
uv run python -m src.backtest --data_path ./data/data_testing_preprocessed.csv --model_path ./trained_models/my_ppo_bot --results_dir ./results --plaintext
```

Output format: `<ppo_return>,<ppo_max_dd>,<ppo_sharpe>,<ppo_sortino>,<fsl_return>,<fsl_max_dd>,<fsl_sharpe>,<fsl_sortino>,<bh_return>,<bh_max_dd>,<bh_sharpe>,<bh_sortino>`

Example: `-0.0016,-0.0069,0.5234,-0.8123,-0.0163,-0.0228,0.3012,-0.6045,-0.0089,-0.0158,0.4100,-0.7200`

The printed output includes a three-way comparison:
1. **PPO Agent** — the trained RL model with its learned dynamic stop-loss ratios.
2. **Buy & Hold Baseline** — equal-weight allocation across all tickers from day 1.
3. **PPO + Fixed SL** — the same RL model but with the stop-loss ratio overridden to a fixed value (default 95%, configurable via `--fixed_stoploss_ratio`).

Each strategy reports: total return, max drawdown, **Sharpe ratio**, and **Sortino ratio**. The Sharpe ratio measures risk-adjusted return relative to total volatility; the Sortino ratio is similar but only penalises *downside* volatility.

### 6. Visualize Backtest Results (Optional)
Plot the portfolio total assets over time, maximum drawdown, and per-ticker close prices with overlaid buy/sell signals.
```bash
uv run python -m src.plot_backtest \
    --account_path results/my_ppo_bot_account_history.csv \
    --action_path  results/my_ppo_bot_action_history.csv \
    --baseline_path results/baseline_buy_and_hold_account_history.csv \
    --fixed_sl_path results/fixed_stoploss_account_history.csv \
    --data_path data/data_testing_preprocessed.csv \
    --test_start 2024-06-01
```

The `--baseline_path` and `--fixed_sl_path` flags are optional; when provided, their strategy's portfolio value and max drawdown are overlaid on the first two charts for direct comparison.

The `--data_path` flag is optional; when provided, the third graph plots each ticker's actual close price with per-ticker buy/sell arrows. Without it, the graph falls back to plotting the aggregate asset value (market exposure).

The `--test_start` flag is also optional; when provided, the plot will exclude any timestamp prior to the specified date, effectively clipping out the window-size warmup period for a truer representation of the test performance.

### 7. Generate Static HTML Report
Create a comprehensive, academic-style static HTML report containing dataset information, performance metrics (total return, max drawdown, Sharpe ratio, Sortino ratio), and embedded visualizations from backtesting and training.

```bash
uv run python -m src.generate_report \
    --account_path results/my_ppo_bot_account_history.csv \
    --action_path results/my_ppo_bot_action_history.csv \
    --log_path trained_models/my_ppo_bot_training_log.csv \
    --train_data_path data/data_training_preprocessed.csv \
    --test_data_path data/data_testing_preprocessed.csv \
    --baseline_path results/baseline_buy_and_hold_account_history.csv \
    --fixed_sl_path results/fixed_stoploss_account_history.csv \
    --train_start 2020-01-01 \
    --test_start 2024-06-01 \
    --output_path results/report.html
```

### 8. Hyperparameter Tuning Pipeline

Automates the entire 4-step pipeline (fetch → feature-engineer → train → backtest) across a grid of hyperparameters, accumulating results into a single CSV.

**Note:** To cleanly avoid look-ahead bias without penalizing the dataset length, the pipeline dynamically provisions a 5-day "warm-up" data lookback explicitly for the training and backtesting fetchers. The data is sliced chronologically natively inside `train_ppo.py` and `backtest.py` based on `window_size` rather than relying on `VecFrameStack`.

#### Configure the grid

Edit the `run_pipeline_config.json` file in the root directory to specify the search space:

| Variable | Description |
|---|---|
| `date_ranges` | List of `[train_start, train_end, test_start, test_end]` lists |
| `ticker_lists` | List of ticker lists (shared for train & test) |
| `total_timesteps_list` | Training timesteps |
| `reward_weight_pnl_list` | Reward weight for PnL |
| `reward_weight_drawdown_list` | Reward weight for drawdown penalty |
| `cash_penalty_list` | Penalty proportion for uninvested cash |
| `upside_pnl_multiplier_list` | Standard positive PnL multiplier |
| `stoploss_min_list` | Minimum stop-loss ratio threshold |
| `stoploss_max_list` | Maximum stop-loss ratio threshold |
| `n_steps_list`, `ent_coef_list`, `learning_rate_list`, `gamma_list`, `episode_length_list` | PPO hyperparameters |

Fixed values (not part of the grid): `interval=1m`, `seed=42`, `fixed_stoploss_ratio=0.95`, `initial_amount=1000000`.

#### Preview combinations (dry run)
```bash
uv run python -m src.run_pipeline --dry-run --config run_pipeline_config.json
```

#### Run the full grid
```bash
uv run python -m src.run_pipeline --config run_pipeline_config.json --output ./results/hparam_results.csv
```

#### Resume from a specific combo
```bash
uv run python -m src.run_pipeline --start-from 5 --output ./results/hparam_results.csv
```

#### Using local history data
To run the pipeline using locally collected 1-minute historical data (from `history_collector.py`) instead of querying Yahoo Finance every time, use the following flags:
```bash
uv run python -m src.run_pipeline --use-local-history --local-history-dir ./history --output ./results/hparam_results.csv
```
This is especially useful since the Yahoo Finance API cannot provide 1-minute level data earlier than 30 calendar days.

#### Output CSV format

```
combo_id,train_start,train_end,test_start,test_end,tickers,total_timesteps,
reward_weight_pnl,reward_weight_drawdown,cash_penalty_proportion,upside_pnl_multiplier,
stoploss_min,stoploss_max,
n_steps,ent_coef,learning_rate,gamma,
episode_length,ppo_return,ppo_max_dd,ppo_sharpe,ppo_sortino,
fsl_return,fsl_max_dd,fsl_sharpe,fsl_sortino,
bh_return,bh_max_dd,bh_sharpe,bh_sortino
```

Each combo runs in an isolated temp directory that is cleaned up afterwards, ensuring no state leakage between runs.

### 9. Analyse Hyperparameter Results (Pareto Front)

After accumulating results from the hyperparameter pipeline, use `analyse_pareto` to visualise the return–drawdown Pareto front and compare Sharpe/Sortino ratios across the three strategies (PPO, Fixed-SL, Buy-and-Hold).

```bash
uv run python -m src.analyse_pareto --input hparam_results.csv
```

| Parameter | Default | Description |
|---|---|---|
| `--input` | `hparam_results.csv` | Path to the pipeline results CSV |
| `--output` | `pareto_front.png` | Output path for the Pareto front plot |
| `--output-sharpe` | `sharpe_comparison.png` | Output path for the Sharpe ratio comparison chart |
| `--output-sortino` | `sortino_comparison.png` | Output path for the Sortino ratio comparison chart |

The tool:
1. Aggregates results by `reward_weight_drawdown` across all tickers and rolling windows.
2. Computes and plots the **Pareto front** (average return vs average max drawdown).
3. Generates separate **grouped bar charts** comparing average Sharpe and Sortino ratios for PPO, Fixed-SL, and Buy-and-Hold at each weight setting.
4. Prints a summary table with all metrics and a cross-strategy comparison to stdout.

## Continuous 1-Minute Data Collection

The `history_collector` module periodically fetches **1-minute OHLCV data** from Yahoo Finance and saves it as per-ticker, per-day CSV files. This is useful because Yahoo Finance's API sometimes provides inconsistent historical coverage for 1-minute data — continuous collection over time builds a reliable long-term dataset.

### How It Works

- For each configured ticker and each day in the lookback window, it checks whether a CSV file already exists.
- If the file exists → skip.
- If the file is missing → fetch from Yahoo Finance via `data_fetcher.py` and save.
- On non-trading days (weekends, holidays), Yahoo Finance returns no data. The collector logs a warning and moves on — **no empty files are created**.

### Storage Format

Files are saved as:
```
<TICKER>_<YYYY-MM-DD>_1m.csv
```

Example:
```
history/
├── AAPL_2026-03-03_1m.csv
├── AAPL_2026-03-04_1m.csv
├── TSLA_2026-03-03_1m.csv
└── TSLA_2026-03-04_1m.csv
```

### CLI Usage

```bash
uv run python -m src.history_collector \
  --tickers AAPL TSLA MSFT META AMZN GOOGL \
  --data_dir ./history \
  --lookback_days 7
```

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--tickers` | Yes | — | Space-separated list of ticker symbols |
| `--data_dir` | Yes | — | Directory to store CSV files |
| `--lookback_days` | No | `7` | Number of days to look back (including today) |

### Scheduling with Cron (Ubuntu)

The script runs **once per invocation** and exits. Use cron to schedule it every 6 hours:

```bash
crontab -e
```

Add the following line (adjust paths as needed):

```cron
0 */6 * * * cd /path/to/candlestick && /home/yourusername/.local/bin/uv run python -m src.history_collector --tickers AAPL TSLA MSFT META AMZN GOOGL --data_dir ./history >> ./history/collector.log 2>&1
```

This ensures:
- Data is checked and fetched 4 times daily.
- Output is appended to `./history/collector.log`.
- The process is stateless — safe to restart or reschedule at any time.

