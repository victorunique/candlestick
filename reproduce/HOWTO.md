# Reproducing Research Results

This guide explains how to reproduce the hyperparameter tuning and trading agent evaluation results using the provided configuration and historical 1-minute data.

> [!IMPORTANT]
> All commands listed in this guide must be executed from the **project root directory** (not from within the `reproduce` folder).

## Prerequisites

Ensure you have the following installed on your system:
- **Python** (version 3.10+ recommended)
- **uv** (fast Python package installer and resolver)

If you haven't synced the dependencies yet, run the following command in the project root directory:
```bash
uv sync
```

---

## Step-by-Step Instructions

### 1. Extract the Historical Data
The 1-minute historical data (covering ticker symbols `AAPL`, `MSFT`, and `NVDA`) is packaged in `reproduce/history.zip`. Unzip it directly into the project root directory:

```bash
unzip reproduce/history.zip -d .
```

*Note: This creates a `./history` directory in the project root containing the historical `.csv` files required for localized training and testing.*

### 2. Copy the Configuration File
Copy the hyperparameter grid configuration file `run_pipeline_config.json` from the `reproduce` directory into the project root directory (backup the existing one if needed):

```bash
cp reproduce/run_pipeline_config.json .
```

### 3. Run the Hyperparameter Tuning Pipeline
Execute the pipeline module using `uv`. To prevent Yahoo Finance API rate limits and data availability limitations, specify the local history dataset using `--use-local-history`.

Run the command in the background, redirecting output to `pipeline.log`:

```bash
uv run python -m src.run_pipeline --output ./results/hparam_results.csv --use-local-history >./pipeline.log 2>&1 &
```

#### What this command does:
- **`uv run python -m src.run_pipeline`**: Executes the hyperparameter tuning script within the virtual environment managed by `uv`.
- **`--output ./results/hparam_results.csv`**: Saves the cumulative grid search results to this CSV path.
- **`--use-local-history`**: Tells the pipeline to fetch OHLCV data locally from the `./history` folder (which defaults to `--local-history-dir ./history`).
- **`>./pipeline.log 2>&1 &`**: Redirects both standard output (stdout) and standard error (stderr) to `pipeline.log` and runs the process in the background.

---

## Monitoring Progress

Since the tuning process runs in the background, you can monitor the execution log in real time:

```bash
tail -f pipeline.log
```

Alternatively, you can inspect the output file `./results/hparam_results.csv` to see the completed hyperparameter combinations.

---

## Next Steps: Pareto Front and Significance Analysis

Once the pipeline completes, you can visualize the return-drawdown Pareto front and run Wilcoxon signed-rank tests using the generated results:

1. **Pareto Front Visualizations**:
   ```bash
   uv run python -m src.analyse_pareto --input ./results/hparam_results.csv
   ```
   This will output:
   - `pareto_front.png`: Average return vs. average max drawdown trade-off curve.
   - `sharpe_comparison.png` and `sortino_comparison.png`: Comparison of Sharpe and Sortino ratios between PPO (Dynamic SL), Fixed-SL, and Buy-and-Hold strategies.

2. **Statistical Significance Testing**:
   ```bash
   uv run python -m src.wilcoxon_test --input ./results/hparam_results.csv
   ```
