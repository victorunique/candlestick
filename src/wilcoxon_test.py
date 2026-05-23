"""Wilcoxon signed-rank tests comparing PPO vs Fixed-SL and Buy-and-Hold.

Performs non-parametric paired tests on Return, Max Drawdown, and Sortino
ratio. To avoid pseudoreplication, results are first aggregated by
(ticker, window) — averaging across all reward_weight_drawdown settings —
so that each (ticker, window) pair contributes exactly one paired observation.

Usage:
    uv run python -m src.wilcoxon_test
    uv run python -m src.wilcoxon_test --input hparam_results_all.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REQUIRED_COLUMNS = {
    "tickers", "test_start",
    "ppo_return", "ppo_max_dd", "ppo_sortino",
    "fsl_return", "fsl_max_dd", "fsl_sortino",
    "bh_return", "bh_max_dd", "bh_sortino",
}

# Metric columns to aggregate
METRIC_COLUMNS = [
    "ppo_return", "ppo_max_dd", "ppo_sortino",
    "fsl_return", "fsl_max_dd", "fsl_sortino",
    "bh_return", "bh_max_dd", "bh_sortino",
]

# Significance threshold
ALPHA = 0.05


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and validate expected columns."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def aggregate_by_window(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by (ticker, window) to eliminate pseudoreplication.

    For each (ticker, test_start) group, computes the mean of all
    performance metrics across the different reward_weight_drawdown
    settings. This ensures each (ticker, window) contributes exactly
    one independent observation to the Wilcoxon test.
    """
    agg = df.groupby(["tickers", "test_start"])[METRIC_COLUMNS].mean()
    return agg.reset_index()


def run_wilcoxon(series_a: pd.Series, series_b: pd.Series) -> dict:
    """Run a Wilcoxon signed-rank test on two paired series.

    Returns a dict with keys: statistic, p_value, significant.
    If all differences are zero, returns p_value=1.0 (no difference).
    """
    diff = series_a.values - series_b.values

    # If all differences are zero, no test is needed
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    # Drop zero differences (Wilcoxon convention) and check if enough remain
    non_zero = diff[diff != 0]
    if len(non_zero) < 1:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
        }

    stat, p_val = wilcoxon(series_a, series_b, alternative="two-sided")
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": bool(p_val < ALPHA),
    }


def build_results_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the Wilcoxon test results table.

    First aggregates by (ticker, window) to avoid pseudoreplication,
    then compares PPO vs Fixed SL and PPO vs Buy&Hold
    across Return, Max Drawdown, and Sortino metrics.

    Returns a DataFrame with columns:
        Comparison, Metric, Test, p-value, Significant
    """
    agg = aggregate_by_window(df)

    comparisons = [
        ("PPO vs Fixed SL", "ppo", "fsl"),
        ("PPO vs Buy&Hold", "ppo", "bh"),
    ]
    metrics = [
        ("Return", "return"),
        ("Max Drawdown", "max_dd"),
        ("Sortino", "sortino"),
    ]

    rows = []
    for comp_label, prefix_a, prefix_b in comparisons:
        for metric_label, metric_suffix in metrics:
            col_a = f"{prefix_a}_{metric_suffix}"
            col_b = f"{prefix_b}_{metric_suffix}"
            result = run_wilcoxon(agg[col_a], agg[col_b])
            rows.append({
                "Comparison": comp_label,
                "Metric": metric_label,
                "Test": "Wilcoxon",
                "p-value": result["p_value"],
                "Significant": "Yes" if result["significant"] else "No",
            })

    return pd.DataFrame(rows)


def print_results(table: pd.DataFrame, n_obs: int, n_raw: int) -> None:
    """Print the results table and interpretation to stdout."""
    print("=" * 72)
    print("Wilcoxon Signed-Rank Test Results")
    print(f"Raw rows loaded: {n_raw}")
    print(f"Number of paired observations: {n_obs}")
    print(f"  (aggregated by ticker × window to avoid pseudoreplication)")
    print(f"Significance level: α = {ALPHA}")
    print("=" * 72)
    print()

    # Print formatted table
    print(table.to_string(index=False))
    print()

    # Interpretation
    print("-" * 72)
    print("Interpretation:")
    print("-" * 72)
    for _, row in table.iterrows():
        sig = "✓ Significant" if row["Significant"] == "Yes" else "✗ Not significant"
        direction = ""
        if row["Significant"] == "Yes":
            direction = " — PPO median differs from baseline"
        print(f"  {row['Comparison']:20s} | {row['Metric']:14s} | "
              f"p={row['p-value']:.6f} | {sig}{direction}")
    print("=" * 72)


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Wilcoxon signed-rank tests for PPO vs baselines.",
    )
    parser.add_argument(
        "--input", type=str, default="hparam_results_all.csv",
        help="Path to the hyperparameter results CSV.",
    )
    args = parser.parse_args(argv)

    df = load_data(args.input)
    n_raw = len(df)
    agg = aggregate_by_window(df)
    n_obs = len(agg)
    print(f"Loaded {n_raw} rows from {args.input}")
    print(f"Aggregated to {n_obs} independent (ticker, window) pairs\n")

    table = build_results_table(df)
    print_results(table, n_obs=n_obs, n_raw=n_raw)


if __name__ == "__main__":
    main()
