"""
Tool-3: Static HTML Report Generator

Generates a static academic-style HTML report containing backtesting results,
dataset information, and training curves.

Usage:
    uv run python -m src.generate_report \
        --account_path results/account_history.csv \
        --action_path results/action_history.csv \
        --train_data_path data/dow30_2009_2019.csv \
        --test_data_path data/dow30_2020_2021.csv \
        --output_path results/report.html
"""

import argparse
import base64
import os
import sys
import tempfile
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend

from src.backtest import compute_sharpe, compute_sortino
from src.plot_backtest import plot_backtest
from src.plot_training_curve import plot_training_curve


def encode_image_base64(filepath: str) -> str:
    """Encode an image file to base64 string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_report(args):
    # Process backtest metrics
    if not os.path.exists(args.account_path):
        print(f"ERROR: account history not found at '{args.account_path}'")
        sys.exit(1)
        
    df_acct = pd.read_csv(args.account_path, parse_dates=["timestamp"])
    if getattr(args, "test_start", None):
        df_acct = df_acct[df_acct["timestamp"] >= args.test_start].copy()
    total_assets = df_acct["total_assets"]
    initial_value = total_assets.iloc[0]
    final_val = total_assets.iloc[-1]
    ret_pct = ((final_val - initial_value) / initial_value) * 100

    running_max = total_assets.cummax()
    drawdown_pct = ((total_assets - running_max) / running_max) * 100
    max_dd = drawdown_pct.min()
    ppo_sharpe = compute_sharpe(np.array(total_assets))
    ppo_sortino = compute_sortino(np.array(total_assets))

    # Process baseline metrics
    baseline_ret_pct = None
    baseline_max_dd = None
    bl_final = None
    bl_sharpe = None
    bl_sortino = None
    if args.baseline_path and os.path.exists(args.baseline_path):
        df_bl = pd.read_csv(args.baseline_path, parse_dates=["date"])
        if getattr(args, "test_start", None):
            df_bl = df_bl[df_bl["date"] >= args.test_start].copy()
        bl_initial = df_bl["total_assets"].iloc[0]
        bl_final = df_bl["total_assets"].iloc[-1]
        baseline_ret_pct = ((bl_final - bl_initial) / bl_initial) * 100
        bl_running_max = df_bl["total_assets"].cummax()
        bl_drawdown_pct = ((df_bl["total_assets"] - bl_running_max) / bl_running_max) * 100
        baseline_max_dd = bl_drawdown_pct.min()
        bl_sharpe = compute_sharpe(np.array(df_bl["total_assets"]))
        bl_sortino = compute_sortino(np.array(df_bl["total_assets"]))

    # Process fixed stoploss metrics
    fixed_sl_ret_pct = None
    fixed_sl_max_dd = None
    fsl_final = None
    fsl_sharpe = None
    fsl_sortino = None
    if args.fixed_sl_path and os.path.exists(args.fixed_sl_path):
        df_fsl = pd.read_csv(args.fixed_sl_path, parse_dates=["timestamp"])
        if getattr(args, "test_start", None):
            df_fsl = df_fsl[df_fsl["timestamp"] >= args.test_start].copy()
        fsl_initial = df_fsl["total_assets"].iloc[0]
        fsl_final = df_fsl["total_assets"].iloc[-1]
        fixed_sl_ret_pct = ((fsl_final - fsl_initial) / fsl_initial) * 100
        
        fsl_running_max = df_fsl["total_assets"].cummax()
        fsl_drawdown_pct = ((df_fsl["total_assets"] - fsl_running_max) / fsl_running_max) * 100
        fixed_sl_max_dd = fsl_drawdown_pct.min()
        fsl_sharpe = compute_sharpe(np.array(df_fsl["total_assets"]))
        fsl_sortino = compute_sortino(np.array(df_fsl["total_assets"]))

    # Process training dataset metrics
    train_tickers = []
    train_start_date = "N/A"
    train_end_date = "N/A"
    train_total_days = "N/A"
    if getattr(args, "train_data_path", None) and os.path.exists(args.train_data_path):
        df_train_data = pd.read_csv(args.train_data_path, parse_dates=["date"])
        if getattr(args, "train_start", None):
            df_train_data = df_train_data[df_train_data["date"] >= args.train_start].copy()
        train_tickers = sorted(df_train_data["tic"].unique())
        train_start_date = df_train_data["date"].min().strftime("%Y-%m-%d")
        train_end_date = df_train_data["date"].max().strftime("%Y-%m-%d")
        train_total_days = (df_train_data["date"].max() - df_train_data["date"].min()).days + 1

    # Process testing dataset metrics
    test_tickers = []
    test_start_date = "N/A"
    test_end_date = "N/A"
    test_total_days = "N/A"
    if getattr(args, "test_data_path", None) and os.path.exists(args.test_data_path):
        df_test_data = pd.read_csv(args.test_data_path, parse_dates=["date"])
        if getattr(args, "test_start", None):
            df_test_data = df_test_data[df_test_data["date"] >= args.test_start].copy()
        test_tickers = sorted(df_test_data["tic"].unique())
        test_start_date = df_test_data["date"].min().strftime("%Y-%m-%d")
        test_end_date = df_test_data["date"].max().strftime("%Y-%m-%d")
        test_total_days = (df_test_data["date"].max() - df_test_data["date"].min()).days + 1

    # Process training log metrics
    total_timesteps = "N/A"
    final_reward = "N/A"
    if args.log_path and os.path.exists(args.log_path):
        df_train = pd.read_csv(args.log_path)
        if not df_train.empty:
            total_timesteps = df_train["timesteps"].iloc[-1]
            final_reward = df_train["ep_rew_mean"].iloc[-1]

    # Plot Backtest
    bt_figs = plot_backtest(
        account_path=args.account_path,
        action_path=args.action_path,
        baseline_path=args.baseline_path,
        fixed_sl_path=args.fixed_sl_path,
        data_path=getattr(args, "test_data_path", None),
        return_figs=True,
        fixed_sl_ratio=args.fixed_stoploss_ratio,
        test_start=getattr(args, "test_start", None),
    )
    bt_b64s = []
    for fig in bt_figs:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            fig.savefig(tf.name, dpi=150, bbox_inches="tight")
            bt_b64s.append(encode_image_base64(tf.name))
        os.remove(tf.name)

    # Plot Training Curve
    tc_b64s = None
    if args.log_path and os.path.exists(args.log_path):
        tc_figs = plot_training_curve(
            log_path=args.log_path,
            rolling_window=5,
            return_figs=True
        )
        tc_b64s = []
        for fig in tc_figs:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                fig.savefig(tf.name, dpi=150, bbox_inches="tight")
                tc_b64s.append(encode_image_base64(tf.name))
            os.remove(tf.name)

    # Build HTML Report
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Strategy Evaluation Report</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --border-color: #e0e0e0;
            --text-color: #333;
            --bg-color: #f8f9fa;
        }}
        body {{
            font-family: 'Helvetica Neue', Arial, sans-serif;
            color: var(--text-color);
            background-color: var(--bg-color);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #fff;
            padding: 50px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid var(--border-color);
            border-radius: 8px;
        }}
        h1, h2, h3 {{
            color: var(--primary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        h1 {{
            text-align: center;
            border-bottom: none;
            margin-top: 10px;
            font-size: 32px;
            margin-bottom: 5px;
        }}
        .generation-date {{
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 40px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 15px;
        }}
        th, td {{
            border: 1px solid var(--border-color);
            padding: 12px 15px;
            text-align: left;
        }}
        th {{
            background-color: #fbfcfc;
            font-weight: 600;
        }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #c0392b; font-weight: bold; }}
        .graph-container {{
            text-align: center;
            margin: 40px 0;
            page-break-inside: avoid;
        }}
        .graph-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .caption {{
            font-style: italic;
            color: #7f8c8d;
            font-size: 13px;
            margin-top: 10px;
        }}
        .footer {{
            text-align: center;
            margin-top: 60px;
            font-size: 12px;
            color: #95a5a6;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }}
        @media print {{
            body {{
                background-color: #fff;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                border: none;
                margin: 0;
                padding: 0;
                max-width: 100%;
            }}
            .graph-container img {{
                box-shadow: none;
            }}
            @page {{
                margin: 1.5cm;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Strategy Evaluation Report</h1>
        <div class="generation-date">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

        <h2>1. Dataset & Environment Information</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Training Data</th>
                <th>Testing Data</th>
            </tr>
            <tr>
                <td><strong>Start Date</strong></td>
                <td>{train_start_date}</td>
                <td>{test_start_date}</td>
            </tr>
            <tr>
                <td><strong>End Date</strong></td>
                <td>{train_end_date}</td>
                <td>{test_end_date}</td>
            </tr>
            <tr>
                <td><strong>Total Days</strong></td>
                <td>{train_total_days}</td>
                <td>{test_total_days}</td>
            </tr>
            <tr>
                <td><strong>Tickers Evaluated</strong></td>
                <td>{", ".join(train_tickers) if train_tickers else "N/A"}</td>
                <td>{", ".join(test_tickers) if test_tickers else "N/A"}</td>
            </tr>
            <tr>
                <td><strong>Initial Capital</strong></td>
                <td>${args.train_initial_capital:,.2f}</td>
                <td>${initial_value:,.2f}</td>
            </tr>
        </table>

        <h2>2. Backtest Performance Metrics</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Final Portfolio</th>
                <th>Return</th>
                <th>Max Drawdown</th>
                <th>Sharpe Ratio</th>
                <th>Sortino Ratio</th>
            </tr>
            <tr>
                <td><strong>PPO Agent</strong></td>
                <td>${final_val:,.2f}</td>
                <td class="{'positive' if ret_pct >= 0 else 'negative'}">{ret_pct:+.2f}%</td>
                <td class="{'negative' if max_dd < 0 else ''}">{max_dd:.2f}%</td>
                <td>{ppo_sharpe:.4f}</td>
                <td>{ppo_sortino:.4f}</td>
            </tr>'''

    if fixed_sl_ret_pct is not None:
        html += f'''
            <tr>
                <td><strong>PPO Fixed SL ({args.fixed_stoploss_ratio:.0%})</strong></td>
                <td>${fsl_final:,.2f}</td>
                <td class="{'positive' if fixed_sl_ret_pct >= 0 else 'negative'}">{fixed_sl_ret_pct:+.2f}%</td>
                <td class="{'negative' if fixed_sl_max_dd < 0 else ''}">{fixed_sl_max_dd:.2f}%</td>
                <td>{fsl_sharpe:.4f}</td>
                <td>{fsl_sortino:.4f}</td>
            </tr>'''

    if baseline_ret_pct is not None:
        html += f'''
            <tr>
                <td><strong>Buy and Hold</strong></td>
                <td>${bl_final:,.2f}</td>
                <td class="{'positive' if baseline_ret_pct >= 0 else 'negative'}">{baseline_ret_pct:+.2f}%</td>
                <td class="{'negative' if baseline_max_dd < 0 else ''}">{baseline_max_dd:.2f}%</td>
                <td>{bl_sharpe:.4f}</td>
                <td>{bl_sortino:.4f}</td>
            </tr>'''

    html += f'''
        </table>

        <h2>3. Backtest Visualisation</h2>
        
        <h3>3.1. Portfolio Value Over Time</h3>
        <p class="caption">This graph shows the total portfolio assets over time, comparing the PPO agent against the buy-and-hold baseline and the fixed stop-loss strategy. The shaded regions denote periods of overall profit (green) and loss (red) relative to the initial capital.</p>
        <div class="graph-container">
            <img src="data:image/png;base64,{bt_b64s[0]}" alt="Portfolio Value Graph">
        </div>

        <h3>3.2. Maximum Drawdown</h3>
        <p class="caption">This graph visualises the maximum peak-to-trough drop in portfolio value as a percentage. It is useful for assessing the risk and historically worst-case scenario over the evaluated period.</p>
        <div class="graph-container">
            <img src="data:image/png;base64,{bt_b64s[1]}" alt="Max Drawdown Graph">
        </div>

        <h3>3.3. Ticker Prices and Trading Signals</h3>
        <p class="caption">This graph overlays the agent's buy/sell executions onto the individual asset prices. Green arrows indicate buys, red arrows indicate normal sells, and red cross marks denote forced stop-loss liquidations.</p>
        <div class="graph-container">
            <img src="data:image/png;base64,{bt_b64s[2]}" alt="Ticker Prices Graph">
        </div>'''

    if tc_b64s is not None:
        html += f'''
        <h2>4. Training Log Metrics</h2>
        <table>
            <tr><th>Total Timesteps</th><td>{total_timesteps}</td></tr>
            <tr><th>Final Episode Reward Mean</th><td>{final_reward:.4f}</td></tr>
        </table>

        <h2>5. Training Curve Visualisation</h2>
        
        <h3>5.1. Episode Reward Mean</h3>
        <p class="caption">This plot depicts the average reward achieved by the agent during training. A general upward trend signifies that the agent is learning and successfully optimising its trading strategy across episodes.</p>
        <div class="graph-container">
            <img src="data:image/png;base64,{tc_b64s[0]}" alt="Episode Reward Graph">
        </div>

        <h3>5.2. Policy and Value Loss</h3>
        <p class="caption">The policy loss represents the performance of the actor network, whilst the value loss indicates the accuracy of the critic network's state-value estimates. Both metrics help diagnose the stability of the neural network training.</p>
        <div class="graph-container">
            <img src="data:image/png;base64,{tc_b64s[1]}" alt="Policy and Value Loss Graph">
        </div>

        <h3>5.3. Entropy Loss and Approx KL</h3>
        <p class="caption">Entropy loss measures the randomness or exploration rate of the agent's actions, and Approx KL divergence tracks the size of the policy updates. These are pivotal hyperparameter tuning metrics to prevent premature convergence to sub-optimal policies.</p>
        <div class="graph-container">
            <img src="data:image/png;base64,{tc_b64s[2]}" alt="Entropy and KL Graph">
        </div>'''

    html += '''
        <div class="footer">
            Report produced by automated analysis tool.
        </div>
    </div>
</body>
</html>
'''

    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"Report successfully generated at: {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a static HTML report for training and backtesting.")
    parser.add_argument(
        "--account_path", type=str, required=True,
        help="Path to the account_history CSV from backtest.py"
    )
    parser.add_argument(
        "--action_path", type=str, required=True,
        help="Path to the action_history CSV from backtest.py"
    )
    parser.add_argument(
        "--log_path", type=str, required=True,
        help="Path to the training_log CSV produced by RewardLoggingCallback"
    )
    parser.add_argument(
        "--train_data_path", type=str, required=True,
        help="Path to training data CSV for environment/ticker information"
    )
    parser.add_argument(
        "--test_data_path", type=str, required=True,
        help="Path to original data CSV for testing environment/ticker information"
    )
    parser.add_argument(
        "--baseline_path", type=str, required=True,
        help="Path to baseline_buy_and_hold_account_history.csv"
    )
    parser.add_argument(
        "--fixed_sl_path", type=str, required=True,
        help="Path to fixed_stoploss_account_history.csv"
    )
    parser.add_argument(
        "--train_initial_capital", type=float, default=1000000.0,
        help="Initial capital for the training dataset. (Default: 1,000,000)"
    )
    parser.add_argument(
        "--output_path", type=str, default="report.html",
        help="Path to save the generated HTML report"
    )
    parser.add_argument(
        "--fixed_stoploss_ratio", type=float, default=0.95,
        help="Fixed stop-loss ratio used during backtesting (default: 0.95)"
    )
    parser.add_argument(
        "--test_start", type=str, default=None,
        help="Start date/time for reporting actually begins (excludes warmup from report)"
    )
    parser.add_argument(
        "--train_start", type=str, default=None,
        help="Start date/time for training reporting actually begins (excludes warmup from report for training data)"
    )

    args = parser.parse_args()
    generate_report(args)


if __name__ == "__main__":
    main()
