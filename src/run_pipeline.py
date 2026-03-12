"""Hyperparameter tuning pipeline for candlestick trading agent.

Automates the 4-step end-to-end training & backtesting pipeline with
grid search over user-specified parameter lists.  Edit the *_LIST constants
below to define the search grid, then run:

    uv run python -m src.run_pipeline --dry-run     # preview combos/commands
    uv run python -m src.run_pipeline               # execute all combos

Results are appended incrementally to a CSV file for later analysis.
"""

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER GRID — loaded from JSON configuration file
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# FIXED PARAMETERS (not part of the grid)
# ═══════════════════════════════════════════════════════════════════════════

FIXED_INTERVAL = "1m"
FIXED_SEED = 42
FIXED_STOPLOSS_RATIO = 0.95
FIXED_INITIAL_CASH = 1000000  # used only for documentation; hard-coded in tools

MODEL_NAME = "hparam_ppo"


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS CSV SCHEMA
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_CSV_HEADER = [
    "combo_id",
    "train_start", "train_end", "test_start", "test_end",
    "tickers", "total_timesteps",
    "reward_weight_pnl", "reward_weight_drawdown",
    "n_steps", "ent_coef", "learning_rate", "gamma", "episode_length",
    "ppo_return", "ppo_max_dd",
    "fsl_return", "fsl_max_dd",
    "bh_return", "bh_max_dd",
]

PLAINTEXT_KEYS = [
    "ppo_return", "ppo_max_dd",
    "fsl_return", "fsl_max_dd",
    "bh_return", "bh_max_dd",
]


# ═══════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def generate_combinations(
    date_ranges: list,
    ticker_lists: list,
    total_timesteps_list: list,
    reward_weight_pnl_list: list,
    reward_weight_drawdown_list: list,
    n_steps_list: list,
    ent_coef_list: list,
    learning_rate_list: list,
    gamma_list: list,
    episode_length_list: list,
) -> list[dict]:
    """Return the Cartesian product of all parameter lists as a list of dicts."""
    combos = []
    for (
        (train_start, train_end, test_start, test_end),
        tickers,
        total_ts,
        rw_pnl,
        rw_dd,
        n_st,
        ent,
        lr,
        gam,
        ep_len,
    ) in itertools.product(
        date_ranges,
        ticker_lists,
        total_timesteps_list,
        reward_weight_pnl_list,
        reward_weight_drawdown_list,
        n_steps_list,
        ent_coef_list,
        learning_rate_list,
        gamma_list,
        episode_length_list,
    ):
        combos.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "tickers": list(tickers),
            "total_timesteps": total_ts,
            "reward_weight_pnl": rw_pnl,
            "reward_weight_drawdown": rw_dd,
            "n_steps": n_st,
            "ent_coef": ent,
            "learning_rate": lr,
            "gamma": gam,
            "episode_length": ep_len,
        })
    return combos


def build_commands(combo: dict, work_dir: str, use_local_history: bool = False, local_history_dir: str = "./history") -> list[list[str]]:
    """Build the 6 subprocess commands for a single combo.

    Returns a list of 6 command lists:
      0: data_fetcher  (train)
      1: feature_engineer (train)
      2: data_fetcher  (test)
      3: feature_engineer (test)
      4: train_ppo
      5: backtest (--plaintext)
    """
    train_raw = os.path.join(work_dir, "data_training.csv")
    train_preprocessed = os.path.join(work_dir, "data_training_preprocessed.csv")
    test_raw = os.path.join(work_dir, "data_testing.csv")
    test_preprocessed = os.path.join(work_dir, "data_testing_preprocessed.csv")
    model_dir = os.path.join(work_dir, "trained_models")
    results_dir = os.path.join(work_dir, "results")
    model_path = os.path.join(model_dir, MODEL_NAME)

    tickers = combo["tickers"]

    test_start_dt = datetime.strptime(combo["test_start"], "%Y-%m-%d")
    test_warmup_start_dt = test_start_dt - timedelta(days=5)
    test_warmup_start_str = test_warmup_start_dt.strftime("%Y-%m-%d")

    train_start_dt = datetime.strptime(combo["train_start"], "%Y-%m-%d")
    train_warmup_start_dt = train_start_dt - timedelta(days=5)
    train_warmup_start_str = train_warmup_start_dt.strftime("%Y-%m-%d")

    cmds = []

    # 0: fetch training data
    if use_local_history:
        cmds.append([
            "uv", "run", "python", "-m", "src.data_loader",
            "--start_date", train_warmup_start_str,
            "--end_date", combo["train_end"],
            "--ticker_list", *tickers,
            "--data_dir", local_history_dir,
            "--output_path", train_raw
        ])
    else:
        cmds.append([
            "uv", "run", "python", "-m", "src.data_fetcher",
            "--start_date", train_warmup_start_str,
            "--end_date", combo["train_end"],
            "--ticker_list", *tickers,
            "--output_path", train_raw,
            "--interval", FIXED_INTERVAL,
        ])

    # 1: feature engineer training data
    cmds.append([
        "uv", "run", "python", "-m", "src.feature_engineer",
        "--input_path", train_raw,
        "--output_path", train_preprocessed,
        "--start_date", combo["train_start"],
    ])

    # 2: fetch test data
    if use_local_history:
        cmds.append([
            "uv", "run", "python", "-m", "src.data_loader",
            "--start_date", test_warmup_start_str,
            "--end_date", combo["test_end"],
            "--ticker_list", *tickers,
            "--data_dir", local_history_dir,
            "--output_path", test_raw
        ])
    else:
        cmds.append([
            "uv", "run", "python", "-m", "src.data_fetcher",
            "--start_date", test_warmup_start_str,
            "--end_date", combo["test_end"],
            "--ticker_list", *tickers,
            "--output_path", test_raw,
            "--interval", FIXED_INTERVAL,
        ])

    # 3: feature engineer test data
    cmds.append([
        "uv", "run", "python", "-m", "src.feature_engineer",
        "--input_path", test_raw,
        "--output_path", test_preprocessed,
        "--start_date", combo["test_start"],
    ])

    # 4: train PPO
    cmds.append([
        "uv", "run", "python", "-m", "src.train_ppo",
        "--data_path", train_preprocessed,
        "--model_dir", model_dir,
        "--model_name", MODEL_NAME,
        "--total_timesteps", str(combo["total_timesteps"]),
        "--seed", str(FIXED_SEED),
        "--n_steps", str(combo["n_steps"]),
        "--ent_coef", str(combo["ent_coef"]),
        "--learning_rate", str(combo["learning_rate"]),
        "--gamma", str(combo["gamma"]),
        "--episode_length", str(combo["episode_length"]),
        "--reward_weight_pnl", str(combo["reward_weight_pnl"]),
        "--reward_weight_drawdown", str(combo["reward_weight_drawdown"]),
    ])

    # 5: backtest with --plaintext
    cmds.append([
        "uv", "run", "python", "-m", "src.backtest",
        "--data_path", test_preprocessed,
        "--model_path", model_path,
        "--results_dir", results_dir,
        "--plaintext",
        "--fixed_stoploss_ratio", str(FIXED_STOPLOSS_RATIO),
    ])

    return cmds


def parse_plaintext_output(raw: str) -> dict:
    """Parse the comma-separated plaintext line from backtest into a dict.

    Expected format: ``ppo_ret,ppo_dd,fsl_ret,fsl_dd,bh_ret,bh_dd``
    """
    parts = [p.strip() for p in raw.strip().split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"Expected 6 comma-separated values from --plaintext output, "
            f"got {len(parts)}: {raw!r}"
        )
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Non-numeric value in --plaintext output: {raw!r}") from exc

    return dict(zip(PLAINTEXT_KEYS, values))


def append_result_row(
    output_path: str,
    combo_id: int,
    combo: dict,
    metrics: dict,
) -> None:
    """Append one result row to the CSV, creating the file + header if needed."""
    write_header = not os.path.exists(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(RESULTS_CSV_HEADER)
        row = [
            combo_id,
            combo["train_start"], combo["train_end"],
            combo["test_start"], combo["test_end"],
            " ".join(combo["tickers"]),
            combo["total_timesteps"],
            combo["reward_weight_pnl"], combo["reward_weight_drawdown"],
            combo["n_steps"], combo["ent_coef"],
            combo["learning_rate"], combo["gamma"],
            combo["episode_length"],
            metrics["ppo_return"], metrics["ppo_max_dd"],
            metrics["fsl_return"], metrics["fsl_max_dd"],
            metrics["bh_return"], metrics["bh_max_dd"],
        ]
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning pipeline for candlestick trading agent."
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print all combos and commands without executing anything.",
    )
    parser.add_argument(
        "--config", type=str, default="run_pipeline_config.json",
        help="Path to the JSON configuration file containing hyperparameters (default: run_pipeline_config.json).",
    )
    parser.add_argument(
        "--output", type=str, default="./results/hparam_results.csv",
        help="Path to the incremental results CSV (default: ./results/hparam_results.csv).",
    )
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="1-based combo index to start from (skip earlier combos).",
    )
    parser.add_argument(
        "--use-local-history", action="store_true", default=False,
        help="Load raw training and testing data directly from a local folder instead of Yahoo Finance API.",
    )
    parser.add_argument(
        "--local-history-dir", type=str, default="./history",
        help="Path to the local history folder (default: ./history).",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = json.load(f)

    combos = generate_combinations(
        date_ranges=config["date_ranges"],
        ticker_lists=config["ticker_lists"],
        total_timesteps_list=config.get("total_timesteps_list", [300000]),
        reward_weight_pnl_list=config.get("reward_weight_pnl_list", [1.0]),
        reward_weight_drawdown_list=config.get("reward_weight_drawdown_list", [0.2]),
        n_steps_list=config.get("n_steps_list", [2048]),
        ent_coef_list=config.get("ent_coef_list", [0.01]),
        learning_rate_list=config.get("learning_rate_list", [0.0001]),
        gamma_list=config.get("gamma_list", [0.99]),
        episode_length_list=config.get("episode_length_list", [1000]),
    )

    total = len(combos)
    print(f"Total combinations: {total}")

    # Determine project root (where pyproject.toml lives)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for idx, combo in enumerate(combos, start=1):
        if idx < args.start_from:
            continue

        print(f"\n{'=' * 72}")
        print(f"Combo {idx}/{total}")
        print(f"{'=' * 72}")
        for key, val in combo.items():
            print(f"  {key}: {val}")

        # Build commands using a placeholder work_dir for dry-run display
        work_dir = f"<temp_dir_combo_{idx}>" if args.dry_run else tempfile.mkdtemp(prefix=f"candlestick_combo{idx}_")
        cmds = build_commands(
            combo,
            work_dir=work_dir,
            use_local_history=args.use_local_history,
            local_history_dir=args.local_history_dir,
        )

        step_labels = [
            "Fetch training data",
            "Feature-engineer training data",
            "Fetch test data",
            "Feature-engineer test data",
            "Train PPO",
            "Backtest (--plaintext)",
        ]

        if args.dry_run:
            for label, cmd in zip(step_labels, cmds):
                print(f"\n  [{label}]")
                print(f"    {' '.join(cmd)}")
            continue

        # ── Execute the pipeline ──────────────────────────────────────
        print(f"  Work directory: {work_dir}")
        failed = False

        for step_idx, (label, cmd) in enumerate(zip(step_labels, cmds)):
            print(f"\n  [{label}] ...")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                )
                if result.returncode != 0:
                    print(f"  ✗ {label} FAILED (exit code {result.returncode})")
                    print(f"    stderr: {result.stderr[:500]}")
                    failed = True
                    break
                print(f"  ✓ {label} OK")

                # Capture plaintext output from backtest (last step)
                if step_idx == 5:
                    # The plaintext CSV line may not be the last line
                    # (backtest prints other messages after it).
                    # Scan stdout lines in reverse for a valid 6-float CSV.
                    plaintext_line = None
                    for line in reversed(result.stdout.strip().split("\n")):
                        parts = line.strip().split(",")
                        if len(parts) == 6:
                            try:
                                [float(p) for p in parts]
                                plaintext_line = line.strip()
                                break
                            except ValueError:
                                continue
                    if plaintext_line is None:
                        print("  ✗ Could not find plaintext CSV in backtest output")
                        print(f"    stdout: {result.stdout[:500]}")
                        failed = True
                        break

            except Exception as exc:
                print(f"  ✗ {label} EXCEPTION: {exc}")
                failed = True
                break

        if not failed:
            try:
                metrics = parse_plaintext_output(plaintext_line)
                append_result_row(args.output, combo_id=idx, combo=combo, metrics=metrics)
                print(f"\n  ✓ Results appended to {args.output}")
                print(f"    ppo_return={metrics['ppo_return']}, ppo_max_dd={metrics['ppo_max_dd']}")
            except (ValueError, NameError) as exc:
                print(f"  ✗ Failed to parse backtest output: {exc}")
        else:
            print(f"  ⚠ Combo {idx} failed — skipping to next combo.")

        # ── Cleanup temp directory ────────────────────────────────────
        try:
            shutil.rmtree(work_dir)
            print(f"  🧹 Cleaned up {work_dir}")
        except Exception as exc:
            print(f"  ⚠ Failed to clean up {work_dir}: {exc}")

    if args.dry_run:
        print(f"\n{'=' * 72}")
        print("Dry run complete. No commands were executed.")
    else:
        print(f"\n{'=' * 72}")
        print(f"All combinations done. Results at: {args.output}")


if __name__ == "__main__":
    main()
