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
import os
import shutil
import subprocess
import tempfile


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETER GRID — edit these lists to define the search space
# ═══════════════════════════════════════════════════════════════════════════

# Each tuple: (train_start, train_end, test_start, test_end)
DATE_RANGES = [
    ("2026-02-06", "2026-02-13", "2026-02-13", "2026-02-14"),
    ("2026-02-13", "2026-02-20", "2026-02-20", "2026-02-21"),
    ("2026-02-20", "2026-02-27", "2026-02-27", "2026-02-28"),
    ("2026-02-09", "2026-02-14", "2026-02-16", "2026-02-17"),
    ("2026-02-16", "2026-02-21", "2026-02-23", "2026-02-24"),
    ("2026-02-23", "2026-02-28", "2026-03-02", "2026-03-03"),
    ("2026-02-11", "2026-02-18", "2026-02-18", "2026-02-19"),
    ("2026-02-18", "2026-02-25", "2026-02-25", "2026-02-26"),
    ("2026-02-25", "2026-03-04", "2026-03-04", "2026-03-05"),
]

# Each element is a list of tickers (same list used for train & test)
TICKER_LISTS = [
    ["SPY"],
    ["AAPL", "MSFT"],
]

TOTAL_TIMESTEPS_LIST = [300000]

REWARD_WEIGHT_PNL_LIST = [1.0]
REWARD_WEIGHT_DRAWDOWN_LIST = [0.2]

N_STEPS_LIST = [2048]
ENT_COEF_LIST = [0.01, 0.03]
LEARNING_RATE_LIST = [0.0001]
GAMMA_LIST = [0.99, 0.95]
EPISODE_LENGTH_LIST = [1000]


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


def build_commands(combo: dict, work_dir: str) -> list[list[str]]:
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

    cmds = []

    # 0: fetch training data
    cmds.append([
        "uv", "run", "python", "-m", "src.data_fetcher",
        "--start_date", combo["train_start"],
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
    ])

    # 2: fetch test data
    cmds.append([
        "uv", "run", "python", "-m", "src.data_fetcher",
        "--start_date", combo["test_start"],
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
        "--output", type=str, default="./results/hparam_results.csv",
        help="Path to the incremental results CSV (default: ./results/hparam_results.csv).",
    )
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="1-based combo index to start from (skip earlier combos).",
    )

    args = parser.parse_args()

    combos = generate_combinations(
        date_ranges=DATE_RANGES,
        ticker_lists=TICKER_LISTS,
        total_timesteps_list=TOTAL_TIMESTEPS_LIST,
        reward_weight_pnl_list=REWARD_WEIGHT_PNL_LIST,
        reward_weight_drawdown_list=REWARD_WEIGHT_DRAWDOWN_LIST,
        n_steps_list=N_STEPS_LIST,
        ent_coef_list=ENT_COEF_LIST,
        learning_rate_list=LEARNING_RATE_LIST,
        gamma_list=GAMMA_LIST,
        episode_length_list=EPISODE_LENGTH_LIST,
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
        cmds = build_commands(combo, work_dir=work_dir)

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
