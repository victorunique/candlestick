"""Tests for src.run_pipeline – hyperparameter tuning pipeline."""

import json
import os
import subprocess
import sys
import tempfile

import pytest

from src.run_pipeline import (
    generate_combinations,
    build_commands,
    parse_plaintext_output,
    append_result_row,
    RESULTS_CSV_HEADER,
)


# ---------------------------------------------------------------------------
# generate_combinations
# ---------------------------------------------------------------------------

class TestGenerateCombinations:
    """Cartesian product logic."""

    def test_single_element_per_list(self):
        """One element in each list → exactly one combination."""
        date_ranges = [("2020-01-01", "2020-06-30", "2020-07-01", "2020-12-31")]
        ticker_lists = [["AAPL"]]
        total_timesteps = [10000]
        reward_pnl = [1.0]
        reward_dd = [0.5]
        n_steps = [2048]
        ent_coef = [0.01]
        lr = [0.00025]
        gamma = [0.99]
        episode_length = [1000]

        combos = generate_combinations(
            date_ranges=date_ranges,
            ticker_lists=ticker_lists,
            total_timesteps_list=total_timesteps,
            reward_weight_pnl_list=reward_pnl,
            reward_weight_drawdown_list=reward_dd,
            cash_penalty_proportion_list=[0.05],
            upside_pnl_multiplier_list=[1.0],
            stoploss_min_list=[0.90],
            stoploss_max_list=[0.95],
            n_steps_list=n_steps,
            ent_coef_list=ent_coef,
            learning_rate_list=lr,
            gamma_list=gamma,
            episode_length_list=episode_length,
        )

        assert len(combos) == 1
        c = combos[0]
        assert c["train_start"] == "2020-01-01"
        assert c["train_end"] == "2020-06-30"
        assert c["test_start"] == "2020-07-01"
        assert c["test_end"] == "2020-12-31"
        assert c["tickers"] == ["AAPL"]
        assert c["total_timesteps"] == 10000
        assert c["stoploss_min"] == 0.90
        assert c["stoploss_max"] == 0.95

    def test_cartesian_product_count(self):
        """Two date ranges × two ticker lists × rest single → 4 combos."""
        date_ranges = [
            ("2020-01-01", "2020-06-30", "2020-07-01", "2020-12-31"),
            ("2021-01-01", "2021-06-30", "2021-07-01", "2021-12-31"),
        ]
        ticker_lists = [["AAPL"], ["AAPL", "MSFT"]]
        total_timesteps = [10000]
        reward_pnl = [1.0]
        reward_dd = [0.5]
        n_steps = [2048]
        ent_coef = [0.01]
        lr = [0.00025]
        gamma = [0.99]
        episode_length = [1000]

        combos = generate_combinations(
            date_ranges=date_ranges,
            ticker_lists=ticker_lists,
            total_timesteps_list=total_timesteps,
            reward_weight_pnl_list=reward_pnl,
            reward_weight_drawdown_list=reward_dd,
            cash_penalty_proportion_list=[0.05],
            upside_pnl_multiplier_list=[1.0],
            stoploss_min_list=[0.90],
            stoploss_max_list=[0.95],
            n_steps_list=n_steps,
            ent_coef_list=ent_coef,
            learning_rate_list=lr,
            gamma_list=gamma,
            episode_length_list=episode_length,
        )

        assert len(combos) == 4

    def test_combo_keys_present(self):
        """Every combo dict must contain all expected keys."""
        combos = generate_combinations(
            date_ranges=[("2020-01-01", "2020-06-30", "2020-07-01", "2020-12-31")],
            ticker_lists=[["AAPL"]],
            total_timesteps_list=[10000],
            reward_weight_pnl_list=[1.0],
            reward_weight_drawdown_list=[0.5],
            cash_penalty_proportion_list=[0.05],
            upside_pnl_multiplier_list=[1.0],
            stoploss_min_list=[0.90],
            stoploss_max_list=[0.95],
            n_steps_list=[2048],
            ent_coef_list=[0.01],
            learning_rate_list=[0.00025],
            gamma_list=[0.99],
            episode_length_list=[1000],
        )

        expected_keys = {
            "train_start", "train_end", "test_start", "test_end",
            "tickers", "total_timesteps",
            "reward_weight_pnl", "reward_weight_drawdown",
            "cash_penalty_proportion", "upside_pnl_multiplier",
            "stoploss_min", "stoploss_max",
            "n_steps", "ent_coef", "learning_rate", "gamma",
            "episode_length",
        }
        assert set(combos[0].keys()) == expected_keys


# ---------------------------------------------------------------------------
# build_commands
# ---------------------------------------------------------------------------

class TestBuildCommands:
    """Verify generated shell commands contain the right arguments."""

    def _make_combo(self):
        return {
            "train_start": "2020-01-01",
            "train_end": "2020-06-30",
            "test_start": "2020-07-01",
            "test_end": "2020-12-31",
            "tickers": ["AAPL", "MSFT"],
            "total_timesteps": 10000,
            "reward_weight_pnl": 1.0,
            "reward_weight_drawdown": 0.5,
            "cash_penalty_proportion": 0.05,
            "upside_pnl_multiplier": 1.0,
            "stoploss_min": 0.90,
            "stoploss_max": 0.95,
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "gamma": 0.99,
            "episode_length": 1000,
        }

    def test_returns_six_commands(self):
        """Pipeline should produce exactly 6 commands."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        assert len(cmds) == 6

    def test_data_fetcher_train_uses_interval_1m(self):
        """Fixed interval=1m must appear in the training data_fetcher command."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        cmd_str = " ".join(cmds[0])
        assert "--interval" in cmd_str
        assert "1m" in cmd_str

    def test_data_fetcher_test_uses_interval_1m(self):
        """Fixed interval=1m must appear in the test data_fetcher command."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        cmd_str = " ".join(cmds[2])
        assert "--interval" in cmd_str
        assert "1m" in cmd_str

    def test_train_ppo_has_seed(self):
        """Training command must include --seed 42."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        cmd_str = " ".join(cmds[4])
        assert "--seed" in cmd_str
        assert "42" in cmd_str

    def test_backtest_has_plaintext(self):
        """Backtest command must include --plaintext."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        cmd_str = " ".join(cmds[5])
        assert "--plaintext" in cmd_str

    def test_train_ppo_has_train_start(self):
        """Training command must include --train_start."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        cmd_str = " ".join(cmds[4])
        assert "--train_start" in cmd_str
        assert "2020-01-01" in cmd_str

    def test_backtest_has_test_start(self):
        """Backtest command must include --test_start."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        cmd_str = " ".join(cmds[5])
        assert "--test_start" in cmd_str
        assert "2020-07-01" in cmd_str

    def test_backtest_has_fixed_stoploss(self):
        """Backtest command must include --fixed_stoploss_ratio 0.95."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        cmd_str = " ".join(cmds[5])
        assert "--fixed_stoploss_ratio" in cmd_str
        assert "0.95" in cmd_str

    def test_train_ppo_has_reward_weights(self):
        """Training command must include reward weight arguments."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        cmd_str = " ".join(cmds[4])
        assert "--reward_weight_pnl" in cmd_str
        assert "--reward_weight_drawdown" in cmd_str

    def test_tickers_in_data_fetcher(self):
        """Tickers must appear in data_fetcher commands."""
        cmds = build_commands(self._make_combo(), work_dir="/tmp/test")
        cmd_str = " ".join(cmds[0])
        assert "AAPL" in cmd_str
        assert "MSFT" in cmd_str

    def test_warmup_start_date_for_test_data(self):
        """Test data fetcher should use a date 5 days before test_start_date."""
        combo = self._make_combo()
        combo["test_start"] = "2020-07-01"
        cmds = build_commands(combo, work_dir="/tmp/test")
        
        # Test Data Fetcher Command -> Index 2
        fetch_cmd_str = " ".join(cmds[2])
        # Test start is "2020-07-01", minus 5 days is "2020-06-26"
        assert "--start_date 2020-06-26" in fetch_cmd_str

        # Feature Engineer Test Command -> Index 3
        fe_cmd_str = " ".join(cmds[3])
        # feature engineer no longer receives start_date
        assert "--start_date" not in fe_cmd_str

    def test_warmup_start_date_for_train_data(self):
        """Train data fetcher should use a date 5 days before train_start_date."""
        combo = self._make_combo()
        combo["train_start"] = "2020-01-01"
        cmds = build_commands(combo, work_dir="/tmp/test")
        
        # Train Data Fetcher Command -> Index 0
        fetch_cmd_str = " ".join(cmds[0])
        # Train start is "2020-01-01", minus 5 days is "2019-12-27"
        assert "--start_date 2019-12-27" in fetch_cmd_str

        # Feature Engineer Train Command -> Index 1
        fe_cmd_str = " ".join(cmds[1])
        # feature engineer no longer receives start_date
        assert "--start_date" not in fe_cmd_str


# ---------------------------------------------------------------------------
# parse_plaintext_output
# ---------------------------------------------------------------------------

class TestParsePlaintextOutput:
    """Parse the --plaintext CSV line from backtest stdout."""

    def test_valid_output(self):
        """Standard 6-value plaintext output."""
        raw = "-0.0016,-0.0069,-0.0163,-0.0228,-0.0089,-0.0158"
        result = parse_plaintext_output(raw)
        assert len(result) == 6
        assert result["ppo_return"] == pytest.approx(-0.0016)
        assert result["ppo_max_dd"] == pytest.approx(-0.0069)
        assert result["fsl_return"] == pytest.approx(-0.0163)
        assert result["fsl_max_dd"] == pytest.approx(-0.0228)
        assert result["bh_return"] == pytest.approx(-0.0089)
        assert result["bh_max_dd"] == pytest.approx(-0.0158)

    def test_strips_whitespace(self):
        """Leading/trailing whitespace and newlines are ignored."""
        raw = "  0.05,  -0.01, 0.03, -0.02, 0.04, -0.015\n"
        result = parse_plaintext_output(raw)
        assert result["ppo_return"] == pytest.approx(0.05)

    def test_invalid_output_raises(self):
        """Non-numeric or wrong field count raises ValueError."""
        with pytest.raises(ValueError):
            parse_plaintext_output("foo,bar,baz")


# ---------------------------------------------------------------------------
# append_result_row / CSV accumulation
# ---------------------------------------------------------------------------

class TestResultsCsvAccumulation:
    """Incremental CSV writing."""

    def _make_combo(self, idx=1):
        return {
            "train_start": "2020-01-01",
            "train_end": "2020-06-30",
            "test_start": "2020-07-01",
            "test_end": "2020-12-31",
            "tickers": ["AAPL"],
            "total_timesteps": 10000,
            "reward_weight_pnl": 1.0,
            "reward_weight_drawdown": 0.5,
            "cash_penalty_proportion": 0.05,
            "upside_pnl_multiplier": 1.0,
            "stoploss_min": 0.90,
            "stoploss_max": 0.95,
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "gamma": 0.99,
            "episode_length": 1000,
        }

    def _make_metrics(self):
        return {
            "ppo_return": -0.0016,
            "ppo_max_dd": -0.0069,
            "fsl_return": -0.0163,
            "fsl_max_dd": -0.0228,
            "bh_return": -0.0089,
            "bh_max_dd": -0.0158,
        }

    def test_creates_file_with_header(self):
        """First call creates file and writes header + one data row."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        # Remove so append_result_row creates fresh
        os.remove(path)

        try:
            append_result_row(path, combo_id=1, combo=self._make_combo(), metrics=self._make_metrics())
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # header + 1 data row
            assert lines[0].strip() == ",".join(RESULTS_CSV_HEADER)
        finally:
            os.remove(path)

    def test_appends_without_duplicating_header(self):
        """Second call appends data row without re-writing the header."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        os.remove(path)

        try:
            append_result_row(path, combo_id=1, combo=self._make_combo(), metrics=self._make_metrics())
            append_result_row(path, combo_id=2, combo=self._make_combo(2), metrics=self._make_metrics())
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 data rows
        finally:
            os.remove(path)


# ---------------------------------------------------------------------------
# --dry-run CLI integration
# ---------------------------------------------------------------------------

class TestDryRunCli:
    """Running the script with --dry-run should not execute any pipeline steps."""

    def test_dry_run_prints_combos_and_exits(self):
        """--dry-run should succeed, print combo info, and not create temp dirs."""
        config_data = {
            "date_ranges": [["2020-01-01", "2020-06-30", "2020-07-01", "2020-12-31"]],
            "ticker_lists": [["AAPL"]],
            "total_timesteps_list": [10000],
            "reward_weight_pnl_list": [1.0],
            "reward_weight_drawdown_list": [0.5],
            "cash_penalty_proportion_list": [0.05],
            "upside_pnl_multiplier_list": [1.0],
            "stoploss_min_list": [0.90],
            "stoploss_max_list": [0.95],
            "n_steps_list": [2048],
            "ent_coef_list": [0.01],
            "learning_rate_list": [0.00025],
            "gamma_list": [0.99],
            "episode_length_list": [1000]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.run_pipeline", "--dry-run", "--config", config_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )
            assert result.returncode == 0
            # Should contain at least one "Combo" line
            assert "Combo" in result.stdout or "combo" in result.stdout.lower()
            # Should contain "uv run" commands
            assert "uv run" in result.stdout
        finally:
            os.remove(config_path)
