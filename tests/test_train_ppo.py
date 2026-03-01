import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

from src.train_ppo import train_ppo, set_seeds, main


@pytest.fixture
def sample_preprocessed_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    data = []
    
    for i, date in enumerate(dates):
        # Generate dummy data for two assets
        for tic in ["AAPL", "MSFT"]:
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": tic,
                "open": 100 + i,
                "high": 105 + i,
                "low": 95 + i,
                "close": 100 + i,
                "volume": 1000,
                "macd": 1,
                "rsi_30": 50,
                "cci_30": 100,
                "dx_30": 20
            })
            
    df = pd.DataFrame(data)
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    return df
    
def test_train_ppo(sample_preprocessed_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        val_df = sample_preprocessed_data.copy()
        
        model_name = "test_ppo_model"
        
        # Train for very few timesteps just to verify the pipeline doesn't crash
        model, env_normalized = train_ppo(
            df=sample_preprocessed_data,
            total_timesteps=10,
            model_dir=temp_dir,
            model_name=model_name,
            indicators=["macd", "rsi_30", "cci_30", "dx_30"],
            window_size=10,
            ent_coef=0.01,
            learning_rate=0.00025,
            gamma=0.99
        )
        
        assert isinstance(model, PPO)
        
        # Check that files were saved
        assert os.path.exists(os.path.join(temp_dir, f"{model_name}.zip"))
        assert os.path.exists(os.path.join(temp_dir, f"{model_name}_vecnormalize.pkl"))


class TestVecWrapperOrder:
    """Bug 1: VecNormalize must wrap VecFrameStack (not the other way around).
    
    The correct order is DummyVecEnv → VecFrameStack → VecNormalize so that
    normalization statistics match the stacked observation shape the model sees.
    """

    def test_returned_env_is_vecnormalize(self, sample_preprocessed_data):
        """The outermost wrapper returned by train_ppo should be VecNormalize."""
        with tempfile.TemporaryDirectory() as temp_dir:
            _, env = train_ppo(
                df=sample_preprocessed_data,
                total_timesteps=10,
                model_dir=temp_dir,
                model_name="wrapper_order_test",
                indicators=["macd", "rsi_30", "cci_30", "dx_30"],
                window_size=10,
            )
            assert isinstance(env, VecNormalize), (
                f"Expected outermost wrapper to be VecNormalize, got {type(env).__name__}"
            )

    def test_vecnormalize_wraps_vecframestack(self, sample_preprocessed_data):
        """VecNormalize.venv should be a VecFrameStack (correct inner wrapper)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            _, env = train_ppo(
                df=sample_preprocessed_data,
                total_timesteps=10,
                model_dir=temp_dir,
                model_name="wrapper_inner_test",
                indicators=["macd", "rsi_30", "cci_30", "dx_30"],
                window_size=10,
            )
            inner = env.venv  # VecNormalize's inner env
            assert isinstance(inner, VecFrameStack), (
                f"Expected VecNormalize to wrap VecFrameStack, but inner is {type(inner).__name__}"
            )

    def test_normalization_stats_match_stacked_obs(self, sample_preprocessed_data):
        """VecNormalize running mean shape should match the stacked observation shape."""
        window_size = 10
        with tempfile.TemporaryDirectory() as temp_dir:
            _, env = train_ppo(
                df=sample_preprocessed_data,
                total_timesteps=10,
                model_dir=temp_dir,
                model_name="stats_shape_test",
                indicators=["macd", "rsi_30", "cci_30", "dx_30"],
                window_size=window_size,
            )
            obs_shape = env.observation_space.shape
            norm_mean_shape = env.obs_rms.mean.shape
            assert norm_mean_shape == obs_shape, (
                f"VecNormalize stats shape {norm_mean_shape} != observation shape {obs_shape}. "
                "This indicates VecNormalize was applied before VecFrameStack."
            )


class TestSeedReproducibility:
    """Bug 2: seed must be propagated to PPO for full reproducibility."""

    def _get_model_params(self, model):
        """Extract a flat array of all model parameters for comparison."""
        params = []
        for p in model.policy.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def test_same_seed_produces_same_weights(self, sample_preprocessed_data):
        """Training twice with the same seed should produce identical model weights."""
        common_kwargs = dict(
            df=sample_preprocessed_data,
            total_timesteps=512,
            model_name="seed_test",
            indicators=["macd", "rsi_30", "cci_30", "dx_30"],
            window_size=10,
        )

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            set_seeds(42)
            m1, _ = train_ppo(**common_kwargs, model_dir=d1, seed=42)
            set_seeds(42)
            m2, _ = train_ppo(**common_kwargs, model_dir=d2, seed=42)

            p1 = self._get_model_params(m1)
            p2 = self._get_model_params(m2)
            np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=1e-5,
                err_msg="Same seed should produce identical weights")

    def test_different_seed_produces_different_weights(self, sample_preprocessed_data):
        """Training with different seeds should produce different model weights."""
        common_kwargs = dict(
            df=sample_preprocessed_data,
            total_timesteps=512,
            model_name="seed_test",
            indicators=["macd", "rsi_30", "cci_30", "dx_30"],
            window_size=10,
        )

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            set_seeds(42)
            m1, _ = train_ppo(**common_kwargs, model_dir=d1, seed=42)
            set_seeds(99)
            m2, _ = train_ppo(**common_kwargs, model_dir=d2, seed=99)

            p1 = self._get_model_params(m1)
            p2 = self._get_model_params(m2)
            assert not np.allclose(p1, p2, rtol=1e-5, atol=1e-5), (
                "Different seeds should produce different weights"
            )


class TestCLIArgsForwarded:
    """Bug 3: CLI main() must expose and forward hmax, stoploss_penalty,
    profit_loss_ratio, cash_penalty, and seed to train_ppo()."""

    def test_all_params_forwarded(self):
        """Verify that main() passes every CLI arg to train_ppo()."""
        cli_args = [
            "prog",
            "--data_path", "dummy.csv",
            "--model_dir", "/tmp/test_model",
            "--model_name", "cli_test",
            "--total_timesteps", "100",
            "--seed", "7",
            "--window_size", "5",
            "--n_steps", "4096",
            "--ent_coef", "0.02",
            "--learning_rate", "0.001",
            "--gamma", "0.95",
            "--episode_length", "500",
            "--hmax", "50000",
            "--stoploss_penalty", "0.85",
            "--profit_loss_ratio", "2.0",
            "--cash_penalty", "0.1",
        ]

        with patch.object(sys, "argv", cli_args), \
             patch("src.train_ppo.pd.read_csv", return_value=pd.DataFrame()), \
             patch("src.train_ppo.os.path.exists", return_value=True), \
             patch("src.train_ppo.set_seeds"), \
             patch("src.train_ppo.train_ppo") as mock_train:
            mock_train.return_value = (None, None)
            main()

        mock_train.assert_called_once()
        kw = mock_train.call_args
        # Check all arguments are present (either positional or keyword)
        _, kwargs = kw
        assert kwargs["n_steps"] == 4096
        assert kwargs["hmax"] == 50000
        assert kwargs["stoploss_penalty"] == 0.85
        assert kwargs["profit_loss_ratio"] == 2.0
        assert kwargs["cash_penalty"] == 0.1
        assert kwargs["seed"] == 7
        assert kwargs["window_size"] == 5
        assert kwargs["ent_coef"] == 0.02
        assert kwargs["learning_rate"] == 0.001
        assert kwargs["gamma"] == 0.95
        assert kwargs["episode_length"] == 500
