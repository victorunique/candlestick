import pytest
import torch
import gymnasium as gym
from src.custom_models import CNN1DFeaturesExtractor

def test_cnn1d_init_flattened():
    # Simulate a standard SB3 Box space that has been flattened
    n_stack = 60
    n_features = 20
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_stack * n_features,))
    
    model = CNN1DFeaturesExtractor(obs_space, features_dim=128, n_stack=n_stack)
    
    assert model.n_stack == 60
    assert model.n_features == 20
    assert model.flattened_input is True

def test_cnn1d_init_stacked():
    # Simulate a multi-dim Box space
    n_stack = 60
    n_features = 20
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_stack, n_features))
    
    model = CNN1DFeaturesExtractor(obs_space, features_dim=128, n_stack=n_stack)
    
    assert model.n_stack == 60
    assert model.n_features == 20
    assert model.flattened_input is False

def test_cnn1d_forward_flattened():
    n_stack = 10
    n_features = 5
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_stack * n_features,))
    
    model = CNN1DFeaturesExtractor(obs_space, features_dim=64, n_stack=n_stack)
    
    # Batch of 4
    dummy_obs = torch.randn(4, n_stack * n_features)
    output = model(dummy_obs)
    
    assert output.shape == (4, 64)

def test_cnn1d_forward_stacked():
    n_stack = 10
    n_features = 5
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_stack, n_features))
    
    model = CNN1DFeaturesExtractor(obs_space, features_dim=64, n_stack=n_stack)
    
    # Batch of 4
    dummy_obs = torch.randn(4, n_stack, n_features)
    output = model(dummy_obs)
    
    assert output.shape == (4, 64)


# --- Error handling ---

def test_cnn1d_init_flattened_indivisible():
    """ValueError when flattened obs dim is not divisible by n_stack."""
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(50,))
    with pytest.raises(ValueError, match="not divisible by n_stack"):
        CNN1DFeaturesExtractor(obs_space, features_dim=64, n_stack=7)


def test_cnn1d_init_3d_space():
    """ValueError when observation space has 3+ dimensions."""
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(4, 5, 3))
    with pytest.raises(ValueError, match="must be 1D .* or 2D"):
        CNN1DFeaturesExtractor(obs_space, features_dim=64, n_stack=4)


# --- 2D branch where shape[0] != n_stack ---

def test_cnn1d_init_stacked_shape0_ne_nstack():
    """When shape[0] != n_stack, n_features should come from shape[0]."""
    n_stack = 10
    n_features = 8
    # shape is (n_features, n_stack) — reversed order
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_features, n_stack))

    model = CNN1DFeaturesExtractor(obs_space, features_dim=64, n_stack=n_stack)

    assert model.n_features == n_features
    assert model.flattened_input is False


def test_cnn1d_forward_stacked_shape0_ne_nstack():
    """Forward pass works when 2D shape[0] != n_stack."""
    n_stack = 10
    n_features = 8
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_features, n_stack))

    model = CNN1DFeaturesExtractor(obs_space, features_dim=64, n_stack=n_stack)
    dummy_obs = torch.randn(2, n_features, n_stack)
    output = model(dummy_obs)

    assert output.shape == (2, 64)


# --- reshape_input ---

def test_reshape_input_flattened():
    """reshape_input should convert (B, n_stack*n_feat) → (B, n_feat, n_stack)."""
    n_stack = 4
    n_features = 3
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_stack * n_features,))
    model = CNN1DFeaturesExtractor(obs_space, features_dim=32, n_stack=n_stack)

    x = torch.randn(2, n_stack * n_features)
    out = model.reshape_input(x)
    assert out.shape == (2, n_features, n_stack)


def test_reshape_input_stacked():
    """reshape_input should permute (B, n_stack, n_feat) → (B, n_feat, n_stack)."""
    n_stack = 4
    n_features = 3
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_stack, n_features))
    model = CNN1DFeaturesExtractor(obs_space, features_dim=32, n_stack=n_stack)

    x = torch.randn(2, n_stack, n_features)
    out = model.reshape_input(x)
    assert out.shape == (2, n_features, n_stack)


# --- Edge cases ---

def test_cnn1d_forward_single_sample():
    """Forward pass with batch size 1."""
    n_stack = 10
    n_features = 5
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_stack * n_features,))
    model = CNN1DFeaturesExtractor(obs_space, features_dim=64, n_stack=n_stack)

    output = model(torch.randn(1, n_stack * n_features))
    assert output.shape == (1, 64)


def test_cnn1d_different_features_dim():
    """Output dimension should match the requested features_dim."""
    n_stack = 10
    n_features = 5
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_stack * n_features,))

    for dim in (32, 64, 256):
        model = CNN1DFeaturesExtractor(obs_space, features_dim=dim, n_stack=n_stack)
        output = model(torch.randn(2, n_stack * n_features))
        assert output.shape == (2, dim)


def test_cnn1d_eval_mode():
    """Forward pass should succeed in eval mode (BatchNorm uses running stats)."""
    n_stack = 10
    n_features = 5
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(n_stack * n_features,))
    model = CNN1DFeaturesExtractor(obs_space, features_dim=64, n_stack=n_stack)

    # Train with a few batches so running stats are populated
    model.train()
    for _ in range(3):
        model(torch.randn(4, n_stack * n_features))

    model.eval()
    output = model(torch.randn(2, n_stack * n_features))
    assert output.shape == (2, 64)
