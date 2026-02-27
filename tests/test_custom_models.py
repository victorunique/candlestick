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
