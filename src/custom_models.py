import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class CNN1DFeaturesExtractor(BaseFeaturesExtractor):
    """
    1D CNN feature extractor for handling temporal sequential data.
    Input observation can be:
    1. Flattened by VecFrameStack (Batch, n_stack * n_features)
    2. Stacked (Batch, n_stack, n_features)

    It reshapes the input to (Batch, n_features, n_stack) and applies 1D convolutions.
    
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features in the output vector
    :param n_stack: (int) Number of frames stacked (window size)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, n_stack: int = 10):
        super().__init__(observation_space, features_dim)
        
        self.n_stack = n_stack
        
        if len(observation_space.shape) == 1:
            input_dim = observation_space.shape[0]
            if input_dim % n_stack != 0:
                raise ValueError(f"Observation shape {input_dim} is not divisible by n_stack {n_stack}")
            self.n_features = input_dim // n_stack
            self.flattened_input = True
        elif len(observation_space.shape) == 2:
            if observation_space.shape[0] == n_stack:
                self.n_features = observation_space.shape[1]
                self.needs_transpose = True
            else:
                self.n_features = observation_space.shape[0]
                self.needs_transpose = False
            self.flattened_input = False
        else:
            raise ValueError("Observation space must be 1D (flattened) or 2D.")

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.n_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Flatten()
        )

        with torch.no_grad():
            if self.flattened_input:
                sample_input = torch.zeros(1, self.n_stack * self.n_features)
            elif self.needs_transpose:
                sample_input = torch.zeros(1, self.n_stack, self.n_features)
            else:
                sample_input = torch.zeros(1, self.n_features, self.n_stack)
                
            n_flatten = self.cnn(self.reshape_input(sample_input)).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    
    def reshape_input(self, observations: torch.Tensor) -> torch.Tensor:
        if self.flattened_input:
            x = observations.view(-1, self.n_stack, self.n_features)
            # Conv1d expects (Batch, Channels, Length)
            x = x.permute(0, 2, 1)
        elif self.needs_transpose:
            # Input is (Batch, n_stack, n_features) → permute to (Batch, n_features, n_stack)
            x = observations.permute(0, 2, 1)
        else:
            # Input is already (Batch, n_features, n_stack) — no permute needed
            x = observations
        return x

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.reshape_input(observations)
        x = self.cnn(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test CNN1DFeaturesExtractor Architecture.")
    parser.add_argument("--features_dim", type=int, default=128, help="Output feature dimension.")
    parser.add_argument("--n_stack", type=int, default=60, help="Number of stacked frames (window size).")
    parser.add_argument("--n_features", type=int, default=20, help="Number of technical indicators/price cols.")
    args = parser.parse_args()

    print("Initializing CNN1DFeaturesExtractor:")
    print(f"- Window Size (n_stack): {args.n_stack}")
    print(f"- Extracted Feature Dim: {args.features_dim}")
    print(f"- Input indicators/asset: {args.n_features}")

    obs_space = gym.spaces.Box(low=-1, high=1, shape=(args.n_stack * args.n_features,))
    model = CNN1DFeaturesExtractor(observation_space=obs_space, features_dim=args.features_dim, n_stack=args.n_stack)
    
    print("\nModel Architecture:")
    print(model)
    
    dummy_obs = torch.randn(2, args.n_stack * args.n_features)
    print(f"\nPassing dummy tensor of shape {dummy_obs.shape}...")
    
    output = model(dummy_obs)
    print(f"Output tensor shape: {output.shape} (Expected: 2, {args.features_dim})")
    print("Success!")
