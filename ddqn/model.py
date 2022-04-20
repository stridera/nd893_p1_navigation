import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size: int, action_size: int, seed: int, fc1_units: int = 64, fc2_units: int = 64):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.network = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
        )

    def forward(self, state: list[float]) -> int:
        """Build a network that maps state -> action values."""
        return self.network(state)
