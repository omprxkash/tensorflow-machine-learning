import torch
import torch.nn as nn

from src.networks.mlp import MLP


class DQNNetwork(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden_dims=(128, 128)):
        super().__init__()
        self.net = MLP(in_dim, list(hidden_dims), n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
