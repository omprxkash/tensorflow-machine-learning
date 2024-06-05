from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_output_dim(self) -> int:
        for m in reversed(list(self.net.modules())):
            if hasattr(m, 'out_features'):
                return m.out_features
        raise ValueError('No linear layer found')
