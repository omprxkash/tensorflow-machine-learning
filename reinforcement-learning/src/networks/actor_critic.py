from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.networks.mlp import MLP


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden_dims=(64, 64)):
        super().__init__()
        self.body = MLP(in_dim, list(hidden_dims), hidden_dims[-1])
        # override: body ends with ReLU-then-linear; replace last Linear output with 64
        # Actually we want the body to output hidden_dims[-1] features, then two heads.
        # Re-build cleanly:
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, n_actions)
        self.value_head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.body(x)
        action_probs = F.softmax(self.policy_head(features), dim=-1)
        value = self.value_head(features)
        return action_probs, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        action_probs, value = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy

    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy
