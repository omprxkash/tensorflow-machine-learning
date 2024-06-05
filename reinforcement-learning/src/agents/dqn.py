import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.networks.dqn_network import DQNNetwork
from src.training.replay_buffer import ReplayBuffer


class DQNAgent(BaseAgent):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims=(128, 128),
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update_freq: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10_000,
        grad_clip: float = 10.0,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self._epsilon = epsilon_start
        self._steps = 0

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_net = DQNNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size, device=self.device)
        self._loss: float = 0.0

    def choose_action(self, state) -> int:
        if self._training and np.random.random() < self._epsilon:
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(state_t).argmax(dim=1).item())

    def update(self, state, action: int, reward: float, next_state, done: bool) -> float:
        self.replay_buffer.push(state, action, reward, next_state, done)
        self._steps += 1

        # decay epsilon linearly
        frac = min(self._steps / self.epsilon_decay_steps, 1.0)
        self._epsilon = self._epsilon - frac * (self._epsilon - self.epsilon_end)

        if len(self.replay_buffer) < self.batch_size:
            return 0.0  # buffer still warming up

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        self._loss = loss.item()
        return self._loss



    @property
    def device_name(self) -> str:
        return str(self.device)

    @property
    def total_steps(self) -> int:
        return self._steps

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "steps": self._steps,
            "epsilon": self._epsilon,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self._steps = ckpt.get("steps", 0)
        self._epsilon = ckpt.get("epsilon", self.epsilon_end)
