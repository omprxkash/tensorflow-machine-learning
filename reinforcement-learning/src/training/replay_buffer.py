from typing import Optional, Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, device: Optional[torch.device] = None):
        self.capacity = capacity
        self.device = device or torch.device("cpu")
        self._states: list = []
        self._actions: list = []
        self._rewards: list = []
        self._next_states: list = []
        self._dones: list = []
        self._ptr = 0
        self._size = 0

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        if self._size < self.capacity:
            self._states.append(None)
            self._actions.append(None)
            self._rewards.append(None)
            self._next_states.append(None)
            self._dones.append(None)
        self._states[self._ptr] = np.array(state, dtype=np.float32)
        self._actions[self._ptr] = int(action)
        self._rewards[self._ptr] = float(reward)
        self._next_states[self._ptr] = np.array(next_state, dtype=np.float32)
        self._dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if batch_size > self._size:
            raise ValueError(f"Requested {batch_size} samples but buffer only has {self._size}")

        idx = np.random.choice(self._size, batch_size, replace=False)
        states = torch.FloatTensor(np.stack([self._states[i] for i in idx])).to(self.device)
        actions = torch.LongTensor([self._actions[i] for i in idx]).to(self.device)
        rewards = torch.FloatTensor([self._rewards[i] for i in idx]).to(self.device)
        next_states = torch.FloatTensor(np.stack([self._next_states[i] for i in idx])).to(self.device)
        dones = torch.FloatTensor([self._dones[i] for i in idx]).to(self.device)
        return states, actions, rewards, next_states, dones



    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity
    def __repr__(self) -> str:
        return f'ReplayBuffer(size={self._size}/{self.capacity})'
    def __len__(self) -> int:
        return self._size
