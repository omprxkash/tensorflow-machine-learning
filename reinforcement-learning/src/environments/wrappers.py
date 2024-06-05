from collections import deque
from typing import Deque

import numpy as np
import gymnasium as gym


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize Box observations to [-1, 1] using running mean/std."""

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self._obs_rms_mean = np.zeros(env.observation_space.shape)
        self._obs_rms_var = np.ones(env.observation_space.shape)
        self._count = 0

    def observation(self, obs):
        self._count += 1
        delta = obs - self._obs_rms_mean
        self._obs_rms_mean += delta / self._count
        self._obs_rms_var += delta * (obs - self._obs_rms_mean)
        std = np.sqrt(self._obs_rms_var / max(self._count - 1, 1) + self.epsilon)
        return (obs - self._obs_rms_mean) / std


class RecordEpisodeStats(gym.Wrapper):
    """Tracks per-episode return and length, accessible as `env.episode_stats`."""

    def __init__(self, env: gym.Env, deque_size: int = 100):
        super().__init__(env)
        self.episode_stats: Deque = deque(maxlen=deque_size)
        self._ep_return = 0.0
        self._ep_length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ep_return = 0.0
        self._ep_length = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_return += reward
        self._ep_length += 1
        if terminated or truncated:
            self._total_episodes += 1
            self.episode_stats.append(
                {"return": self._ep_return, "length": self._ep_length}
            )
        return obs, reward, terminated, truncated, info
