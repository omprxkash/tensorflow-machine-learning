from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MiniGame(gym.Env):
    """A compact 5×5 grid — same mechanics as GridWorld but faster for unit tests."""

    metadata = {"render_modes": ["human"]}

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    _ACTION_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

    def __init__(self, max_steps: int = 50):
        super().__init__()
        self.size = 5
        self.holes = {(2, 2)}
        self.goal = (4, 4)
        self.max_steps = max_steps

        self.n_states = self.size * self.size
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)

        self._agent_pos: Tuple[int, int] = (0, 0)
        self._steps: int = 0

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_pos = (0, 0)
        self._steps = 0
        return self._pos_to_state(self._agent_pos), {}

    def step(self, action: int):
        dr, dc = self._ACTION_DELTA[action]
        r, c = self._agent_pos
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.size and 0 <= nc < self.size):
            nr, nc = r, c
            reward = -1.0
        elif (nr, nc) in self.holes:
            reward = -5.0
        elif (nr, nc) == self.goal:
            reward = 10.0
        else:
            reward = -0.1

        self._agent_pos = (nr, nc)
        self._steps += 1

        terminated = (self._agent_pos == self.goal) or (self._agent_pos in self.holes)
        truncated = self._steps >= self.max_steps

        return self._pos_to_state(self._agent_pos), reward, terminated, truncated, {}

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for r, c in self.holes:
            grid[r][c] = "H"
        gr, gc = self.goal
        grid[gr][gc] = "G"
        ar, ac = self._agent_pos
        grid[ar][ac] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print(f"Steps: {self._steps}/{self.max_steps}")
        print()
