from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    _ACTION_DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
    _ACTION_NAMES = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}

    def __init__(
        self,
        size: int = 8,
        holes: Optional[List[Tuple[int, int]]] = None,
        walls: Optional[List[Tuple[int, int]]] = None,
        goal: Optional[Tuple[int, int]] = None,
        max_steps: int = 200,
    ):
        super().__init__()
        self.size = size
        self.holes = set(map(tuple, holes)) if holes else {(2, 3), (3, 5), (5, 2), (6, 4)}
        self.walls = set(map(tuple, walls)) if walls else set()
        self.goal = tuple(goal) if goal else (size - 1, size - 1)
        self.max_steps = max_steps

        self.n_states = size * size
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)

        self._agent_pos: Tuple[int, int] = (0, 0)
        self._steps: int = 0

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.size)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_pos = (0, 0)
        self._steps = 0
        return self._pos_to_state(self._agent_pos), {}

    def step(self, action: int):
        dr, dc = self._ACTION_DELTA[action]
        r, c = self._agent_pos
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.size and 0 <= nc < self.size) or (nr, nc) in self.walls:
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

        return self._pos_to_state(self._agent_pos), reward, terminated, truncated, {"step": self._steps}

    def render(self):
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for r, c in self.holes:
            grid[r][c] = "H"
        for r, c in self.walls:
            grid[r][c] = "#"
        gr, gc = self.goal
        grid[gr][gc] = "G"
        ar, ac = self._agent_pos
        grid[ar][ac] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()

    def render_heatmap(self, Q_table: np.ndarray, ax=None, title: str = "Q-value heatmap"):
        import matplotlib.pyplot as plt
        V = np.max(Q_table, axis=1).reshape(self.size, self.size)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(V, cmap="viridis")
        plt.colorbar(im, ax=ax)
        gr, gc = self.goal
        ax.plot(gc, gr, "r*", markersize=15)
        for r, c in self.holes:
            ax.plot(c, r, "wx", markersize=12)
        ax.set_title(title)
        return ax
