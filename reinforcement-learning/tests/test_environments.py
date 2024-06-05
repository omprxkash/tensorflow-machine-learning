import pytest
from src.environments.gridworld import GridWorld
from src.environments.minigame import MiniGame


class TestGridWorld:
    def setup_method(self):
        self.env = GridWorld(size=8)

    def test_reset_returns_valid_state(self):
        state, info = self.env.reset()
        assert 0 <= state < self.env.n_states
        assert state == 0  # starts at (0,0)

    def test_step_returns_tuple(self):
        self.env.reset()
        result = self.env.step(0)
        assert len(result) == 5

    def test_wall_boundary_keeps_agent_in_place(self):
        self.env.reset()
        # Trying to go UP from (0,0) — out of bounds
        state, reward, _, _, _ = self.env.step(GridWorld.UP)
        assert state == 0
        assert reward == -1.0

    def test_goal_gives_positive_reward(self):
        env = GridWorld(size=2, holes=[], goal=(1, 1), max_steps=100)
        env.reset()
        env._agent_pos = (1, 0)
        state, reward, terminated, _, _ = env.step(GridWorld.RIGHT)
        assert reward == 10.0
        assert terminated

    def test_hole_gives_negative_reward(self):
        env = GridWorld(size=3, holes=[(0, 1)], goal=(2, 2), max_steps=100)
        env.reset()
        state, reward, terminated, _, _ = env.step(GridWorld.RIGHT)
        assert reward == -5.0
        assert terminated

    def test_observation_space_consistent(self):
        assert self.env.observation_space.n == self.env.n_states
        assert self.env.action_space.n == 4


class TestMiniGame:
    def setup_method(self):
        self.env = MiniGame()

    def test_reset_and_step(self):
        state, _ = self.env.reset()
        assert state == 0
        next_state, reward, terminated, truncated, _ = self.env.step(1)
        assert 0 <= next_state < self.env.n_states

    def test_truncation_at_max_steps(self):
        env = MiniGame(max_steps=2)
        env.reset()
        env.step(3)
        _, _, _, truncated, _ = env.step(3)
        assert truncated
