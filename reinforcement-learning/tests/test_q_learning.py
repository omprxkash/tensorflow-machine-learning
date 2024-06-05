import numpy as np
import pytest

from src.environments.gridworld import GridWorld
from src.agents.q_learning import QLearningAgent


@pytest.fixture
def env():
    return GridWorld(size=4, holes=[], goal=(3, 3), max_steps=50)


@pytest.fixture
def agent(env):
    return QLearningAgent(
        n_states=env.n_states,
        n_actions=env.action_space.n,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.01,
    )


def test_q_table_shape(env, agent):
    assert agent.Q.shape == (env.n_states, env.action_space.n)


def test_choose_action_valid(env, agent):
    state, _ = env.reset()
    action = agent.choose_action(state)
    assert 0 <= action < env.action_space.n


def test_epsilon_decays_after_update(env, agent):
    state, _ = env.reset()
    eps_before = agent.get_epsilon()
    action = agent.choose_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.update(state, action, reward, next_state, terminated or truncated)
    assert agent.get_epsilon() <= eps_before


def test_q_table_updates(env, agent):
    state, _ = env.reset()
    Q_before = agent.Q.copy()
    action = agent.choose_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    agent.update(state, action, reward, next_state, terminated or truncated)
    assert not np.array_equal(agent.Q, Q_before)


def test_short_training_run(env, agent):
    for _ in range(5):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
    assert agent.get_epsilon() < 1.0


def test_greedy_mode(env, agent):
    agent.set_training(False)
    state, _ = env.reset()
    actions = [agent.choose_action(state) for _ in range(10)]
    assert len(set(actions)) == 1  # deterministic greedy


class TestSARSAAgent:
    def test_q_table_shape(self):
        from src.agents.sarsa import SARSAAgent
        env = GridWorld(size=4, holes=[], goal=(3, 3), max_steps=50)
        agent = SARSAAgent(n_states=env.n_states, n_actions=env.action_space.n)
        assert agent.Q.shape == (env.n_states, env.action_space.n)

    def test_on_policy_update(self):
        from src.agents.sarsa import SARSAAgent
        env = GridWorld(size=4, holes=[], goal=(3, 3), max_steps=50)
        agent = SARSAAgent(n_states=env.n_states, n_actions=env.action_space.n)
        state, _ = env.reset()
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        Q_before = agent.Q.copy()
        agent.update(state, action, reward, next_state, terminated or truncated)
        assert not np.array_equal(agent.Q, Q_before)
