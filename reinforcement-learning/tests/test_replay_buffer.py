import numpy as np
import pytest
import torch

from src.training.replay_buffer import ReplayBuffer


@pytest.fixture
def buffer():
    return ReplayBuffer(capacity=500)


def _push_n(buf, n, obs_dim=4):
    for i in range(n):
        state = np.random.randn(obs_dim).astype(np.float32)
        next_state = np.random.randn(obs_dim).astype(np.float32)
        buf.push(state, i % 2, float(i), next_state, bool(i % 5 == 0))


def test_len_after_push(buffer):
    _push_n(buffer, 200)
    assert len(buffer) == 200


def test_capacity_is_respected():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 150)
    assert len(buf) == 100


def test_sample_shapes(buffer):
    _push_n(buffer, 200)
    states, actions, rewards, next_states, dones = buffer.sample(64)
    assert states.shape == (64, 4)
    assert actions.shape == (64,)
    assert rewards.shape == (64,)
    assert next_states.shape == (64, 4)
    assert dones.shape == (64,)


def test_sample_types(buffer):
    _push_n(buffer, 200)
    states, actions, rewards, next_states, dones = buffer.sample(32)
    assert states.dtype == torch.float32
    assert actions.dtype == torch.int64
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.float32


def test_sample_requires_enough_data():
    buf = ReplayBuffer(capacity=100)
    _push_n(buf, 30)
    with pytest.raises(ValueError):
        buf.sample(64)
