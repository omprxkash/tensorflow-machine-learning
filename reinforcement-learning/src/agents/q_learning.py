import numpy as np

from src.agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self._epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state: int) -> int:
        if self._training and np.random.random() < self._epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        target = reward + (0.0 if done else self.gamma * np.max(self.Q[next_state]))
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)


    def reset_epsilon(self, value: float = 1.0) -> None:
        self._epsilon = max(self.epsilon_min, value)

    def get_q_table(self) -> np.ndarray:
        return self.Q.copy()

    def save(self, path: str) -> None:
        np.save(path, self.Q)

    def load(self, path: str) -> None:
        self.Q = np.load(path if path.endswith(".npy") else path + ".npy")
