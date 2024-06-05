from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self):
        self._training = True
        self._epsilon = 0.0
        self._total_env_steps: int = 0

    @abstractmethod
    def choose_action(self, state):
        ...

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    def get_epsilon(self) -> float:
        return self._epsilon

    def set_training(self, training: bool) -> None:
        self._training = training

    @property
    def total_env_steps(self) -> int:
        return self._total_env_steps
