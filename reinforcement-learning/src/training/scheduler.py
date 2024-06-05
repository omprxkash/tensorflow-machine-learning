class LinearDecay:
    """Decays a value linearly from `start` to `end` over `n_steps` calls."""

    def __init__(self, start: float, end: float, n_steps: int):
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self._t = 0
        self._current = start

    def step(self) -> float:
        frac = min(self._t / self.n_steps, 1.0)
        self._current = self.start + frac * (self.end - self.start)
        self._t += 1
        return self._current

    @property
    def value(self) -> float:
        return self._current

    def reset(self) -> None:
        self._t = 0
        self._current = self.start
