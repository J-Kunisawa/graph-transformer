import numpy as np
from typing import Tuple


class Meter(object):
    """Abstract meter interface."""

    def reset(self) -> None:
        raise NotImplementedError

    def add(self, value: float, n: int = 1) -> None:
        raise NotImplementedError

    def value(self) -> Tuple[float, float]:
        raise NotImplementedError


class AverageValueMeter(Meter):
    """
    Online average and (sample) standard deviation using Welford's algorithm.

    Methods:
      - add(value, n=1): add `value` observed `n` times.
      - value(): returns (mean, std) where std is sample std (NaN if insufficient samples).
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        self.n = 0  # total count
        self.mean = 0.0
        self._m2 = 0.0  # sum of squares of differences from the current mean
        self.val = 0.0

    def add(self, value: float, n: int = 1) -> None:
        """
        Add `value` observed `n` times. For n > 1 this internally repeats the incremental update.
        """
        self.val = float(value)
        # handle integer counts >1 by repeated updates (keeps numerical stability)
        for _ in range(max(1, int(n))):
            self.n += 1
            delta = self.val - self.mean
            self.mean += delta / float(self.n)
            delta2 = self.val - self.mean
            self._m2 += delta * delta2

    def value(self) -> Tuple[float, float]:
        if self.n < 1:
            return float("nan"), float("nan")
        if self.n < 2:
            return float(self.mean), float("nan")
        variance = self._m2 / float(self.n - 1)
        return float(self.mean), float(np.sqrt(variance))