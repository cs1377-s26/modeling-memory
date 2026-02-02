from dataclasses import dataclass
import numpy as np
from abc import ABC
from .utils import clamp
from enum import IntEnum

@dataclass
class Card:
    difficulty: float

    @staticmethod
    def sample(seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            sample = rng.normal(loc=5.0, scale=2.0)
        else:
            sample = np.random.normal(loc=5.0, scale=2.0)

        difficulty = clamp(round(sample), 1, 10)
        return Card(difficulty=difficulty)

class Rating(IntEnum):
    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4

class Learner(ABC):
    def initialize(self, rating: Rating):
        pass

    def update(self, recall_prob: float, rating: Rating):
        pass

    def attempt(self, dt: int) -> Rating:
        pass

class Schedule(ABC):
    def initialize(self, rating: Rating):
        pass

    def interval(self) -> int:
        pass

    def update(self, rating: Rating, dt: int):
        pass