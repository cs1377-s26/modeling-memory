from dataclasses import dataclass
from torch.distributions import Normal
import torch
from abc import ABC
from .utils import clamp
from enum import IntEnum

@dataclass
class Card:
    difficulty: float

    @staticmethod
    def sample(seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        difficulty = clamp(round(Normal(5., 2.).sample().item()), 1, 10)
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