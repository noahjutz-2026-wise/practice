from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class SimulationArgs:
    n_actions: int
    epsilon: float
    alpha: Callable[[int], float]
    n_steps: int
    initial_q: float
    is_stationary: bool
