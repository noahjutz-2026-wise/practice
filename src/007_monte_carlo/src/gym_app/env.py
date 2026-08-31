import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.core import ActType, Env, ObsType, WrapperObsType


class DiscreteCartPole(ObservationWrapper):
    """
    CartPole-v1 wrapper that digitizes continuous observations into discrete bins.
    """

    def __init__(self, env: Env[ObsType, ActType], n_bins: tuple[int, int, int, int]):
        super().__init__(env)
        self.n_bins = n_bins
        self.bin_lo = np.array([-2.4, -2.5, -0.2095, -2.5])
        bin_hi = np.array([2.4, 2.5, 0.2095, 2.5])
        n_bins_arr = np.array(n_bins, dtype=np.float64)
        self.bin_step = (bin_hi - self.bin_lo) / (n_bins_arr - 1)
        self.bin_max = np.array(n_bins, dtype=np.int64) - 1  # last valid index

    def observation(self, observation: ObsType) -> WrapperObsType:
        return np.clip(
            ((observation - self.bin_lo) / self.bin_step + 0.5).astype(np.int64),
            0,
            self.bin_max,
        )
