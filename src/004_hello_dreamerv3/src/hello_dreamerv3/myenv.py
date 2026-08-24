import elements
import embodied
import numpy as np


class MyEnv(embodied.Env):
    @property
    def obs_space(self):
        return {
            "feat": elements.Space(np.float32, (1,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            "action": elements.Space(np.uint8),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        # A simple stub implementation
        return {
            "feat": np.array([0.0], np.float32),
            "reward": np.array(0.0, np.float32),
            "is_first": np.array(False, bool),
            "is_last": np.array(False, bool),
            "is_terminal": np.array(False, bool),
        }
