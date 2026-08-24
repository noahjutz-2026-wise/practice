import elements
import numpy as np
import embodied


class MyEnv(embodied.Env):
    @property
    def obs_space(self):
        return {
            "reward": elements.Space(np.uint8),
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
            "reward": np.array(0, np.uint8),
            "is_first": np.array(False, bool),
            "is_last": np.array(False, bool),
            "is_terminal": np.array(False, bool),
        }

