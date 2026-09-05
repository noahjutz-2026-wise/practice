"""Monkey-patches pygame.mixer to avoid missing shared library errors (e.g. libgthread-2.0.so.0)
in headless or containerized environments where audio is not used.
"""

import sys
from unittest.mock import MagicMock

# Mock pygame.mixer so utils.load_sounds type annotations and calls succeed without libgthread
mock_mixer = MagicMock()
mock_mixer.Sound = object
sys.modules["pygame.mixer"] = mock_mixer

try:
    import pygame
    pygame.mixer = mock_mixer
except ImportError:
    pass

# Expose flappy_bird_gymnasium after applying monkey patch
import flappy_bird_gymnasium

__all__ = ["flappy_bird_gymnasium"]
