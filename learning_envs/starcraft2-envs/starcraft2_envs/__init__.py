"""
Reinforcement learning environments for StarCraft 2.

These environments have been adapted from CommFormer, which can be found
here: https://github.com/charleshsc/CommFormer. The changes made to the
environments are purely to make them compatible with the structure of CTDComm.
"""

from .sc2_env import StarCraft2Env
from .sc2_random_env import RandomStarCraft2Env
