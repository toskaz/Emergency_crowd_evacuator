"""MiniGrid-style CNN adapted to a drone-centred 9x9 patch.

Architecture reference:
https://github.com/Ofekirsh/Deep-Reinforcement-Learning-Minigrid/blob/main/models/dqn_base.py
The channel count and Q-value outputs are supplied by the drone controller.
"""
from torch import nn

HISTORY_FRAMES = 4
VIEW_SIZE = 9
VIEW_RADIUS = VIEW_SIZE // 2
ENCODED_SIZE = 64 * 5 * 5


def map_encoder(channels):
    return nn.Sequential(
        nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Flatten(),
    )
