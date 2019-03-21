"""
Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/envs.py

This contains auxiliary wrappers for the atari openAI gym environment e.g. proper resizing of the
input frame and a running average normalisation of said frame after resizing
"""

from __future__ import print_function

import cv2
import gym
import numpy as np
from gym import spaces

# -----------------
# Atari preprocessing and wrappers below
# -----------------

# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = spaces.Box(0.0, 1.0, [1, 42, 42])

    @staticmethod
    def _process_frame42(frame):
        frame = frame[34:34 + 160, :160]
        # Resize by half, then down to 42x42 (essentially mipmapping). If
        # we resize directly we lose pixels that, when mapped to 42x42,
        # aren't close enough to the pixel boundary.
        frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (42, 42))
        frame = frame.mean(2, keepdims=True)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.moveaxis(frame, -1, 0)
        return frame

    def _observation(self, observation):
        return self._process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
