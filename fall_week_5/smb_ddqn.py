import torch
import torch.nn as nn
import random

import gym
from nex_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

from tqdm import tqdm
import pickle
import collections
import cv2

import numpy as np
import matplotlib.pyplot as plt


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every 'skip'-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations for max pooling across time steps

        self._obs_buffer = collections.deque(maxlen=2)
        # maxlen = 2 : for efficiency
        # don't need larger deque since only going to return one observations for
        # all of the observation within the 'skip' frames

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip): # process 'skip' number of steps
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward  # and keep track of all the rewards

            if done:
                break

            max_frame = np.max(np.stack(self._obs_buffer), axis=0)  # why max_frame?
            # at this point the obs in obs_buffer is the last two steps
            # choose the one with maximum values?
            #
            # ANS:  choosing the maximum values within the two timesteps for each pixel
            #       not sure if this is necessary.......

            return max_frame, total_reward, done, info

    def reset(self):    # self-explanatory
        """Clear past frame buffer and init to first obs"""

        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    Downsamples image to 84X84
    Greyscales image

    Returns numpy array
    """

    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)

        self.observation_space = gym.spaces.Box(low=0,high=255, shape(84,84,1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."

        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)

        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84,84,1])

        return x_t.astype(np.uint8)
