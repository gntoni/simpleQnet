#!/usr/bin/env python

import numpy as np
import scipy.misc
from theano import config

from simpleQnet.network import qNet


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def resize_and_crop(img):
    img = rgb2gray(img)
    img = scipy.misc.imresize(
                                        img,
                                        (110, 84))
    return img[84:, :]


class agent(object):
    def __init__(self, environment):
        # atributes
        self.memory_len = 4
        self.imH = 84
        self.imW = 84

        # initialization
        self._env = environment
        self._network = qNet(self._env.nActions)
        self.memory = None

    def init_memory(self):
        self.memory = np.zeros((self.memory_len, self.imH, self.imW))
        self._env.restart()
        for i in range(self.memory_len):
            self._env.make_action(np.random.randint(self._env.nActions))
            self.memory[i] = resize_and_crop(self._env.get_observation())

    def load_data(self):
        pass

    def initTrain(self):
        pass

    def train(self, batch, minibatchSize):
        return self._network.train_epoch(batch, minibatchSize)

    def play(self, pE, display=True, save=False):
        if save:
            saved_moves = []

        # Run 1 episode
        self._env.restart()
        for i_step in range(100000):

            if self.memory is None:
                self.init_memory()
            if display:
                self._env.gym.render()

            preSt = self.memory.copy()

            q_pred = self._network.forward_pass(np.expand_dims(preSt, 0).astype(config.floatX))

            # exploration vs explotation
            # ---- Probability 1/i_episode of exploring
            explore = np.random.binomial(1, pE)
            if explore:
                action = np.random.randint(self._env.nActions)
            else:
                action = q_pred.argmax()

            reward = self._env.make_action(action)
            terminal = self._env.terminated
            self.memory[:-1] = self.memory[1:]
            self.memory[-1] = resize_and_crop(self._env.get_observation())
            posSt = self.memory.copy()

            if save:
                saved_moves.append(
                    [preSt, posSt, action, reward, terminal])
            if self._env.terminated:
                break

        if save:
            return np.array(saved_moves)
        else:
            return None
