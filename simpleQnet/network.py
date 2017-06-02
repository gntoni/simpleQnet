#!/usr/bin/env python

import numpy as np
import lasagne
import theano
import theano.tensor as T

from modelZoo import simpleQnet
from collections import OrderedDict
from tqdm import tqdm


class qNet(object):
    def __init__(self, nActions):
        self._target_var = T.matrix('targets')
        self._network = simpleQnet.build_model(nActions)

        # Parameters
        self.discount_rate = 0.99

        if not isinstance(self._network, OrderedDict):
            raise AttributeError("Network model must be an OrderedDict")
        self._inputLayer = self._network[self._network.keys()[0]]
        self._outputLayer = self._network[self._network.keys()[-1]]

        # Prediction Variables
        self._prediction = lasagne.layers.get_output(self._outputLayer)
        self._test_prediction = lasagne.layers.get_output(
                                        self._outputLayer,
                                        deterministic=True)

        # Forward pass Function
        self.forward_pass = theano.function(
            [self._inputLayer.input_var],
            self._test_prediction)

        # Loss Variable
        self._loss = lasagne.objectives.squared_error(
            self._prediction,
            self._target_var)
        self._loss = self._loss.mean()

        # Backprop and updates
        self._params = lasagne.layers.get_all_params(
                            self._outputLayer,
                            trainable=True)
        self._updates = lasagne.updates.adamax(
                            self._loss,
                            self._params)

        self._train_fn = theano.function(
                [self._inputLayer.input_var, self._target_var],
                self._loss,
                updates=self._updates)

    def train_epoch(self, batch, batchsize):
        err = 0
        batches = 0
        for start_idx in tqdm(range(0, len(batch) - batchsize + 1, batchsize)):
            minibatch = batch[start_idx:start_idx + batchsize]
            minibatch = minibatch.T
            preSt, postSt, actions, rewards, terminals = minibatch
            preSt = np.concatenate(preSt).reshape((preSt.shape+preSt[0].shape))
            preSt = np.asarray(preSt, dtype=theano.config.floatX)
            postSt = np.concatenate(postSt).reshape((postSt.shape+postSt[0].shape))
            postSt = np.asarray(postSt, dtype=theano.config.floatX)

            # Get the expected rewards for the next state
            postQ = self.forward_pass(postSt)
            # Get the max expected reward
            maxPostQ = postQ.max(axis=1)

            # Get the expected rewards for the current state
            preQ = self.forward_pass(preSt)

            #
            qTargets = preQ.copy()
            # clip rewards
            # rewards = np.clip(rewards, self.min_reward, self.max_reward)

            # Update Q values for the actions taken
            for i, action in enumerate(actions):
                if terminals[i]:
                    qTargets[i, action] = float(rewards[i])
                else:
                    qTargets[i, action] = float(rewards[i]) + self.discount_rate * maxPostQ[i]

            #  Todo: Test clip error:
            #          - Calculate loss
            #          - clip error
            #          - calculate updates
            #          - update

            err += self._train_fn(preSt, postQ)
            batches += 1
        return err/batches
