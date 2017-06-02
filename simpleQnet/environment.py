#!/usr/bin/env python

import gym


class GymEnvironment(object):
    """Environment"""
    def __init__(self, env_id):
        self.gym = gym.make(env_id)
        self.observation = None
        self.terminated = None
        self.nActions = self.gym.action_space.n

    def restart(self):
        self.observation = self.gym.reset()
        self.terminated = None

    def make_action(self, action):
        self.observation, reward, self.terminated, _ = self.gym.step(action)
        return reward

    def get_observation(self):
        if self.observation is not None:
            return self.observation
        else:
            raise AttributeError("Observation requested but not available.")

