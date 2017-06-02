#!/usr/bin/env python

from simpleQnet.agent import agent
from simpleQnet.environment import GymEnvironment

env = GymEnvironment("Breakout-v0")
ag = agent(env)

iterations = 5

for i in range(iterations):
    record = ag.play(10, display=True, save=True)
    error = ag.train(record, 100)
    print error
