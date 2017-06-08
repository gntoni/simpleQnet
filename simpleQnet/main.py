#!/usr/bin/env python

from simpleQnet.agent import agent
from simpleQnet.environment import GymEnvironment

env = GymEnvironment("Breakout-v0")
ag = agent(env)

iterations = 50000

for i in range(iterations):
    print "iteration: " + str(i)
    print "- playing"
    record = ag.play(10, display=True, save=True)
    print "- training"
    error = ag.train(record, 100)
    print error
    ag._network.save_net_model()
