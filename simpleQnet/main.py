#!/usr/bin/env python

import numpy as np
from simpleQnet.agent import agent
from simpleQnet.environment import GymEnvironment

env = GymEnvironment("Breakout-v0")
ag = agent(env)

iterations = 50000

records = ag.play(1, display=True, save=True)

for i in range(iterations):
    print "iteration: " + str(i)
    print "- playing"
    record = ag.play(10, display=True, save=True)
    records = np.concatenate((records, record), axis=0)
    print "- training"
    error = ag.train(records, 100)
    print error
    ag._network.save_net_model()
