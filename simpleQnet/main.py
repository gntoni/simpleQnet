#!/usr/bin/env python

import numpy as np
from simpleQnet.agent import agent
from simpleQnet.environment import GymEnvironment

env = GymEnvironment("Breakout-v0")
ag = agent(env)

iterations = 1000000

print "Initialising Memory"
records = ag.play(1, display=False, save=True)

while len(records) < 1000000:
    record = ag.play(1, display=False, save=True)
    records = np.concatenate((records, record), axis=0)

print "Start Training"
n_actions_done = 1
for i in range(iterations):
    print "iteration: " + str(i)
    print "- playing"
    if n_actions_done < 1000000:
        pE = 1 - ((0.9*n_actions_done)/1000000)
    else:
        pE = 0.1
    record = ag.play(pE, display=True, save=True)
    n_actions_done += len(record)
    records = np.delete(records, np.arange(len(record)), axis=0)
    records = np.concatenate((records, record), axis=0)
    print "- training"
    sampleIdx = np.random.randint(0, 999999, 32)
    samples = records[sampleIdx]
    error = ag.train(samples, len(samples))
    print error
    ag._network.save_net_model()
