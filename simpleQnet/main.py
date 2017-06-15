#!/usr/bin/env python

import numpy as np
from simpleQnet.agent import agent
from simpleQnet.environment import GymEnvironment
from matplotlib import pyplot as plt

env = GymEnvironment("Breakout-v0")
ag = agent(env)

iterations = 64

print "Initialising Memory"
records = ag.play(1, display=False, save=True)

while len(records) < iterations:
    record = ag.play(1, display=False, save=True)
    records = np.concatenate((records, record), axis=0)

print "Start Training"
n_actions_done = 1
plt.axis()
plt.ion()
plt.ylabel("G value mean")
plt.xlabel("iteration")

Gmeans = [0]
for i in range(iterations):
    print "iteration: " + str(i)
    print "- playing"
    if n_actions_done < iterations:
        pE = 1 - ((0.9*n_actions_done)/iterations)
    else:
        pE = 0.1
    record = ag.play(pE, display=False, save=True)
    n_actions_done += len(record)
    records = np.delete(records, np.arange(len(record)), axis=0)
    records = np.concatenate((records, record), axis=0)
    print "- training"
    sampleIdx = np.random.randint(0, iterations-1, 32)
    samples = records[sampleIdx]
    error, Gmean = ag.train(samples, len(samples))

    Gmeans.append(Gmean)
    plt.cla()
    plt.plot(Gmeans)
    plt.pause(0.001)
    print error
    if not n_actions_done % 1000:
        ag._network.save_net_model()
