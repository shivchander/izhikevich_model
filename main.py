#!/usr/bin/env python3
__author__ = "SHivchander Sudalairaj"
__license__ = "MIT"

'''
This is a python implementation of the Izhikevich Neuron Model
Credits: Eugene M. Izhikevich
'''
import matplotlib.pyplot as plt
import numpy as np
from model import INeuron


def main():
    I = np.arange(0, 20.5, 0.5)
    v = []
    r = []
    fig, axs = plt.subplots(5, figsize=(6, 8))
    plt_num = 0
    for i in I:
        neuron = INeuron()
        neuron.simulate(I=i)
        r.append(neuron.mean_spike_rate())
        if i in [1, 5, 10, 15, 20]:
            axs[plt_num].plot(neuron.get_timesteps(), neuron.get_v())
            axs[plt_num].set_title("I = "+str(i))
            axs[plt_num].set_ylabel("V_m")
            plt_num += 1
    plt.tight_layout()
    plt.savefig('VvsI.png')
    plt.clf()
    plt.plot(I, r)
    plt.xlabel('Current (mA)')
    plt.ylabel('Mean Spike Rate')
    plt.title('Mean Spike Rate (R) vs I')
    plt.savefig('RvsI.png')
    plt.clf()


if __name__ == '__main__':
    main()
