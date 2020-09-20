#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
This is a python implementation of the Izhikevich Neuron Model
Credits: Eugene M. Izhikevich (neuron_RS2.m)
'''
import matplotlib.pyplot as plt
import numpy as np
from model import INeuron


def q1():
    """
    :return: None (Saves two graphs in the current directory)

    Running simulations with different I values to identify the relationship with mean spike rate and I
    And performing a visual comparison of the spikes by varying the I
    """
    I = np.arange(0, 20.5, 0.5)
    r = []
    fig, axs = plt.subplots(5, figsize=(6, 8))
    plt_num = 0
    for i in I:
        neuron = INeuron()
        neuron.simulate(I=i)
        r.append(neuron.mean_spike_rate())
        if i in [1, 5, 10, 15, 20]:
            axs[plt_num].plot(neuron.get_timesteps(), neuron.get_v())
            axs[plt_num].set_title("I = " + str(i))
            axs[plt_num].set_ylabel("V_m")
            plt_num += 1
    plt.tight_layout()
    plt.savefig('VvsI.pdf')
    plt.clf()
    plt.plot(I, r)
    plt.xlabel('Input (I)')
    plt.ylabel('Mean Spike Rate')
    plt.title('Mean Spike Rate (R) vs I')
    plt.savefig('RvsI.pdf')
    plt.clf()


def q2(wBA=10, Ia=5.0):
    """
    :return: None (Saves two graphs in the current directory)

    Network of two neurons A and B
    """
    Ib = np.arange(0, 20.5, 0.5)
    rq1 = []
    rb = []
    fig, axs = plt.subplots(5, figsize=(6, 8))
    plt_num = 0
    # Simulating Neuron A separately to obtain time series and using that as input for
    # separate simulation of Neuron B
    neuron_A = INeuron()
    neuron_A.simulate(I=Ia)
    for i in Ib:
        # for graph comparison
        neuron_Q1 = INeuron()
        neuron_Q1.simulate(I=i)
        rq1.append(neuron_Q1.mean_spike_rate())

        _Ib = i + (neuron_A.spike_ts * wBA)
        neuron_B = INeuron()
        neuron_B.simulate(I=_Ib)
        rb.append(neuron_B.mean_spike_rate())

        if i in [1, 5, 10, 15, 20]:
            axs[plt_num].plot(neuron_B.get_timesteps(), neuron_B.get_v())
            axs[plt_num].set_title("I = " + str(i))
            axs[plt_num].set_ylabel("V_m")
            plt_num += 1

    plt.tight_layout()
    plt.savefig('VvsI_2.pdf')
    plt.clf()
    plt.plot(Ib, rb, 'b-', label="R_b")
    plt.plot(Ib, rq1, 'r--', label="R")
    plt.xlabel('Input (I)')
    plt.ylabel('Mean Spike Rate')
    plt.legend(loc="upper left")
    plt.title('Mean Spike Rate (R) vs I')
    plt.savefig('RvsI_2.pdf')
    plt.clf()


def main():
    q1()
    q2()


if __name__ == '__main__':
    main()
