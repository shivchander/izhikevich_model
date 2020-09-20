#!/usr/bin/env python3
__author__ = "Shivchander Sudalairaj"
__license__ = "MIT"

'''
model.py: Definition of the izhhikevich neuron

This is a python implementation of the Izhikevich Neuron Model
Credits: Eugene M. Izhikevich (neuron_RS2.m)
'''

import numpy as np


class INeuron:
    def __init__(self):
        self.a = 0.02
        self.b = 0.25
        self.c = -65
        self.d = 6

        self.steps = 1000
        self.tau = 0.25  # step size
        self.timesteps = np.arange(0, self.steps + self.tau, self.tau)  # time steps

        self.v = np.zeros(len(self.timesteps))  # voltage history
        self.v[0] = -64
        self.vt = 30.0  # 30mV threshold
        self.u = np.zeros(len(self.timesteps))  # Recovery history
        self.u[0] = self.b * self.v[0]

        self.T1 = 50  # T1 is the time at which the step input rises
        self.spike_ts = np.zeros(len(self.timesteps))  # spike history
        self.I = np.zeros(len(self.timesteps))         # current history

    def simulate(self, I=None):
        """
        :param I: Current mA, default I = 1mA
        :return: None

        Simulates a single neuron and drives it with I
        """
        if I is None:
            self.I = [1.0] * len(self.timesteps)        # default I = 1.0
        elif isinstance(I, float):
            self.I = [I] * len(self.timesteps)          # user defined I
        else:
            self.I = I                                  # I vector

        for t in range(1, len(self.timesteps)):

            if self.v[t-1] < self.vt:
                # calc membrane potential
                dV = (0.04 * self.v[t - 1] + 5) * self.v[t - 1] + 140 - self.u[t - 1]
                self.v[t] = self.v[t - 1] + (dV + self.I[t - 1]) * self.tau
                # calc recovery variable
                du = self.a * (self.b * self.v[t - 1] - self.u[t - 1])
                self.u[t] = self.u[t - 1] + self.tau * du

            else:  # spike
                self.v[t-1] = self.vt  # reset to spike value
                self.spike_ts[t-1] = 1
                self.v[t] = self.c  # reset membrane voltage
                self.u[t] = self.u[t - 1] + self.d  # reset recovery

    def get_v(self):
        """
        :return: vector which accumulates time series V
        """
        return self.v

    def get_timesteps(self):
        """
        :return: returns the time steps
        """
        return self.timesteps

    def mean_spike_rate(self):
        """
        :return: Mean spike rate

        Calculates the mean spike rate for the corresponding I value
        """
        return sum(self.spike_ts[800:])/800
