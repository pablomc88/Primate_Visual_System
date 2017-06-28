#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of the project published in [1].
#
# The software is licensed under the GNU General Public License. You should have
# received a copy of the GNU General Public License along with the source code.
#
#
# BeginDocumentation
#
# Name: test_LinearFilter
#
# Description: Impulse response of the linear filter by using a dirac delta
# function that is zero everywhere except at time 'tstart'
#
# References:
#
# [1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
# Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
# 2017
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import numpy as np
import sys,os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','models'))

import linear_Filter

def main():
    # Filter parameters
    tau = 20.0 # ms
    n = 2.0

    # Simulation parameters
    step = 0.1 # ms
    tsim = 200.0 # ms

    # Time in which the delta function is not zero
    tstart = 50.0 # ms

    # Input (Dirac delta function)
    input = np.zeros(int(tsim/step))
    input[int(tstart/step)] = 1.0/step
    # Filter response
    response = np.zeros(int(tsim/step))

    # Create linear filter
    LF = linear_Filter.LinearFilter(tau,n,step)

    # Fixed time-step simulation
    time = []
    for t in np.arange(0,len(input)):
        LF.feedInput(input[t])
        LF.update()
        response[t] = LF.last_values[0]
        time.append(t*step)

    # Exact representation of the exponential cascade function
    time = np.array(time)
    if (n):
        exact = np.power(n*time,n) * np.exp(-n*time/tau) /\
        (np.math.factorial(n-1) * np.power(tau,n+1.0))
    else:
        exact = np.exp(-time/tau)/tau

    exact = np.roll(exact,int(tstart/step))
    exact[0:int(tstart/step)] = 0

    # Plot and compare exact and approximate representations
    plt.plot(time,response,'b',label='Recursive filter')
    plt.plot(time,exact,'r',label='Exact function')

    plt.xlabel('time (ms)')
    plt.ylabel('Filter response')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
