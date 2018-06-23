#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of the project published in [1,2].
#
# The software is licensed under the GNU General Public License. You should have
# received a copy of the GNU General Public License along with the source code.
#
#
# BeginDocumentation
#
# Name: linear_Filter
#
# Description: Low-pass temporal filters implemented as Infinite Impulse Response
# (IIR) filters. In this type of filtering, preceding output values, Y (k − i),
# are used in the calculation of the new output values, Y (k), at the current
# time-step k:
#
# Y (k) = sum_j bj X(k-j) - sum_i ai Y(k-i)
#
# where X(k−j) are the preceding input values. Coefficients ai and bj are
# calculated for each filter according to the equations provided in [3].
#
# References:
#
# [1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2018). A Neuronal Network Model
# of the Primate Visual System: Color Mechanisms in the Retina, LGN and V1. In
# International Journal of Neural Systems. Accepted for publication.
#
# [2] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
# Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
# 2017
#
# [3] Wohrer, Adrien, and Pierre Kornprobst. "Virtual retina: a biological
# retina model and simulator, with contrast gain control." Journal of
# computational neuroscience 26.2 (2009): 219-249.
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import numpy as np

class LinearFilter(object):

    def __init__(self,tau,n,step):
        self.tau = tau # time constant (in ms)
        self.n = n # number of low-pass filtering stages
        self.step = step # simulation time step (in ms)
        self.M = 1 # bj coefficients go from b[0] to b[M-1]. For exponential
                    # and  exponential cascade: M = 1
        self.N = self.n + 1 # ai coefficients go from a[1] to a[N]

        # Calculate coefficients of the filter
        self.coefficients()

        # Arrays of input and output values
        self.last_inputs = np.zeros(self.M)
        self.last_values = np.zeros(self.N+1)

    # Coefficients of the normalized exponential cascade filter
    def coefficients(self):
        if(self.n):
            tauC = self.tau/self.n
        else:
            tauC = self.tau

        c = np.exp(-self.step/tauC)
        self.b = np.zeros(1)
        self.b[0] = np.power(1-c,self.N)
        self.a = np.zeros(self.N+1)

        for i in np.arange(0,self.N+1,1):
            self.a[i] = np.power(-c,i) * self.combination(self.N,i)

    # Auxiliary functions to compute combinatorials of the exponential cascade
    def arrangement(self,n,k):
        res=1.0
        for i in np.arange(n,n-k,-1):
            res*=i

        return res

    def combination(self,n,k):
        return self.arrangement(n,k)/self.arrangement(k,k)

    # Load new input value
    def feedInput(self,new_input):
        self.last_inputs[0]=new_input

    # Update dynamics
    def update(self):

        # Rotation on addresses of the last_values
        fakepoint=self.last_values[self.N]
        for i in np.arange(1,self.N+1,1):
            self.last_values[self.N+1-i]=self.last_values[self.N-i]
        self.last_values[0]=fakepoint

        # Computation of a new output value recursively
        self.last_values[0] = self.b[0]* self.last_inputs[0]
        for j in np.arange(1,self.M,1):
            self.last_values[0] += ( self.b[j] * self.last_inputs[j] )
        for k in np.arange(1,self.N+1,1):
            self.last_values[0] -= ( self.a[k] * self.last_values[k] )
        if(self.a[0]!=1.0):
            self.last_values[0] = self.last_values[0] / self.a[0]

