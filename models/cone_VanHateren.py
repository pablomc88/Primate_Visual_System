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
# Name: cone_VanHateren
#
# Description: Python implementation of the model of primate cones and horizontal
# cells by Van Hateren [2] that describes adaptation to the mean background intensity.
# The cone-horizontal cell feedback loop corresponds to the linear version
# (Fig. 5 A [1]). When the model is simulated without horizontal-cell feedback,
# the output is read from the voltage of the inner segment, Vis, which is assumed
# to be the membrane potential of the cone.
#
# References:
#
# [1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
# Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
# 2017
#
# [2] van Hateren, Hans. "A cellular and molecular model of response kinetics
# and adaptation in primate cones and horizontal cells." Journal of vision 5.4
# (2005): 5-5
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import numpy as np

import linear_Filter

class cone(object):

    def __init__(self,step,tauR,tauE,cb,kb,nX,tauC,ac,nc,taum,ais,gamma,tauis,
    gs,tau1,tau2,tauh,githr,with_horizontal_cell_fb):
        self.step = step # simulation time step (in ms)
        self.tauR = tauR # (ms)
        self.tauE = tauE # (ms)
        self.cb = cb # ms^(-1)
        self.kb = kb # ms^(-1)/td
        self.nX = nX
        self.tauC = tauC # ms
        self.ac = ac # arbitrary unit
        self.nc = nc
        self.taum = taum # ms
        self.ais = ais # arbitrary unit
        self.gamma = gamma
        self.tauis = tauis # ms
        self.gs = gs # arbitrary unit
        self.tau1 = tau1 # ms
        self.tau2 = tau2 # ms
        self.tauh = tauh # ms
        self.githr = githr # minimum threshold of conductance gi
        self.with_horizontal_cell_fb = with_horizontal_cell_fb

        # Creation of processing stages
        # Phototransduction
        self.LF_tauR = linear_Filter.LinearFilter(tauR,0,step)
        self.LF_tauE = linear_Filter.LinearFilter(tauE,0,step)
        self.beta = 1.0 # Random number>0 (to avoid dividing by zero)
        self.Q = 0.0
        # Calcium feedback
        self.LF_X = linear_Filter.LinearFilter(1.0,0,step) # Random tau>0
        self.Ios = 0.0
        self.LF_tauC = linear_Filter.LinearFilter(tauC,0,step)
        self.alpha = 1.0 # Random number>0 (to avoid dividing by zero)
        # Inner segment
        self.LF_taum = linear_Filter.LinearFilter(taum,0,step)
        self.LF_tauis = linear_Filter.LinearFilter(tauis,0,step)
        self.gis = 0.0
        self.gi = githr
        # Horizontal cell feedback
        if(self.with_horizontal_cell_fb):
            self.LF_tau1 = linear_Filter.LinearFilter(tau1,0,step)
            self.LF_tau2 = linear_Filter.LinearFilter(tau2,0,step)
            self.LF_tauh = linear_Filter.LinearFilter(tauh,0,step)
            self.It = 0.0

    # Load new input values to the filters.
    def feedInput(self,new_input):
        # Phototransduction cascade
        self.LF_tauR.feedInput(new_input)
        self.LF_tauE.feedInput(self.LF_tauR.last_values[0])
        self.beta =self.cb + self.kb * self.LF_tauE.last_values[0]
        self.Q = 1.0 / self.beta
        # Calcium feedback
        self.LF_X.feedInput(self.Q / self.alpha)
        self.Ios = np.power(self.LF_X.last_values[0],self.nX)
        self.LF_tauC.feedInput(self.Ios)
        self.alpha = 1.0 +\
        np.power(self.ac * self.LF_tauC.last_values[0],self.nc)
        # Inner segment
        self.LF_taum.feedInput(self.Ios / self.gi)
        self.gis = self.ais * np.power(self.LF_taum.last_values[0],self.gamma)
        self.LF_tauis.feedInput(self.gis)
        if(self.LF_tauis.last_values[0] > self.githr):
            self.gi = self.LF_tauis.last_values[0]
        # Horizontal cell feedback
        if(self.with_horizontal_cell_fb):
            self.It = self.gs * (self.LF_taum.last_values[0] -\
            self.LF_tauh.last_values[0])
            self.LF_tau1.feedInput(self.It)
            self.LF_tau2.feedInput(self.LF_tau1.last_values[0])
            self.LF_tauh.feedInput(self.LF_tau2.last_values[0])

    # Update dynamics of linear filters
    def update(self):
        # Phototransduction cascade
        self.LF_tauR.update()
        self.LF_tauE.update()
        # Calcium feedback
        self.LF_X.tau = self.Q # input-dependent tau
        self.LF_X.coefficients() # recalculate coefficients
        self.LF_X.update()
        self.LF_tauC.update()
        # Inner segment
        self.LF_taum.update()
        self.LF_tauis.update()
        # Horizontal cell feedback
        if(self.with_horizontal_cell_fb):
            self.LF_tau1.update()
            self.LF_tau2.update()
            self.LF_tauh.update()

    # The cone membrane potential is assumed to be the voltage of the inner
    # segment, Vis
    def get_membrane_potential(self):
        return self.LF_taum.last_values[0]
