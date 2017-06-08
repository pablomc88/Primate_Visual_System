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
# Name: test_vanHateren
#
# Description: example of performance of the different stages in van Hateren's
# model. The model reproduces results shown in Fig. 6 [2] using a simple stimulus,
# a 100-ms step of contrast 2 at a background illuminance of 100 td.
#
# References:
#
# [1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
# Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
# 2017
# [2] van Hateren, Hans. "A cellular and molecular model of response kinetics
# and adaptation in primate cones and horizontal cells." Journal of vision 5.4
# (2005): 5-5.
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import numpy as np
import sys,os
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','models'))

import cone_VanHateren
reload(cone_VanHateren)

def main():

    # Model parameters (from Fig.7 and Table 1 [1])
    tauR = 0.49
    tauE = 16.8
    cb = 2.8 * 10**(-3)
    kb = 1.63 * 10**(-4)
    nX = 1.0
    tauC = 2.89
    ac = 9.08 * 10**(-2)
    nc = 4.0
    taum = 4.0
    ais = 7.09 * 10**(-2)
    gamma = 0.678
    tauis = 56.9
    gs = 8.81 # from Fig. 8 (linear horizontal-cell feedback)
    tau1 = 4.0
    tau2 = 4.0
    tauh = 20.0
    githr = 0.4

    # Simulation parameters
    step = 0.2 # ms (larger values make horizontal feedback unstable)
    tsim = 600.0 # ms

    # Pulse parameters
    pulse_duration = 100.0 # ms
    pulse_tstart = 325.0 # ms (first 300 ms are used to fill the input and
                            # output buffers of linear filters)
    bkg_illuminance = 100.0 # td
    pulse_contrast = 2.0
    pulse_amplitude = pulse_contrast * bkg_illuminance # td

    # Constants calculated by using a dark stimulus
    # (bkg_illuminance = pulse_amplitude = 0)
    Vis_dark = 30.39 # mV
    Vh_dark = 26.05 # mV

    # Create cone model
    cone = cone_VanHateren.cone(step,tauR,tauE,cb,kb,nX,tauC,ac,nc,taum,ais,
    gamma,tauis,gs,tau1,tau2,tauh,githr,True)

    # Records of model response
    response = np.zeros((14,int(tsim/step)))

    # Fixed time-step simulation
    time = []
    for t in np.arange(0,int(tsim/step)):
        # input value
        if(t*step >= pulse_tstart and t*step < pulse_tstart + pulse_duration):
            input = bkg_illuminance + pulse_amplitude
        else:
            input = bkg_illuminance

        # update dynamics of the model
        cone.feedInput(input)
        cone.update()
        # record response values
        time.append(t*step)
        response[0,t] = input
        response[1,t] = cone.LF_tauE.last_values[0]
        response[2,t] = cone.beta
        response[3,t] = cone.Q
        response[4,t] = cone.Q / cone.alpha
        response[5,t] = cone.LF_X.last_values[0]
        response[6,t] = cone.LF_tauC.last_values[0]
        response[7,t] = cone.alpha
        response[8,t] = cone.LF_taum.last_values[0] - Vis_dark
        response[9,t] = cone.gi
        response[10,t] = cone.LF_taum.last_values[0] -\
        cone.LF_tauh.last_values[0]
        response[11,t] = cone.It
        response[12,t] = cone.LF_tau2.last_values[0]
        response[13,t] = cone.LF_tauh.last_values[0] - Vh_dark


    # Plot response of the different processing stages
    f, axarr = plt.subplots(7, 2)
    f.subplots_adjust(hspace=1.5)
    f.subplots_adjust(wspace=0.4)

    row = 0
    col = 0
    for k in np.arange(14):
        # First 300 ms are discarded
        axarr[row,col].plot(time[int(300.0/step):len(time)],
        response[k,int(300.0/step):len(time)])
        plt.setp(axarr[row,col], yticks=
        [(np.min(response[k,int(300.0/step):len(time)])),
        (np.max(response[k,int(300.0/step):len(time)]))],
        yticklabels=
        [str(round(np.min(response[k,int(300.0/step):len(time)]),2)),
        str(round(np.max(response[k,int(300.0/step):len(time)]),2))])
        if(k<6):
            col = 0
            row+=1
        elif(k==6):
            col = 1
            row = 0
        else:
            col = 1
            row+=1


    axarr[0,0].set_title('Illuminance (td)')
    axarr[1,0].set_title('E*')
    axarr[2,0].set_title('Beta')
    axarr[3,0].set_title('1/Beta')
    axarr[4,0].set_title('alpha/Beta')
    axarr[5,0].set_title('X')
    axarr[6,0].set_title('C')
    axarr[0,1].set_title('1/alpha')
    axarr[1,1].set_title('Vis - Vis_dark')
    axarr[2,1].set_title('gi')
    axarr[3,1].set_title('Vs')
    axarr[4,1].set_title('It')
    axarr[5,1].set_title('Vb')
    axarr[6,1].set_title('Vh - Vh_dark')
    axarr[6,0].set_xlabel('time (ms)')
    axarr[6,1].set_xlabel('time (ms)')

    plt.show()

if __name__ == '__main__':
    main()
