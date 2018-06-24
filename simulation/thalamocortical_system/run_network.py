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
# Name: run_network
#
# Description: creation and simulation of the thalamocortical network
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
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import nest
import nest.topology as tp
import numpy as np
import time
import matplotlib.pyplot as plt
import sys,os
from sys import stdout
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','networks'))

#! ================
#! Class runNetwork
#! ================

class runNetwork(object):

    def __init__(self,simtime):
        # Simulation parameters
        self.Params = {
            'N_LGN': 40, # number of cells per row
            'N_cortex': 80, # number of cells per row
            'visSize': 2.0, # visual angle (degrees)
            'NEST_threads': 8, # threads used in NEST simulation
            'resolution': 1.0, # simulation step (in ms)
            'simtime': simtime # ms
        }

        # Time array
        self.time = np.zeros(int(self.Params['simtime']/self.Params['resolution']))
        for t in np.arange(0,int(self.Params['simtime']/self.Params['resolution'])):
            self.time[t] = (t*self.Params['resolution'])

        # Seeds (the same for all trials)
        np.random.seed(int(time.time()))
        self.seeds = np.arange(self.Params['NEST_threads']) + int((time.time()*100)%2**32)

    #! =================
    #! NEST simulation
    #! =================

    def NESTSimulation(self,retina_spikes):

        # NEST Kernel and Network settings
        nest.ResetKernel()
        nest.ResetNetwork()
        nest.SetKernelStatus(
        {"local_num_threads": self.Params['NEST_threads'],
        'resolution': self.Params['resolution'], "rng_seeds": list(self.seeds)})

        # import network description
        import thalamocortical_system

        # get network info
        models, layers, conns  = thalamocortical_system.get_Network(self.Params)

        # Create models
        for m in models:
                nest.CopyModel(m[0], m[1], m[2])

        print ("\n---Creating layers---\n")
        # Create layers, store layer info in Python variable
        layer_IDs = []
        for l in layers:
            exec ("%s = tp.CreateLayer(%s)" % (l[0],l[1]),globals())
            exec ("copy_var = %s" % l[0],globals())
            layer_IDs.append([l[0],copy_var,l[1]['elements']])
#           print (l[0])

        print ("\n---Connecting layers---\n")
        # Create connections, need to insert variable names
        for c in conns:
                eval('tp.ConnectLayers(%s,%s,c[2])' % (c[0], c[1]))
#                print ('tp.ConnectLayers(%s,%s)' % (c[0], c[1]))

        # Initialize spike generators
        Midget_ganglion_cells_L_ON_spikes = retina_spikes[0]
        Midget_ganglion_cells_L_OFF_spikes = retina_spikes[1]
        Midget_ganglion_cells_M_ON_spikes = retina_spikes[2]
        Midget_ganglion_cells_M_OFF_spikes = retina_spikes[3]
        cell = 0

        for x in np.arange(self.Params['N_LGN']):
            for y in np.arange(self.Params['N_LGN']):
                nest.SetStatus([tp.GetElement(Midget_ganglion_cells_L_ON,(x,y))[0]],
                [{'spike_times':Midget_ganglion_cells_L_ON_spikes[cell],
                'spike_weights':[]}])

                nest.SetStatus([tp.GetElement(Midget_ganglion_cells_L_OFF,(x,y))[0]],
                [{'spike_times':Midget_ganglion_cells_L_OFF_spikes[cell],
                'spike_weights':[]}])

                nest.SetStatus([tp.GetElement(Midget_ganglion_cells_M_ON,(x,y))[0]],
                [{'spike_times':Midget_ganglion_cells_M_ON_spikes[cell],
                'spike_weights':[]}])

                nest.SetStatus([tp.GetElement(Midget_ganglion_cells_M_OFF,(x,y))[0]],
                [{'spike_times':Midget_ganglion_cells_M_OFF_spikes[cell],
                'spike_weights':[]}])

                cell+=1

        ## Check-point: Visualization functions
#        fig = tp.PlotLayer(Color_Luminance_inh_L_ON_L_OFF_vertical,nodesize =80)
#        ctr = tp.FindCenterElement(Parvo_LGN_relay_cell_L_ON)
#        tp.PlotTargets(ctr,Color_Luminance_inh_L_ON_L_OFF_vertical,fig = fig,mask=conns[26][2]['mask'],
#        kernel=conns[26][2]['kernel'],src_size=250,tgt_color='red',tgt_size=20,
#        kernel_color='green')
#        plt.show()

#        ctr = tp.FindCenterElement(Color_Luminance_inh_L_ON_L_OFF_vertical)
#        print ("ctr = ",ctr," L-ON = ",tp.FindCenterElement(Parvo_LGN_relay_cell_L_ON)," L-OFF = ",tp.FindCenterElement(Parvo_LGN_relay_cell_L_OFF))
#        for xx in np.arange(5):
#            for yy in np.arange(5):
#                ctr = [tp.GetElement(Color_Luminance_inh_L_ON_L_OFF_vertical,(xx,yy))[0]]
#                conns = nest.GetConnections(target = [ctr[0]])
#                print ("Cell ",ctr)
#                for nn in np.arange(len(conns)):
#                    print ("conns = ",conns[nn])

#        ctr = tp.FindCenterElement(Parvo_LGN_relay_cell_L_ON)
#        targets = tp.GetTargetNodes(ctr,Color_Luminance_L_ON_L_OFF_vertical)
#        print ("targets = ",targets)

        return layer_IDs

    #! ================================
    #! Recording devices and simulation
    #! ================================

    def runSimulation(self,recorded_models,recorded_spikes,record_Vm):

        recorders = []

        if record_Vm:
            nest.CopyModel('multimeter', 'RecordingNode',
                    {'interval'   : self.Params['resolution'],
                    'record_from': ['V_m'],
                    'record_to'  : ['memory'],
                    'withgid'    : True,
                    'withtime'   : False})

            for population, model in recorded_models:
                    rec = nest.Create('RecordingNode')
                    recorders.append([rec,population,model])
                    tgts = [nd for nd in nest.GetLeaves(population)[0] if nest.GetStatus([nd],
                    'model')[0]==model]
                    nest.Connect(rec, tgts)

        nest.CopyModel('spike_detector', 'RecordingSpikes',
                {"withtime": True,
                "withgid": True,
                "to_file": False})

        spike_detectors = []

        for population, model in recorded_spikes:
                rec = nest.Create('RecordingSpikes')
                spike_detectors.append([rec,population,model])
                tgts = [nd for nd in nest.GetLeaves(population)[0] if nest.GetStatus([nd],
                'model')[0]==model]
                nest.Connect(tgts, rec)

        print ("\n--- Simulation ---\n")
        nest.SetStatus([0],{'print_time': True})
        nest.Simulate(self.Params['simtime'])

        return recorders,spike_detectors

