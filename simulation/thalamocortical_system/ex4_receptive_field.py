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
# Name: ex4_receptive_field
#
# Warning: software not tested!
#
# Description: estimation of spatiotemporal receptive-fields
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
import numpy as np
import sys,os,os.path
import matplotlib.pyplot as plt
from sys import stdout

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data_analysis
import run_network

class experiment_4(object):

    def __init__(self):

        ## Simulation parameters ##

        # Simulation time
        self.simtime = 600.0

        # Simulation object
        self.newSimulation = run_network.runNetwork(self.simtime)

        # Number of trials
        self.trials = 2

        # Folder to save spike times
        self.spike_folder = 'Receptive_field'

        # Square mask where stimuli are displayed (side = 2*mask_side+1)
        self.mask_side = 4

        # Intervals to average
        self.RF_intervals = [[320.0,340.0],[340.0,360.0],[360.0,380.0],[380.0,400.0],[400.0,420.0]]

        # Start time of plots
        self.start_time = 200.0

        # Record membrane potentials
        self.record_Vm = False

        # Cell to analyze
        self.isCenterCell = True
        self.selected_cell = []

        # size of each layer (number of cells)
        N_r = self.newSimulation.Params['N_LGN']
        N_c = self.newSimulation.Params['N_cortex']

        # PSTH bin size
        self.bin_size = 20.0 # ms

        # Layers to track
        self.labels = [
        'Parvo_LGN_relay_cell_L_ON',
        'Parvo_LGN_relay_cell_L_OFF',

        'Color_Luminance_L_ON_L_OFF_vertical',
        'Luminance_preferring_ON_OFF_vertical',
        'Color_preferring_L_ON_M_OFF'
        ]

        ## Graphical parameters ##

        self.plot_intracellular = False
        self.plot_PSTH = False

        # Individual intracellular traces
        self.intracellular_rows = 1
        self.intracellular_cols = 5
        self.intracellular_starting_row = 0
        self.intracellular_starting_col = 0

        # PSTHs
        self.PSTH_rows = 1
        self.PSTH_cols = 5
        self.PSTH_starting_row = 0
        self.PSTH_starting_col = 0

        ## End of parameters ##

        # Initialize PSTH
        self.PSTHs = np.zeros((len(self.labels),int(self.simtime/self.bin_size)))

        # IDs from NEST
        self.layers_to_record = [] # For membrane potentials

        # Data recorders used in NEST simulation
        self.potentials = []
        self.spikes = []

        # Retina references
        self.retina_labels = [
                'Midget_ganglion_cells_L_ON',
                'Midget_ganglion_cells_L_OFF',
                'Midget_ganglion_cells_M_ON',
                'Midget_ganglion_cells_M_OFF']


        # Receptive_field
        self.RF_mask = []

        self.RF_bright = np.zeros((len(self.RF_intervals),len(self.labels),
        N_r,N_r))

        self.RF_dark = np.zeros((len(self.RF_intervals),len(self.labels),
        N_r,N_r))


    # Initialize/clean folders
    def initializeFolders(self):

        if os.path.isdir("../../data/thalamocortical_system/results"):
            if len(os.listdir("../../data/thalamocortical_system/results")) > 0:
                os.system("rm -r ../../data/thalamocortical_system/results/*")
        else:
            os.system("mkdir ../../data/thalamocortical_system/results")

        if os.path.isdir("../../data/thalamocortical_system/data"):
            if len(os.listdir("../../data/thalamocortical_system/data")) > 0:
                os.system("rm -r ../../data/thalamocortical_system/data/*")
        else:
            os.system("mkdir ../../data/thalamocortical_system/data")

        if os.path.isdir("../../data/thalamocortical_system/spikes")==False:
            os.system("mkdir ../../data/thalamocortical_system/spikes")

        if os.path.isdir("../../data/thalamocortical_system/spikes/"+self.spike_folder):
            if len(os.listdir("../../data/thalamocortical_system/spikes/"+self.spike_folder)) > 0:
                os.system("rm -r ../../data/thalamocortical_system/spikes/"+self.spike_folder+"/*")
        else:
            os.system("mkdir ../../data/thalamocortical_system/spikes/"+self.spike_folder)


    # Initialize arrays with NEST IDs
    def loadLayers(self,layer_IDs):

        self.layers_to_record = []
        self.layer_sizes = []

        for layer in self.labels:
            id_found = False
            for ll in layer_IDs:
                if (ll[0] == layer):
                    self.layers_to_record.append((ll[1],ll[2]))
                    self.layer_sizes.append(len(nest.GetNodes(ll[1])[0]))
                    id_found = True
            if id_found == False:
                # Assign random layer
                self.layers_to_record.append((ll[1],ll[2]))
                print ("Warning: layer %s not found!" % layer)

        # center cell
        if self.isCenterCell:

            type = 0
            for layer in self.labels:
                layer_side = int(np.sqrt(self.layer_sizes[type]))
                center_row = int(layer_side/2.0)
                center_col = int(layer_side/2.0)

                for cell in np.arange(self.layer_sizes[type]):
                    row = int(cell/layer_side)
                    col = np.remainder(cell,layer_side)
                    if row == center_row and col == center_col:
                        self.selected_cell.append(cell)
                type+=1

        #            print ("self.selected_cell = ",self.selected_cell)

    # Compute mask
    def createRFmask(self):

        l_side = self.newSimulation.Params['N_LGN']
        center_row = int(l_side/2.0)
        center_col = int(l_side/2.0)

        for cell in np.arange(l_side*l_side):
            row = int(cell/l_side)
            col = np.remainder(cell,l_side)

            if(row >= center_row - self.mask_side and row<= center_row + self.mask_side and
            col >= center_col - self.mask_side and col<= center_col + self.mask_side):
                self.RF_mask.append([row,col])


#        print ("self.RF_mask = ",self.RF_mask)

    # NEST simulation
    def NESTSimulation(self,trial,stim):
        layer_IDs = self.newSimulation.NESTSimulation(data_analysis.loadSpikes(self.newSimulation.Params['N_LGN'],
        self.retina_labels,self.spike_folder,"retina",stim,trial,0))
        self.loadLayers(layer_IDs)

        [self.potentials,self.spikes] = self.newSimulation.runSimulation(self.layers_to_record,
        self.layers_to_record,self.record_Vm)

        data_analysis.updatePSTH(self.simtime,self.spikes,self.layers_to_record,self.selected_cell,
        self.PSTHs,self.bin_size,self.layer_sizes)

    # Plot results
    def plotResults(self):

        print ("\n--- Plotting results ---\n")

        # Individual intracellular traces
        if self.plot_intracellular and self.record_Vm:

            fig = plt.figure()
            fig.subplots_adjust(hspace=1.5)
            fig.subplots_adjust(wspace=0.4)

            start_pos = int(self.start_time/self.newSimulation.Params['resolution'])
            time = self.newSimulation.time[start_pos:len(self.newSimulation.time)]

            data_analysis.membranePotentials(self.start_time,self.newSimulation.Params['resolution'],
            self.simtime,self.potentials,self.layers_to_record,self.labels,self.selected_cell,self.intracellular_rows,self.intracellular_cols,self.intracellular_starting_row,self.intracellular_starting_col,"thalamocortical_system")


        # PSTHs
        if self.plot_PSTH:

            fig = plt.figure()
            fig.subplots_adjust(hspace=1.5)
            fig.subplots_adjust(wspace=0.4)

            start_pos = int(self.start_time/self.newSimulation.Params['resolution'])
            time = self.newSimulation.time[start_pos:len(self.newSimulation.time)]

            data_analysis.PSTH(self.start_time,self.newSimulation.Params['resolution'],
            self.simtime,self.spikes,self.layers_to_record,self.labels,self.selected_cell,
            self.PSTH_rows,self.PSTH_cols,self.PSTH_starting_row,self.PSTH_starting_col,
            self.trials,self.PSTHs,self.bin_size,"thalamocortical_system")

        plt.show()

    # Save spike times
    def saveSpikes(self,stim,folder,trial):
        print ("\n--- Saving spikes ---\n")
        data_analysis.saveSpikes(self.newSimulation.Params['N_cortex'],self.spikes,
        self.layers_to_record,self.labels,folder,"thalamocortical_system",stim,trial,
        self.layer_sizes)

    def updateRFmaps(self,pos,dark,loaded_spikes = False):

        row = self.RF_mask[pos][0]
        col = self.RF_mask[pos][1]
        RF_index = 0

        for interval in self.RF_intervals:
            counter = 0

            for population, model in self.layers_to_record:

                if(loaded_spikes == False):
                    [data,selected_senders,pop] = data_analysis.getData(population,model, self.spikes[self.selected_cell[counter]])
                    spike_times = (data[0]['times'])[selected_senders[0]]
                else:
                    spike_times = [self.start_time] # valid only with the standard PSTH

                # Single-trial estimation of PSTH
                if self.trials ==1:
                    [PSTH_times,PSTH_array] = data_analysis.singleTrialPSTH(0.0,self.simtime,spike_times)
                    response = np.mean(PSTH_array[int(interval[0]):int(interval[1])])

                # Standard PSTH
                else:
                    PSTH_array = (1000.0/self.bin_size) * self.PSTHs[counter,:]/self.trials
                    if int(interval[0]/self.bin_size) == int(interval[1]/self.bin_size):
                        response = np.mean(PSTH_array[int(interval[0]/self.bin_size):int(interval[0]/self.bin_size) + 1])
                    else:
                        response = np.mean(PSTH_array[int(interval[0]/self.bin_size):int(interval[1]/self.bin_size)])

                if dark==0:
                    self.RF_bright[RF_index,counter,row,col] = response
                else:
                    self.RF_dark[RF_index,counter,row,col] = response

                counter+=1

            RF_index+=1


    def receptiveField(self):
        fig = plt.figure()
        data_analysis.receptiveField(fig,self.newSimulation.Params['N_LGN'],
        self.RF_intervals,self.layers_to_record,
        self.labels,self.RF_bright,self.RF_dark)
        plt.show()

    # Reset PSTHs for every different stimulus
    def resetPSTHs(self):
        self.PSTHs = np.zeros((len(self.labels),int(self.simtime/self.bin_size)))

#! =================
#! Main
#! =================

if __name__ == '__main__':

    ex4 = experiment_4()
    ex4.initializeFolders()
    ex4.createRFmask()

    for pos in np.arange(len(ex4.RF_mask)):
        print ("pos = %s" % pos)

        # Bright stimulus
        for trial in np.arange(ex4.trials):
            print ("\n--- Trial %s ---\n" % trial)
            ex4.NESTSimulation(trial,"_disk_"+"_bright_"+str(pos))
            ex4.saveSpikes("_disk_"+"_bright_"+str(pos),ex4.spike_folder,trial)

        ex4.updateRFmaps(pos,0)
        ex4.plotResults()
        ex4.resetPSTHs()

        # Dark stimulus
        for trial in np.arange(ex4.trials):
            print ("\n--- Trial %s ---\n" % trial)
            ex4.NESTSimulation(trial,"_disk_"+"_dark_"+str(pos))
            ex4.saveSpikes("_disk_"+"_dark_"+str(pos),ex4.spike_folder,trial)

        ex4.updateRFmaps(pos,1)
        ex4.plotResults()
        ex4.resetPSTHs()

    ex4.receptiveField()
