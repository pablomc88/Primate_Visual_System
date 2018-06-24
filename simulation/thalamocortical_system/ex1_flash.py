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
# Name: ex1_flash
#
# Description: thalamocortical response to light flashes
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

class experiment_1(object):

    def __init__(self):

        ## Simulation parameters ##

        # Simulation time
        self.simtime = 1000.0

        # Simulation object
        self.newSimulation = run_network.runNetwork(self.simtime)

# Number of trials
        self.trials = 2

        # Folder to load/save spike data
        self.spike_folder = 'flashing_square'

        # Stimulus ID
#        self.stim = '_disk_'
        self.stim = '_square_'

        # Start time of plots
        self.start_time = 200.0

        # Record membrane potentials
        self.record_Vm = False

        # Cell to analyze
        self.selected_cell = []
        # Cell to analyze is the center cell in every 2D grid
        self.isCenterCell = True

        # size of each layer (number of cells)
        N_r = self.newSimulation.Params['N_LGN']
        N_c = self.newSimulation.Params['N_cortex']

        # PSTH bin size
        self.bin_size = 10.0 # ms

        # Layers to track
        self.labels = [
        'Parvo_LGN_relay_cell_L_ON',
        'Parvo_LGN_relay_cell_L_OFF',
        'Parvo_LGN_relay_cell_M_ON',
        'Parvo_LGN_relay_cell_M_OFF',

        'Parvo_LGN_interneuron_ON',
        'Parvo_LGN_interneuron_OFF',

        'Color_Luminance_L_ON_L_OFF_vertical',
        'Color_Luminance_L_ON_L_OFF_horizontal',
        'Color_Luminance_L_OFF_L_ON_vertical',
        'Color_Luminance_L_OFF_L_ON_horizontal',
        'Color_Luminance_M_ON_M_OFF_vertical',
        'Color_Luminance_M_ON_M_OFF_horizontal',
        'Color_Luminance_M_OFF_M_ON_vertical',
        'Color_Luminance_M_OFF_M_ON_horizontal',

        'Luminance_preferring_ON_OFF_vertical',
        'Luminance_preferring_ON_OFF_horizontal',
        'Luminance_preferring_OFF_ON_vertical',
        'Luminance_preferring_OFF_ON_horizontal',

        'Color_preferring_L_ON_M_OFF',
        'Color_preferring_M_ON_L_OFF',

        'Color_Luminance_inh_L_ON_L_OFF_vertical',
        'Color_Luminance_inh_L_ON_L_OFF_horizontal',
        'Color_Luminance_inh_L_OFF_L_ON_vertical',
        'Color_Luminance_inh_L_OFF_L_ON_horizontal',
        'Color_Luminance_inh_M_ON_M_OFF_vertical',
        'Color_Luminance_inh_M_ON_M_OFF_horizontal',
        'Color_Luminance_inh_M_OFF_M_ON_vertical',
        'Color_Luminance_inh_M_OFF_M_ON_horizontal',

        'Luminance_preferring_inh_ON_OFF_vertical',
        'Luminance_preferring_inh_ON_OFF_horizontal',
        'Luminance_preferring_inh_OFF_ON_vertical',
        'Luminance_preferring_inh_OFF_ON_horizontal',

        'Color_preferring_inh_L_ON_M_OFF',
        'Color_preferring_inh_M_ON_L_OFF'
        ]

        # Parameters of the topographical plot
        self.top_labels = [
        'Parvo_LGN_relay_cell_L_ON',
        'Parvo_LGN_relay_cell_L_OFF',

        'Color_Luminance_L_ON_L_OFF_vertical',
        'Color_Luminance_M_ON_M_OFF_horizontal',
        'Luminance_preferring_ON_OFF_vertical',
        'Color_preferring_L_ON_M_OFF',

        'Color_Luminance_inh_L_ON_L_OFF_vertical',
        'Luminance_preferring_inh_ON_OFF_vertical',
        'Color_preferring_inh_L_ON_M_OFF'
        ]

        # Average activity of the population
        self.pop_labels = [
        'Color_Luminance_L_ON_L_OFF_vertical',
        'Color_Luminance_L_ON_L_OFF_horizontal',
        'Color_Luminance_L_OFF_L_ON_vertical',
        'Color_Luminance_L_OFF_L_ON_horizontal',
        'Color_Luminance_M_ON_M_OFF_vertical',
        'Color_Luminance_M_ON_M_OFF_horizontal',
        'Color_Luminance_M_OFF_M_ON_vertical',
        'Color_Luminance_M_OFF_M_ON_horizontal',

        'Luminance_preferring_ON_OFF_vertical',
        'Luminance_preferring_ON_OFF_horizontal',
        'Luminance_preferring_OFF_ON_vertical',
        'Luminance_preferring_OFF_ON_horizontal',

        'Color_preferring_L_ON_M_OFF',
        'Color_preferring_M_ON_L_OFF',

        'Color_Luminance_inh_L_ON_L_OFF_vertical',
        'Color_Luminance_inh_L_ON_L_OFF_horizontal',
        'Color_Luminance_inh_L_OFF_L_ON_vertical',
        'Color_Luminance_inh_L_OFF_L_ON_horizontal',
        'Color_Luminance_inh_M_ON_M_OFF_vertical',
        'Color_Luminance_inh_M_ON_M_OFF_horizontal',
        'Color_Luminance_inh_M_OFF_M_ON_vertical',
        'Color_Luminance_inh_M_OFF_M_ON_horizontal',

        'Luminance_preferring_inh_ON_OFF_vertical',
        'Luminance_preferring_inh_ON_OFF_horizontal',
        'Luminance_preferring_inh_OFF_ON_vertical',
        'Luminance_preferring_inh_OFF_ON_horizontal',

        'Color_preferring_inh_L_ON_M_OFF',
        'Color_preferring_inh_M_ON_L_OFF'
        ]

        ## Graphical parameters ##

        self.plot_intracellular = False
        self.plot_PSTH = False
        self.plot_topographical = True

        # Individual intracellular traces
        self.intracellular_rows = 4
        self.intracellular_cols = 5
        self.intracellular_starting_row = 0
        self.intracellular_starting_col = 0

        # PSTHs
        self.PSTH_rows = 4
        self.PSTH_cols = 3
        self.PSTH_starting_row = 0
        self.PSTH_starting_col = 0

        # Topographical plot
        self.topographical_rows = 9
        self.topographical_cols = 5
        self.topographical_time_intervals = [450.0,500.0,550.0,600.0,750.0,800.0]
        self.topographical_V_mins = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0]
        self.topographical_V_maxs = [150., 150., 50.,50.,50.,50.,100.,100.,100.]
        self.pop_V_mins = [0.0]
        self.pop_V_maxs = [10.0]
        self.topographical_isSpikes = True # (False = membrane potential, True = spikes)

        ## End of parameters ##

        # Initialize PSTH
        self.PSTHs = np.zeros((len(self.labels),int(self.simtime/self.bin_size)))

        # IDs from NEST
        self.layers_to_record = []

        # Data recorders used in NEST simulation
        self.potentials = []
        self.spikes = []

        # Retina references
        self.retina_labels = [
                'Midget_ganglion_cells_L_ON',
                'Midget_ganglion_cells_L_OFF',
                'Midget_ganglion_cells_M_ON',
                'Midget_ganglion_cells_M_OFF']
#                'Small_bistratified_ganglion_cells_S_ON']

        # Topographical plot
        self.top_layers_to_record = []
        self.top_PSTHs = np.zeros((int(self.newSimulation.Params['N_cortex']*self.newSimulation.Params['N_cortex']),
        len(self.labels),int(self.simtime/self.bin_size)))
        self.top_PSTH_index = []

        if (self.plot_topographical and self.topographical_isSpikes):
            self.usetop_PSTHs = True # True = update PSTHS for topographical plot (slower)
        else:
            self.usetop_PSTHs = False

        # Population average
        self.pop_layers_to_record = []
        self.pop_PSTH_index = []


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
        self.top_PSTH_index = []
        self.top_layers_to_record = []
        self.pop_PSTH_index = []
        self.pop_layers_to_record = []

        self.layer_sizes = []
        self.layer_sizes_top = []
        self.layer_sizes_pop = []

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

        for layer in self.top_labels:
            # Search first for the matching spiking label
            index = 0
            for spiking_label in self.labels:
                if(layer==spiking_label):
                    self.top_PSTH_index.append(index)
                index+=1

            # Then search for the NEST ID
            id_found = False
            for ll in layer_IDs:
                if (ll[0] == layer):
                    self.top_layers_to_record.append((ll[1],ll[2]))
                    self.layer_sizes_top.append(len(nest.GetNodes(ll[1])[0]))
                    id_found = True
            if id_found == False:
                # Assign random layer
                self.top_layers_to_record.append((ll[1],ll[2]))
                print ("Warning: layer %s not found!" % layer)

        for layer in self.pop_labels:
            # Search first for the matching spiking label
            index = 0
            for spiking_label in self.labels:
                if(layer==spiking_label):
                    self.pop_PSTH_index.append(index)
                index+=1

            # Then search for the NEST ID
            id_found = False
            for ll in layer_IDs:
                if (ll[0] == layer):
                    self.pop_layers_to_record.append((ll[1],ll[2]))
                    self.layer_sizes_pop.append(len(nest.GetNodes(ll[1])[0]))
                    id_found = True
            if id_found == False:
                # Assign random layer
                self.pop_layers_to_record.append((ll[1],ll[2]))
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



    # NEST simulation
    def NESTSimulation(self,trial):
        layer_IDs = self.newSimulation.NESTSimulation(data_analysis.loadSpikes(self.newSimulation.Params['N_LGN'],
        self.retina_labels,self.spike_folder,"retina",self.stim,trial,0))
        self.loadLayers(layer_IDs)

        [self.potentials,self.spikes] = self.newSimulation.runSimulation(self.layers_to_record,
        self.layers_to_record,self.record_Vm)

        data_analysis.updatePSTH(self.simtime,self.spikes,self.layers_to_record,self.selected_cell,
        self.PSTHs,self.bin_size,self.layer_sizes)

        if self.usetop_PSTHs:
            # Selected cells are passed as a list although the values of this list
            # are not used. layer_sizes is used instead.
            data_analysis.updatePSTH(self.simtime,self.spikes,self.layers_to_record,
            list(np.arange(2)),self.top_PSTHs,self.bin_size,self.layer_sizes)

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

        # Topographical plot
        if self.plot_topographical:

            fig = plt.figure()
            fig.subplots_adjust(hspace=1.5)
            fig.subplots_adjust(wspace=0.4)

            data_analysis.topographical(fig,self.newSimulation.Params['N_cortex'],self.topographical_time_intervals,
            self.newSimulation.Params['resolution'],self.simtime,self.spikes,
            self.top_layers_to_record,self.top_labels,self.topographical_rows,self.topographical_cols,
            self.topographical_V_mins,self.topographical_V_maxs,self.topographical_isSpikes,
            self.trials,self.top_PSTHs,self.bin_size,self.top_PSTH_index,self.layer_sizes_top)

            data_analysis.topographical(fig,self.newSimulation.Params['N_cortex'],self.topographical_time_intervals,
            self.newSimulation.Params['resolution'],self.simtime,self.spikes,
            self.pop_layers_to_record,self.pop_labels,self.topographical_rows,self.topographical_cols,
            self.pop_V_mins,self.pop_V_maxs,self.topographical_isSpikes,
            self.trials,self.top_PSTHs,self.bin_size,self.pop_PSTH_index,self.layer_sizes_pop,True)

        plt.show()

    def saveSpikes(self,stim,folder,trial):
        print ("\n--- Saving spikes ---\n")
        data_analysis.saveSpikes(self.newSimulation.Params['N_cortex'],self.spikes,
        self.layers_to_record,self.labels,folder,"thalamocortical_system",stim,trial,
        self.layer_sizes)


#! =================
#! Main
#! =================

if __name__ == '__main__':

    ex1 = experiment_1()
    ex1.initializeFolders()
    for trial in np.arange(ex1.trials):
        print ("\n--- Trial %s ---\n" % trial)
        ex1.NESTSimulation(trial)
        ex1.saveSpikes(ex1.stim,ex1.spike_folder,trial)
    ex1.plotResults()
