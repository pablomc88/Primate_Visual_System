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
# Name: ex3_disk_area_response
#
# Description: thalamocortical response to flashing spots of varying diameter
# plotted as response versus spot diameter (area-response curve)
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

class experiment_3(object):

    def __init__(self):

        ## Simulation parameters ##

        # Simulation time
        self.simtime = 1000.0

        # Simulation object
        self.newSimulation = run_network.runNetwork(self.simtime)

        # Number of trials
        self.trials = 2

        # Folder to save spike times
        self.spike_folder = 'areaResponseCurves'

        # Start time of plots
        self.start_time = 200.0

        # Pulse parameters
        self.pulse_duration = 500.0 # ms
        self.pulse_tstart = 500.0 # ms (first 300 ms are used to fill the input and
                                # output buffers of linear filters)

        # Record membrane potentials
        self.record_Vm = False

        # disk parameters
#        self.disk_diameters = [0.0,0.05,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0] # degrees
        self.disk_diameters = [0.15,0.5] # degrees

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
#        'Parvo_LGN_relay_cell_L_OFF',
        'Parvo_LGN_relay_cell_M_ON',
#        'Parvo_LGN_relay_cell_M_OFF',

#        'Parvo_LGN_interneuron_ON',
#        'Parvo_LGN_interneuron_OFF',

        'Color_Luminance_L_ON_L_OFF_vertical',
        'Color_Luminance_L_ON_L_OFF_horizontal',
        'Color_Luminance_L_OFF_L_ON_vertical',
#        'Color_Luminance_L_OFF_L_ON_horizontal',
        'Color_Luminance_M_ON_M_OFF_vertical',
#        'Color_Luminance_M_ON_M_OFF_horizontal',
        'Color_Luminance_M_OFF_M_ON_vertical',
#        'Color_Luminance_M_OFF_M_ON_horizontal',

        'Luminance_preferring_ON_OFF_vertical',
        'Luminance_preferring_ON_OFF_horizontal',
        'Luminance_preferring_OFF_ON_vertical',
        'Luminance_preferring_OFF_ON_horizontal',

        'Color_preferring_L_ON_M_OFF',
        'Color_preferring_M_ON_L_OFF',

        'Color_Luminance_inh_L_ON_L_OFF_vertical',
#        'Color_Luminance_inh_L_ON_L_OFF_horizontal',
#        'Color_Luminance_inh_L_OFF_L_ON_vertical',
#        'Color_Luminance_inh_L_OFF_L_ON_horizontal',
#        'Color_Luminance_inh_M_ON_M_OFF_vertical',
#        'Color_Luminance_inh_M_ON_M_OFF_horizontal',
#        'Color_Luminance_inh_M_OFF_M_ON_vertical',
#        'Color_Luminance_inh_M_OFF_M_ON_horizontal',

        'Luminance_preferring_inh_ON_OFF_vertical'
#        'Luminance_preferring_inh_ON_OFF_horizontal',
#        'Luminance_preferring_inh_OFF_ON_vertical',
#        'Luminance_preferring_inh_OFF_ON_horizontal',

#        'Color_preferring_inh_L_ON_M_OFF',
#        'Color_preferring_inh_M_ON_L_OFF'
        ]

        # Area-response labels
        self.area_labels = [
        'Parvo_LGN_relay_cell_L_ON',
#        'Parvo_LGN_relay_cell_L_OFF',
        'Parvo_LGN_relay_cell_M_ON',
#        'Parvo_LGN_relay_cell_M_OFF',

#        'Parvo_LGN_interneuron_ON',
#        'Parvo_LGN_interneuron_OFF',

        'Color_Luminance_L_ON_L_OFF_vertical',
        'Color_Luminance_L_ON_L_OFF_horizontal',
        'Color_Luminance_L_OFF_L_ON_vertical',
#        'Color_Luminance_L_OFF_L_ON_horizontal',
        'Color_Luminance_M_ON_M_OFF_vertical',
#        'Color_Luminance_M_ON_M_OFF_horizontal',
        'Color_Luminance_M_OFF_M_ON_vertical',
#        'Color_Luminance_M_OFF_M_ON_horizontal',

        'Luminance_preferring_ON_OFF_vertical',
        'Luminance_preferring_ON_OFF_horizontal',
        'Luminance_preferring_OFF_ON_vertical',
        'Luminance_preferring_OFF_ON_horizontal',

        'Color_preferring_L_ON_M_OFF',
        'Color_preferring_M_ON_L_OFF',

        'Color_Luminance_inh_L_ON_L_OFF_vertical',
#        'Color_Luminance_inh_L_ON_L_OFF_horizontal',
#        'Color_Luminance_inh_L_OFF_L_ON_vertical',
#        'Color_Luminance_inh_L_OFF_L_ON_horizontal',
#        'Color_Luminance_inh_M_ON_M_OFF_vertical',
#        'Color_Luminance_inh_M_ON_M_OFF_horizontal',
#        'Color_Luminance_inh_M_OFF_M_ON_vertical',
#        'Color_Luminance_inh_M_OFF_M_ON_horizontal',

        'Luminance_preferring_inh_ON_OFF_vertical'
#        'Luminance_preferring_inh_ON_OFF_horizontal',
#        'Luminance_preferring_inh_OFF_ON_vertical',
#        'Luminance_preferring_inh_OFF_ON_horizontal',

#        'Color_preferring_inh_L_ON_M_OFF',
#        'Color_preferring_inh_M_ON_L_OFF'
        ]

        ## Graphical parameters ##

        self.plot_intracellular = False
        self.plot_PSTH = False

        # Individual intracellular traces
        self.intracellular_rows = 4
        self.intracellular_cols = 5
        self.intracellular_starting_row = 0
        self.intracellular_starting_col = 0

        # PSTHs
        self.PSTH_rows = 4
        self.PSTH_cols = 5
        self.PSTH_starting_row = 0
        self.PSTH_starting_col = 0

        # Area-response curve
        self.area_rows = 3
        self.area_cols = 5

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
                'Midget_ganglion_cells_M_OFF',
                'Small_bistratified_ganglion_cells_S_ON']


        # Area-response curve
        self.area_amp = np.zeros((len(self.area_labels),len(self.disk_diameters)))
        self.area_ph = np.zeros((len(self.area_labels),len(self.disk_diameters)))
        self.area_recorded_models = []
        self.area_PSTH_index = []

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
        self.area_PSTH_index = []
        self.area_recorded_models = []

        self.layer_sizes = []
        self.layer_sizes_area = []

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


        for layer in self.area_labels:
            # Search first for the matching spiking label
            index = 0
            for spiking_label in self.labels:
                if(layer==spiking_label):
                    self.area_PSTH_index.append(index)
                index+=1

            # Then search for the NEST ID
            id_found = False
            for ll in layer_IDs:
                if (ll[0] == layer):
                    self.area_recorded_models.append((ll[1],ll[2]))
                    self.layer_sizes_area.append(len(nest.GetNodes(ll[1])[0]))
                    id_found = True
            if id_found == False:
                # Assign random layer
                self.area_recorded_models.append((ll[1],ll[2]))
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


    # NEST simulation
    def NESTSimulation(self,trial,stim):
        layer_IDs = self.newSimulation.NESTSimulation(data_analysis.loadSpikes(self.newSimulation.Params['N_LGN'],
        self.retina_labels,self.spike_folder,"retina",stim,trial,0))
        self.loadLayers(layer_IDs)

        [self.potentials,self.spikes] = self.newSimulation.runSimulation(self.layers_to_record,
        self.layers_to_record,self.record_Vm)

        data_analysis.updatePSTH(self.simtime,self.spikes,self.layers_to_record,self.selected_cell,
        self.PSTHs,self.bin_size,self.layer_sizes)

    # Update area-response curve
    def areaResponseCurve(self,diameter,loaded_spikes = False):
        counter = 0
        sp_type = 0

        for population, model in self.area_recorded_models:

            if(loaded_spikes == False):
                [data,selected_senders,pop] = data_analysis.getData(population,
                model,self.spikes,[self.selected_cell[counter]])
                spike_times = (data[0]['times'])[selected_senders[0]]
            else:
                spike_times = [self.start_time] # valid only with the standard PSTH

            spike_times = spike_times[np.where(spike_times > self.start_time)[0]]

            # Single-trial estimation of PSTH
            if self.trials ==1:
                [PSTH_times,PSTH_array] = data_analysis.singleTrialPSTH(self.pulse_tstart,self.simtime,spike_times)

            # Standard PSTH
            else:
                PSTH_array = (1000.0/self.bin_size) * self.PSTHs[self.area_PSTH_index[sp_type],
                int(self.pulse_tstart/self.bin_size):]/self.trials
                sp_type+=1

            response = PSTH_array

            self.area_amp[counter, self.disk_diameters.index(diameter)] = np.mean(response)
            self.area_ph[counter, self.disk_diameters.index(diameter)] = 0.0

            counter+=1

    # Reset PSTHs for every different stimulus
    def resetPSTHs(self):
        self.PSTHs = np.zeros((len(self.labels),int(self.simtime/self.bin_size)))

    # Plot results
    def plotIntermediateResults(self):

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

    def plotFinalResults(self):

        print ("\n--- Plotting results ---\n")
        fig = plt.figure()

        data_analysis.spatialTuning(self.disk_diameters,self.area_amp,self.area_ph,
        self.area_labels,self.area_rows,self.area_cols,0,0,'Disk diameter (deg)',
        "Ampl. (mV or s^(-1))",'thalamocortical_system')

        plt.show()

    # Save spike times
    def saveSpikes(self,stim,folder,trial):
        print ("\n--- Saving spikes ---\n")
        data_analysis.saveSpikes(self.newSimulation.Params['N_cortex'],self.spikes,
        self.layers_to_record,self.labels,folder,"thalamocortical_system",stim,trial,
        self.layer_sizes)


#! =================
#! Main
#! =================

if __name__ == '__main__':

    ex3 = experiment_3()
    ex3.initializeFolders()

    for d in ex3.disk_diameters:
        print ("\n--- Diameter = %s  degrees ---\n" % d)

        for trial in np.arange(ex3.trials):
            print ("\n--- Trial %s ---\n" % trial)
            ex3.NESTSimulation(trial,'_disk_'+str(d))
            ex3.saveSpikes('_disk_'+str(d),ex3.spike_folder,trial)

        ex3.areaResponseCurve(d)
        ex3.plotIntermediateResults()
        ex3.resetPSTHs()

    # Plot area-response curves
    ex3.plotFinalResults()
