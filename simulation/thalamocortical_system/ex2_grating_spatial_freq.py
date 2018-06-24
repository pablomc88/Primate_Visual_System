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
# Name: ex2_grating_spatial_freq
#
# Description: thalamocortical response to sine-wave gratings of varying spatial
# frequency
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
import scipy.fftpack
import sys,os,os.path
import matplotlib.pyplot as plt
from sys import stdout

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data_analysis
import run_network

class experiment_2(object):

    def __init__(self):

        ## Simulation parameters ##

        # Simulation time
        self.simtime = 1000.0

        # Simulation object
        self.newSimulation = run_network.runNetwork(self.simtime)

        # Number of trials
        self.trials = 2

        # Folder to save spike times
        self.spike_folder = 'Luminance_Grating'

        # Grating parameters
#        self.spatial_frequency = np.array([0.1,0.2,0.3,0.4,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]) # cpd
        self.spatial_frequency = np.array([1.0,3.0]) # cpd
        self.temporal_frequency = 2.0 # Hz

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

        # FFT labels
        self.FFT_labels = [
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

        # FFT plot
        self.FFT_rows = 3
        self.FFT_cols = 5

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
                'Midget_ganglion_cells_M_OFF',
                'Small_bistratified_ganglion_cells_S_ON']

        # FFT
        self.FFTamp = np.zeros((len(self.FFT_labels),len(self.spatial_frequency)))
        self.FFTph = np.zeros((len(self.FFT_labels),len(self.spatial_frequency)))
        self.FFT_recorded_models = []
        self.FFT_PSTH_index = []


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
        self.FFT_PSTH_index = []
        self.FFT_recorded_models = []

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

        for layer in self.FFT_labels:
            # Search first for the matching spiking label
            index = 0
            for spiking_label in self.labels:
                if(layer==spiking_label):
                    self.FFT_PSTH_index.append(index)
                index+=1

            # Then search for the NEST ID
            id_found = False
            for ll in layer_IDs:
                if (ll[0] == layer):
                    self.FFT_recorded_models.append((ll[1],ll[2]))
                    id_found = True
            if id_found == False:
                # Assign random layer
                self.FFT_recorded_models.append((ll[1],ll[2]))
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

    # Compute FFT and get first harmonic
    def firstHarmonic(self,freq,loaded_spikes = False):
        counter = 0
        sp_type = 0
        for population, model in self.FFT_recorded_models:

            if(loaded_spikes == False):
                [data,selected_senders,pop] = data_analysis.getData(population,
                model,self.spikes,[self.selected_cell[self.FFT_PSTH_index[sp_type]]])
                spike_times = (data[0]['times'])[selected_senders[0]]
            else:
                spike_times = [self.start_time] # valid only with the standard PSTH

            spike_times = spike_times[np.where(spike_times > self.start_time)[0]]

            # Single-trial estimation of PSTH
            if self.trials ==1:
                [PSTH_times,PSTH_array] = data_analysis.singleTrialPSTH(self.start_time,self.simtime,spike_times)
                # Fourier sample spacing
                T = 0.001 # s

            # Standard PSTH
            else:
                PSTH_array = (1000.0/self.bin_size) * self.PSTHs[self.FFT_PSTH_index[sp_type],
                int(self.start_time/self.bin_size):]/self.trials
                # Fourier sample spacing
                T = self.bin_size*0.001 # s

            sp_type+=1

            response = PSTH_array

            # Fourier fundamental frequency
            # Number of samplepoints
            N = len(response)
            # FFT
            yf = scipy.fftpack.fft(response)
            phase = np.angle(yf[:N//2]) *(180.0/np.pi)
            yf = 2.0/N * np.abs(yf[:N//2])
            xf = np.array(np.linspace(0.0, 1.0/(2.0*T), N/2))

            main_freq = np.where(xf>=self.temporal_frequency)[0][0]

            # To correct small deviations
            ampl_possible_choices = [yf[main_freq-1],yf[main_freq],yf[main_freq+1]]
            phase_possible_choices = [phase[main_freq-1],phase[main_freq],phase[main_freq+1]]

    #            print ("model = ",model," , freq = ",self.temporal_frequency," , ampl selected = ",np.max(ampl_possible_choices))
    #            plt.plot(xf,yf)
    #            plt.show()

            self.FFTamp[counter, np.where(self.spatial_frequency==freq)[0][0]] = np.max(ampl_possible_choices)
            self.FFTph[counter, np.where(self.spatial_frequency==freq)[0][0]] = phase_possible_choices[
            np.argmax(ampl_possible_choices)]

    #            self.FFTamp[counter, np.where(self.spatial_frequency==freq)[0][0]] = yf[main_freq]
    #            self.FFTph[counter, np.where(self.spatial_frequency==freq)[0][0]] = phase[main_freq]

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

        data_analysis.spatialTuning(self.spatial_frequency,self.FFTamp,self.FFTph,
        self.FFT_labels,self.FFT_rows,self.FFT_cols,0,0,'Spatial frequency (cpd)',
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

    ex2 = experiment_2()
    ex2.initializeFolders()

    # Simulation
    for f in ex2.spatial_frequency:
        print ("\n--- Freq = %s  cpd ---\n" % f)

        for trial in np.arange(ex2.trials):
            print ("\n--- Trial %s ---\n" % trial)
            ex2.NESTSimulation(trial,'_sf_'+str(f))
            ex2.saveSpikes('_sf_'+str(f),ex2.spike_folder,trial)

        ex2.firstHarmonic(f)
        ex2.plotIntermediateResults()
        ex2.resetPSTHs()

    # Plot final results
    ex2.plotFinalResults()
