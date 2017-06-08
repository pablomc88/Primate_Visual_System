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
# Name: ex5_receptive_field
#
# Description: Estimation of spatiotemporal receptive-fields. The space is
# divided in a grid of sampling points and bright and dark spots are briefly
# flashed for 40 ms at every point, measuring the response to the center cell.
# A composite receptive-field profile is obtained by computing the difference
# between the responses the bright and dark stimuli [2]
#
# References:
#
# [1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
# Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
# 2017
#
# [2] DeAngelis, Gregory C., Izumi Ohzawa, and R. D. Freeman. "Spatiotemporal
# organization of simple-cell receptive fields in the cat's striate cortex. I.
# General characteristics and postnatal development." Journal of neurophysiology
# 69.4 (1993): 1091-1117.
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import numpy as np
import sys,os,os.path
import matplotlib.pyplot as plt
from sys import stdout

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data_analysis
reload(data_analysis)

import run_network
reload(run_network)

class experiment_5(object):

    def __init__(self):

        ## Simulation parameters ##

        # Simulation time
        self.simtime = 600.0

        # Number of trials
        self.trials = 5

        # Folder to save spike times
        self.spike_folder = 'receptive_field'

        # Stimulus type
        self.stim = '_disk_'

        # Square mask where stimuli are displayed (side = 2*mask_side+1)
        self.mask_side = 1

        # Intervals to average
        self.RF_intervals = [[310.0,350.0],[350.0,390.0],[400.0,440.0]]

        # Start time of plots
        self.start_time = 200.0

        # disk parameters
        self.radius = 0.09 # degrees

        # Pulse parameters
        self.pulse_duration = 40.0 # ms
        self.pulse_tstart = 300.0 # ms (first 300 ms are used to fill the input and
                                # output buffers of linear filters)
        self.bkg_illuminance = 250.0 # td
        self.pulse_contrast = 1.0
        self.pulse_amplitude = self.pulse_contrast * self.bkg_illuminance # td

        # Cell to analyze
        self.selected_cell = 210

        # PSTH bin size
        self.bin_size = 40.0 # ms

        # Layers to track (labels for figures)
        self.labels = [
        'Midget_ganglion_cells_L_ON',
        'Midget_ganglion_cells_L_OFF',
        'Midget_ganglion_cells_M_ON',
        'Midget_ganglion_cells_M_OFF'
        ]

        # Spiking layers
        self.sp_labels = [
        'Midget_ganglion_cells_L_ON',
        'Midget_ganglion_cells_L_OFF',
        'Midget_ganglion_cells_M_ON',
        'Midget_ganglion_cells_M_OFF'
        ]

        # Parameters of the topographical plot
        self.top_labels = [
        'Midget_ganglion_cells_L_ON',
        'Midget_ganglion_cells_L_OFF'
        ]

        ## Graphical parameters ##

        self.plot_intracellular = False
        self.plot_PSTH = False
        self.plot_topographical = False

        # Individual intracellular traces
        self.intracellular_rows = 3
        self.intracellular_cols = 4
        self.intracellular_video_step = 1.0 # ms
        self.intracellular_starting_row = 2
        self.intracellular_starting_col = 2

        # PSTHs
        self.PSTH_rows = 3
        self.PSTH_cols = 2
        self.PSTH_starting_row = 0
        self.PSTH_starting_col = 1

        # Topographical plot
        self.topographical_rows = 2
        self.topographical_cols = 4
        self.topographical_time_intervals = [0.0,300.0,350.0,400.0,600.0]
        self.topographical_V_mins = [0.0,0.0]
        self.topographical_V_maxs = [150.0,150.0]
        self.topographical_isSpikes = True # (False = membrane potential, True = spikes)

        ## End of parameters ##

        # Initialize PSTH
        self.PSTHs = np.zeros((len(self.sp_labels),int(self.simtime/self.bin_size)))

        # IDs from NEST
        self.layers_to_record = [] # For membrane potentials
        self.s_layers_to_record = [] # For spikes

        # Data recorders used in NEST simulation
        self.potentials = []
        self.spikes = []

        # Simulation object
        self.newSimulation = run_network.runNetwork(self.simtime)

        # Input arrays
        self.L_cone_input = np.zeros((self.newSimulation.Params['N']*self.newSimulation.Params['N'],
        int(self.newSimulation.Params['simtime']/self.newSimulation.Params['resolution'])))
        self.M_cone_input = np.zeros((self.newSimulation.Params['N']*self.newSimulation.Params['N'],
        int(self.newSimulation.Params['simtime']/self.newSimulation.Params['resolution'])))
        self.S_cone_input = np.zeros((self.newSimulation.Params['N']*self.newSimulation.Params['N'],
        int(self.newSimulation.Params['simtime']/self.newSimulation.Params['resolution'])))
        # Video of the input stimulus
        self.show_video = False
        self.inputIm = np.zeros((self.newSimulation.Params['N'],self.newSimulation.Params['N'],
        int(self.newSimulation.Params['simtime']/self.newSimulation.Params['resolution'])))

        # Retinal mosaic (default: uniform distribution)
        self.mosaic = np.ones(((3,self.newSimulation.Params['N'],self.newSimulation.Params['N'])))

        # Topographical plot
        self.top_layers_to_record = []
        self.top_PSTHs = np.zeros((int(self.newSimulation.Params['N']*self.newSimulation.Params['N']),
        len(self.sp_labels),int(self.simtime/self.bin_size)))
        self.top_PSTH_index = []

        if (self.plot_topographical and self.topographical_isSpikes):
            self.usetop_PSTHs = True # True = update PSTHS for topographical plot (slower)
        else:
            self.usetop_PSTHs = False

        # Receptive_field
        self.RF_mask = []

        self.RF_bright = np.zeros((len(self.RF_intervals),len(self.labels),
        self.newSimulation.Params['N'],self.newSimulation.Params['N']))

        self.RF_dark = np.zeros((len(self.RF_intervals),len(self.labels),
        self.newSimulation.Params['N'],self.newSimulation.Params['N']))

    # Initialize/clean folders
    def initializeFolders(self):

        if os.path.isdir("../../data/retina/results") and self.newSimulation.Params['load_cone_from_file']==False:
            if len(os.listdir("../../data/retina/results")) > 0:
                os.system("rm -r ../../data/retina/results/*")
        else:
            os.system("mkdir ../../data/retina/results")

        if os.path.isdir("../../data/retina/input"):
            if len(os.listdir("../../data/retina/input")) > 0:
                os.system("rm -r ../../data/retina/input/*")
        else:
            os.system("mkdir ../../data/retina/input")

        if os.path.isdir("../../data/retina/data"):
            if len(os.listdir("../../data/retina/data")) > 0:
                os.system("rm -r ../../data/retina/data/*")
        else:
            os.system("mkdir ../../data/retina/data")

        if os.path.isdir("../../data/retina/spikes")==False:
            os.system("mkdir ../../data/retina/spikes")

        if os.path.isdir("../../data/retina/spikes/"+self.spike_folder):
            if len(os.listdir("../../data/retina/spikes/"+self.spike_folder)) > 0:
                os.system("rm -r ../../data/retina/spikes/"+self.spike_folder+"/*")
        else:
            os.system("mkdir ../../data/retina/spikes/"+self.spike_folder)

        if os.path.isdir("../../data/retina/mosaic")==False:
            os.system("mkdir ../../data/retina/mosaic")

    # Compute input value
    def getInput(self,t,resolution,dark_st):

        if(t*resolution >= self.pulse_tstart and
        t*resolution < self.pulse_tstart + self.pulse_duration):
            if dark_st==0:
                input = self.bkg_illuminance + self.pulse_amplitude
            else:
                input = self.bkg_illuminance - self.pulse_amplitude
        else:
            input = self.bkg_illuminance

        return [input,input,input]

    # Initialize arrays with NEST IDs
    def loadLayers(self,layer_IDs):

        self.layers_to_record = []
        self.s_layers_to_record = []
        self.top_PSTH_index = []
        self.top_layers_to_record = []

        for layer in self.labels:
            id_found = False
            for ll in layer_IDs:
                if (ll[0] == layer):
                    self.layers_to_record.append((ll[1],ll[2]))
                    id_found = True
            if id_found == False:
                # Assign random layer
                self.layers_to_record.append((ll[1],ll[2]))
                print "Warning: layer %s not found!" % layer

        for layer in self.sp_labels:
            id_found = False
            for ll in layer_IDs:
                if (ll[0] == layer):
                    self.s_layers_to_record.append((ll[1],ll[2]))
                    id_found = True
            if id_found == False:
                # Assign random layer
                self.s_layers_to_record.append((ll[1],ll[2]))
                print "Warning: layer %s not found!" % layer

        for layer in self.top_labels:
            # Search first for the matching spiking label
            index = 0
            for spiking_label in self.sp_labels:
                if(layer==spiking_label):
                    self.top_PSTH_index.append(index)
                index+=1

            # Then search for the NEST ID
            id_found = False
            for ll in layer_IDs:
                if (ll[0] == layer):
                    self.top_layers_to_record.append((ll[1],ll[2]))
                    id_found = True
            if id_found == False:
                # Assign random layer
                self.top_layers_to_record.append((ll[1],ll[2]))
                print "Warning: layer %s not found!" % layer

    # Compute mask
    def createRFmask(self):

        center_row = int(self.newSimulation.Params['N']/2.0)
        center_col = int(self.newSimulation.Params['N']/2.0)

        for cell in np.arange(self.newSimulation.Params['N']*self.newSimulation.Params['N']):
            row = int(cell/self.newSimulation.Params['N'])
            col = np.remainder(cell,self.newSimulation.Params['N'])

            if(row >= center_row - self.mask_side and row<= center_row + self.mask_side and
            col >= center_col - self.mask_side and col<= center_col + self.mask_side):
                self.RF_mask.append([row,col])

#        print "self.RF_mask = ",self.RF_mask

    # Create input stimulus and simulate photoreceptors' response
    def simulatePhotoreceptors(self,pos,dark_st):

        if(self.newSimulation.Params['load_cone_from_file'] == False):

            center_row = self.RF_mask[pos][0]
            center_col = self.RF_mask[pos][1]

            print "\n--- Computing input ---\n"

            L_input = []
            M_input = []
            S_input = []

            for t in np.arange(0,int(self.newSimulation.Params['simtime']/self.newSimulation.Params['resolution'])):
                [L,M,S] = self.getInput(t,self.newSimulation.Params['resolution'],dark_st)
                L_input.append(L)
                M_input.append(M)
                S_input.append(S)

            for cell in np.arange(self.newSimulation.Params['N']*self.newSimulation.Params['N']):
                stdout.write("\r cell: %d" % cell)
                stdout.flush()

                row = int(cell/self.newSimulation.Params['N'])
                col = np.remainder(cell,self.newSimulation.Params['N'])
                r = np.sqrt((row - center_row)**2 + (col - center_col)**2)

                # disk
                if(r <= self.radius*(self.newSimulation.Params['N']/self.newSimulation.Params['visSize'])):
                    self.L_cone_input[cell,:] = L_input
                    self.M_cone_input[cell,:] = M_input
                    self.S_cone_input[cell,:] = S_input
                else:
                    self.L_cone_input[cell,:] = self.bkg_illuminance
                    self.M_cone_input[cell,:] = self.bkg_illuminance
                    self.S_cone_input[cell,:] = self.bkg_illuminance

                # Video to show
                if self.show_video:
                    self.inputIm[row,col,:] = self.L_cone_input[cell,:]


            # Save input stimulus to file
            np.savetxt('../../data/retina/results/L_cone_input.out', self.L_cone_input)
            np.savetxt('../../data/retina/results/M_cone_input.out', self.M_cone_input)
            np.savetxt('../../data/retina/results/S_cone_input.out', self.S_cone_input)
            np.savetxt('../../data/retina/results/center_cell.out', (self.selected_cell,))

            if self.show_video:
                for t in np.arange(0,int(self.newSimulation.Params['simtime']/\
                self.newSimulation.Params['resolution'])):
                    np.savetxt('../../data/retina/results/inputIm_%s.out'%t, self.inputIm[:,:,t])

        # Load input stimulus from file
        else:
            self.L_cone_input = np.loadtxt('../../data/retina/results/L_cone_input.out')
            self.M_cone_input = np.loadtxt('../../data/retina/results/M_cone_input.out')
            self.S_cone_input = np.loadtxt('../../data/retina/results/S_cone_input.out')
            self.selected_cell = int(np.loadtxt('../../data/retina/results/center_cell.out'))
            if self.show_video:
                for t in np.arange(0,int(self.newSimulation.Params['simtime']/\
                self.newSimulation.Params['resolution'])):
                    self.inputIm[:,:,t] = np.loadtxt('../../data/retina/results/inputIm_%s.out'%t)

        cone_input = []
        cone_input.append(self.L_cone_input)
        cone_input.append(self.M_cone_input)
        cone_input.append(self.S_cone_input)

        # Simulate photoreceptors' response
        self.newSimulation.simulatePhotoreceptors(cone_input)


    # NEST simulation
    def NESTSimulation(self,load_mod=True):
        layer_IDs = self.newSimulation.NESTSimulation(self.mosaic,load_mod)
        self.loadLayers(layer_IDs)

        [self.potentials,self.spikes] = self.newSimulation.runSimulation(self.layers_to_record,
        self.s_layers_to_record)

        data_analysis.updatePSTH(self.simtime,self.spikes,self.s_layers_to_record,self.selected_cell,
        self.PSTHs,self.bin_size,0)

        if self.usetop_PSTHs:
            data_analysis.updatePSTH(self.simtime,self.spikes,self.s_layers_to_record,
            list(np.arange(self.newSimulation.Params['N']*self.newSimulation.Params['N'])),
            self.top_PSTHs,self.bin_size,0)

    # Plot results
    def plotResults(self):

        print "\n--- Plotting results ---\n"

        # Individual intracellular traces
        if self.plot_intracellular:

            fig = plt.figure()
            fig.subplots_adjust(hspace=1.5)
            fig.subplots_adjust(wspace=0.4)

            start_pos = int(self.start_time/self.newSimulation.Params['resolution'])
            time = self.newSimulation.time[start_pos:len(self.newSimulation.time)]

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (0,0))
            ax.plot(time,self.L_cone_input[self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('Input')

            # Save input stimulus
            np.savetxt('../../data/retina/input/L_input', self.L_cone_input[self.selected_cell,start_pos:len(self.newSimulation.time)])

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (0,1))
            ax.plot(time,
            self.newSimulation.cone_response[0,self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('L cones')

            # Save cone response
            np.savetxt('../../data/retina/input/L_cone', self.newSimulation.cone_response[0,self.selected_cell,start_pos:len(self.newSimulation.time)])

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (0,2))
            ax.plot(time,
            self.newSimulation.cone_response[1,self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('M cones')

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (0,3))
            ax.plot(time,
            self.newSimulation.cone_response[2,self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('S cones')

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (1,0))
            ax.plot(time,
            self.newSimulation.L_cone_metabotropic[self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('L cones metabotropic')

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (1,1))
            ax.plot(time,
            self.newSimulation.M_cone_metabotropic[self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('M cones metabotropic')

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (1,2))
            ax.plot(time,
            self.newSimulation.S_cone_metabotropic[self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('S cones metabotropic')

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (1,3))
            ax.plot(time,
            self.newSimulation.L_cone_ionotropic[self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('L cones ionotropic')

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (2,0))
            ax.plot(time,
            self.newSimulation.M_cone_ionotropic[self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('M cones ionotropic')

            ax = plt.subplot2grid((self.intracellular_rows,self.intracellular_cols), (2,1))
            ax.plot(time,
            self.newSimulation.S_cone_ionotropic[self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('S cones ionotropic')

            data_analysis.membranePotentials(self.start_time,self.newSimulation.Params['resolution'],
            self.simtime,self.potentials,self.layers_to_record,self.labels,self.selected_cell,self.intracellular_rows,self.intracellular_cols,self.intracellular_starting_row,self.intracellular_starting_col,"retina")

        # PSTHs
        if self.plot_PSTH:

            fig = plt.figure()
            fig.subplots_adjust(hspace=1.5)
            fig.subplots_adjust(wspace=0.4)

            start_pos = int(self.start_time/self.newSimulation.Params['resolution'])
            time = self.newSimulation.time[start_pos:len(self.newSimulation.time)]

            ax = plt.subplot2grid((self.PSTH_rows,self.PSTH_cols), (0,0))
            ax.plot(time,self.L_cone_input[self.selected_cell,start_pos:len(self.newSimulation.time)])
            ax.set_title('Input')

            data_analysis.PSTH(self.start_time,self.newSimulation.Params['resolution'],
            self.simtime,self.spikes,self.s_layers_to_record,self.sp_labels,self.selected_cell,
            self.PSTH_rows,self.PSTH_cols,self.PSTH_starting_row,self.PSTH_starting_col,
            self.trials,self.PSTHs,self.bin_size,"retina")

        # Topographical plot
        if self.plot_topographical:

            fig = plt.figure()
            fig.subplots_adjust(hspace=1.5)
            fig.subplots_adjust(wspace=0.4)

            data_analysis.topographical(fig,self.newSimulation.Params['N'],self.topographical_time_intervals,
            self.newSimulation.Params['resolution'],self.simtime,self.spikes,
            self.top_layers_to_record,self.top_labels,self.topographical_rows,self.topographical_cols,
            self.topographical_V_mins,self.topographical_V_maxs,self.topographical_isSpikes,
            self.trials,self.top_PSTHs,self.bin_size,self.top_PSTH_index,0)

        # Video sequence
#        data_analysis.videoSeq(self.newSimulation.Params['N'],self.inputIm,
#        self.simtime,self.newSimulation.Params['resolution'],self.intracellular_video_step)

        plt.show()

    # Save spike times
    def saveSpikes(self,stim,folder,trial):
        data_analysis.saveSpikes(self.newSimulation.Params['N'],self.spikes,
        self.s_layers_to_record,self.sp_labels,folder,"retina",stim,trial,0)


    def updateRFmaps(self,pos,dark,loaded_spikes = False):

        row = self.RF_mask[pos][0]
        col = self.RF_mask[pos][1]

        RF_index = 0

        for interval in self.RF_intervals:
            counter = 0

            for population, model in self.s_layers_to_record:

                if(loaded_spikes == False):
                    [data,selected_senders,pop] = data_analysis.getData(population,model,self.spikes,[self.selected_cell])
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
        data_analysis.receptiveField(fig,self.newSimulation.Params['N'],
        self.RF_intervals,self.s_layers_to_record,
        self.labels,self.RF_bright,self.RF_dark)
        plt.show()

    # Reset PSTHs for every different stimulus
    def resetPSTHs(self):
        self.PSTHs = np.zeros((len(self.sp_labels),int(self.simtime/self.bin_size)))

#! =================
#! Main
#! =================

if __name__ == '__main__':

    ex5 = experiment_5()
    ex5.initializeFolders()
    ex5.createRFmask()

    for pos in np.arange(len(ex5.RF_mask)):
        print "pos = %s" % pos

        # Bright stimulus
        ex5.simulatePhotoreceptors(pos,0)

        for trial in np.arange(ex5.trials):
            print "\n--- Trial %s ---\n" % trial
            ex5.NESTSimulation()
            ex5.saveSpikes(ex5.stim+"_bright_"+str(pos),ex5.spike_folder,trial)

        ex5.updateRFmaps(pos,0)
	ex5.plotResults()
        ex5.resetPSTHs()      

        # Dark stimulus
        ex5.simulatePhotoreceptors(pos,1)

        for trial in np.arange(ex5.trials):
            print "\n--- Trial %s ---\n" % trial
            ex5.NESTSimulation()
            ex5.saveSpikes(ex5.stim+"_dark_"+str(pos),ex5.spike_folder,trial)

        ex5.updateRFmaps(pos,1)
	ex5.plotResults()
        ex5.resetPSTHs()
        

    ex5.receptiveField()
