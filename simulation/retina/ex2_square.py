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
# Name: ex2_square
#
# Description: retina response to  chromatic or achromatic square surfaces
# displayed on a gray background. The following aspects have been taken into
# account:
#
# 1) Chromatic squares are equal in luminance to the background
#
# 2) The luminance contrast of the achromatic squares was adjusted to be similar
# to the L-M cone contrast of the chromatic squares
#
# 3) Cone contrast is computed from the amount of cone excitation (E) in the square
# relative to the cone excitation in the gray background and expressed as [3]:
# C_cone = (E_square - E_bkg)/ E_bkg
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
# [3] Stromeyer, C. F., G. R. Cole, and R. E. Kronauer. "Second-site adaptation
# in the red-green chromatic pathways." Vision Research 25.2 (1985): 219-237.
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import numpy as np
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

        # Number of trials
        self.trials = 2

        # Folder to save spike times
        self.spike_folder = 'flashing_square'

        # ID of the stimulus type
        self.stim = '_square_'

        # Type of square:
        # 0 = red square, 1 = green square, 2 = black square, 3 = white square
        self.square_type = 2

        # Start time of plots
        self.start_time = 200.0

        # Center of the square (in number of cells)
        self.square_center = [19.5,19.5]

        # Side length (in number of cells)
        self.side_length = 20.0

        # Pulse parameters
        self.pulse_duration = 250.0 # ms
        self.pulse_tstart = 500.0 # ms (first 300 ms are used to fill the input and
                                # output buffers of linear filters)
        self.bkg_illuminance = 250.0 # td
        self.pulse_contrast = 0.8
        self.pulse_amplitude = self.pulse_contrast * self.bkg_illuminance # td

        # Cell to analyze
        self.selected_cell = 0
        # Cell to analyze is the center cell in every 2D grid
        self.isCenterCell = True

        # PSTH bin size
        self.bin_size = 10.0 # ms

        # Layers to track (labels for figures)
        self.labels = ['H1_Horizontal_cells',
        'H2_Horizontal_cells',
        'Midget_bipolar_cells_L_ON',
        'Midget_bipolar_cells_L_OFF',
        'Midget_bipolar_cells_M_ON',
        'Midget_bipolar_cells_M_OFF',
        'Diffuse_bipolar_cells_S_ON',
        'S_cone_bipolar_cells_S_ON',
        'AII_amacrine_cells',
        'Midget_ganglion_cells_L_ON',
        'Midget_ganglion_cells_L_OFF',
        'Midget_ganglion_cells_M_ON',
        'Midget_ganglion_cells_M_OFF',
        'Small_bistratified_ganglion_cells_S_ON'
        ]

        # Spiking layers
        self.sp_labels = [
        'Midget_ganglion_cells_L_ON',
        'Midget_ganglion_cells_L_OFF',
        'Midget_ganglion_cells_M_ON',
        'Midget_ganglion_cells_M_OFF',
        'Small_bistratified_ganglion_cells_S_ON'
        ]

        # Parameters of the topographical plot
        self.top_labels = [
        'Midget_ganglion_cells_L_ON',
        'Midget_ganglion_cells_L_OFF'
        ]

        ## Graphical parameters ##

        self.plot_intracellular = False
        self.plot_PSTH = False
        self.plot_topographical = True

        # Individual intracellular traces
        self.intracellular_rows = 6
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
        self.topographical_cols = 5
        self.topographical_time_intervals = [450.0,500.0,550.0,600.0,750.0,800.0]
        self.topographical_V_mins = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.topographical_V_maxs = [200.0, 200.0, 200.0, 200.0, 200.0]
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
    def getInput(self,t,resolution):

        if(t*resolution >= self.pulse_tstart and
        t*resolution < self.pulse_tstart + self.pulse_duration):
            if (self.square_type == 0):
                L_input = self.pulse_amplitude/2.0 + self.bkg_illuminance
                M_input = self.bkg_illuminance - self.pulse_amplitude/2.0
            elif (self.square_type == 1):
                L_input = self.bkg_illuminance - self.pulse_amplitude/2.0
                M_input = self.pulse_amplitude/2.0 + self.bkg_illuminance
            elif (self.square_type == 2):
                L_input = self.bkg_illuminance - self.pulse_amplitude
                M_input = self.bkg_illuminance - self.pulse_amplitude
            elif (self.square_type == 3):
                L_input = self.pulse_amplitude + self.bkg_illuminance
                M_input = self.pulse_amplitude + self.bkg_illuminance
        else:
            L_input = self.bkg_illuminance
            M_input = self.bkg_illuminance

        return [L_input,M_input,0.0]

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
                print ("Warning: layer %s not found!" % layer)

        for layer in self.sp_labels:
            id_found = False
            for ll in layer_IDs:
                if (ll[0] == layer):
                    self.s_layers_to_record.append((ll[1],ll[2]))
                    id_found = True
            if id_found == False:
                # Assign random layer
                self.s_layers_to_record.append((ll[1],ll[2]))
                print ("Warning: layer %s not found!" % layer)

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
                print ("Warning: layer %s not found!" % layer)

    # Create input stimulus and simulate photoreceptors' response
    def simulatePhotoreceptors(self):

        if(self.newSimulation.Params['load_cone_from_file'] == False):

            center_row = int(self.newSimulation.Params['N']/2.0)
            center_col = int(self.newSimulation.Params['N']/2.0)

            print ("\n--- Computing input ---\n")

            L_input = []
            M_input = []
            S_input = []

            for t in np.arange(0,int(self.newSimulation.Params['simtime']/self.newSimulation.Params['resolution'])):
                [L,M,S] = self.getInput(t,self.newSimulation.Params['resolution'])
                L_input.append(L)
                M_input.append(M)
                S_input.append(S)

            for cell in np.arange(self.newSimulation.Params['N']*self.newSimulation.Params['N']):
                stdout.write("\r cell: %d" % cell)
                stdout.flush()

                row = int(cell/self.newSimulation.Params['N'])
                col = np.remainder(cell,self.newSimulation.Params['N'])

                # square
                if(row >= (self.square_center[0] - self.side_length/2.0) and
                row <= (self.square_center[0] + self.side_length/2.0) and
                col >= (self.square_center[1] - self.side_length/2.0) and
                col <= (self.square_center[1] + self.side_length/2.0)):
                    self.L_cone_input[cell,:] = L_input
                    self.M_cone_input[cell,:] = M_input
                    self.S_cone_input[cell,:] = S_input
                else:
                    self.L_cone_input[cell,:] = self.bkg_illuminance
                    self.M_cone_input[cell,:] = self.bkg_illuminance
                    self.S_cone_input[cell,:] = self.bkg_illuminance

                # center cell
                if row == center_row and col == center_col and self.isCenterCell:
                    self.selected_cell = cell

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

        print ("\n--- Plotting results ---\n")

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

            if self.topographical_isSpikes:
                recs = self.spikes
            else:
                recs = self.potentials

            data_analysis.topographical(fig,self.newSimulation.Params['N'],self.topographical_time_intervals,
            self.newSimulation.Params['resolution'],self.simtime,recs,
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


#! =================
#! Main
#! =================

if __name__ == '__main__':

    ex2 = experiment_2()
    ex2.initializeFolders()

    ex2.simulatePhotoreceptors()
    for trial in np.arange(ex2.trials):
        print ("\n--- Trial %s ---\n" % trial)
        ex2.NESTSimulation()
        ex2.saveSpikes(ex2.stim,ex2.spike_folder,trial)
    ex2.plotResults()
