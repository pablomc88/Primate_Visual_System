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
# Name: ex3_grating_spatial_freq
#
# Description: retina response to sine-wave gratings of varying spatial frequency
# with the following properties:
#
# 1) All gratings used are of the same mean luminance
#
# 2) The red-green equiluminant gratings were produced by modulating the red
# and green in antiphase with equal amplitudes but opposite in sign. The same
# applies to the blue-yellow gratings.
#
# 3) Stimuli for the three cone-isolating directions (L-, M-, and S-cone), are
# produced by adjusting the modulation of one cone to the sine wave and the other
# two cones to the background level.
#
# References:
#
# [1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
# Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
# 2017
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import numpy as np
import scipy.fftpack
import sys,os,os.path
import matplotlib.pyplot as plt
from sys import stdout

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data_analysis
reload(data_analysis)

import run_network
reload(run_network)

class experiment_3(object):

    def __init__(self):

        ## Simulation parameters ##

        # Simulation time
        self.simtime = 1000.0

        # Number of trials
        self.trials = 2

        # Folder to save spike times
        self.spike_folder = 'Grating_Luminance'
#        self.spike_folder = 'Grating_Chromatic'
#        self.spike_folder = 'Grating_L_cone'
#        self.spike_folder = 'Grating_M_cone'
#        self.spike_folder = 'test'

        # Type of grating:
        # (0 = luminance grating, 1 = chromatic isoluminant grating (L vs M), 2 =
        # chromatic isoluminant grating (LM vs S), 3 = L cone-isolating, 4 =
        # M cone-isolating, 5 = S cone-isolating)
        self.grating_type = 0

        # Grating parameters
#        self.spatial_frequency = np.array([0.1,0.5,1.0,2.0,3.0,4.0,5.0,6.0]) # cpd
        self.spatial_frequency = np.array([0.1,0.5,1.0,3.0,4.0,5.0]) # cpd
        self.temporal_frequency = 2.0 # Hz
        self.bkg_illuminance = 250.0 # td
        # Michelson contrast:
        # Imax = bkg + ampl
        # Imin = bkg - ampl
        # contrast = (Imax - Imin)/(Imax + Imin) = ampl/bkg
        self.contrast = 0.6

        # Random distribution of cones
        self.generate_random_mosaic = False
        self.density_ratio = [0.6,0.3,0.1] # L-M-S cones
        self.load_mosaic = False

        # Start time of plots
        self.start_time = 200.0

        # Cell to analyze
        self.selected_cell = 0
        # Cell to analyze is the center cell in every 2D grid
        self.isCenterCell = True

        # PSTH bin size
        self.bin_size = 40.0 # ms

        # Layers to track (labels for figures)
        self.labels = [
        'H1_Horizontal_cells',
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

        # FFT labels
        self.FFT_labels = [
        'Midget_ganglion_cells_L_ON',
        'Midget_ganglion_cells_L_OFF',
        'Midget_ganglion_cells_M_ON',
        'Midget_ganglion_cells_M_OFF'
        ]

        ## Graphical parameters ##

        self.plot_intracellular = False
        self.plot_PSTH = False

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

        # FFT plot
        self.FFT_rows = 2
        self.FFT_cols = 2

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

        # FFT
        self.FFTamp = np.zeros((len(self.FFT_labels),len(self.spatial_frequency)))
        self.FFTph = np.zeros((len(self.FFT_labels),len(self.spatial_frequency)))
        self.FFT_recorded_models = []
        self.FFT_PSTH_index = []

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
    def getInput(self,stim_frequency,t,resolution,x,y,x0,y0):

        input = self.bkg_illuminance*(1.0 + self.contrast*\
        np.cos( ((x-x0)*stim_frequency*\
        (self.newSimulation.Params['visSize']/self.newSimulation.Params['N']) -\
        0.001*self.temporal_frequency*resolution*t)*2*np.pi ))

        input_out_of_phase = self.bkg_illuminance*(1.0 + self.contrast*\
        np.cos( ((x-x0)*stim_frequency*\
        (self.newSimulation.Params['visSize']/self.newSimulation.Params['N']) -\
        0.001*self.temporal_frequency*resolution*t)*2*np.pi + np.pi))

        if self.grating_type == 0:
            input_L = input
            input_M = input
            input_S = 0.0

        elif self.grating_type == 1:
            input_L = input
            input_M = input_out_of_phase
            input_S = 0.0

        elif self.grating_type == 2:
            input_L = input*0.5
            input_M = input*0.5
            input_S = input_out_of_phase

        elif self.grating_type == 3:
            input_L = input
            input_M = self.bkg_illuminance
            input_S = self.bkg_illuminance

        elif self.grating_type == 4:
            input_L = self.bkg_illuminance
            input_M = input
            input_S = self.bkg_illuminance

        else:
            input_L = self.bkg_illuminance*0.5
            input_M = self.bkg_illuminance*0.5
            input_S = input


        return [input_L,input_M,input_S]

    # Initialize arrays with NEST IDs
    def loadLayers(self,layer_IDs):

        self.layers_to_record = []
        self.s_layers_to_record = []
        self.FFT_PSTH_index = []
        self.FFT_recorded_models = []

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

        for layer in self.FFT_labels:
            # Search first for the matching spiking label
            index = 0
            for spiking_label in self.sp_labels:
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
                print "Warning: layer %s not found!" % layer


    # Randomize locations of cones
    def randomMosaic(self):

        self.mosaic = np.zeros(((3,self.newSimulation.Params['N'],self.newSimulation.Params['N'])))
        mosaic_im = np.zeros((self.newSimulation.Params['N'],self.newSimulation.Params['N']))

        # Generate a new mosaic
        if (self.load_mosaic == False):
            cell = 0
            draw_cell = [0,0]
            for x in np.arange(self.newSimulation.Params['N']):
                for y in np.arange(self.newSimulation.Params['N']):
                    if (cell == self.selected_cell):
                        draw_cell = [y,x]
                    cell+=1

                    # Draw samples from a uniform distribution
                    s = np.random.uniform(0,1,1)[0]
                    if(s<self.density_ratio[0]):
                        self.mosaic[0,x,y]=1
                        mosaic_im[x,y] = 0.0
                    elif(s>=self.density_ratio[0] and s<(self.density_ratio[0]+\
                    self.density_ratio[1])):
                        self.mosaic[1,x,y]=1
                        mosaic_im[x,y] = 1.0
                    else:
                        self.mosaic[2,x,y]=1
                        mosaic_im[x,y] = 0.075

            np.savetxt('../../data/retina/mosaic/L_cone_mosaic.out', self.mosaic[0,:,:])
            np.savetxt('../../data/retina/mosaic/M_cone_mosaic.out', self.mosaic[1,:,:])
            np.savetxt('../../data/retina/mosaic/S_cone_mosaic.out', self.mosaic[2,:,:])

        # Load mosaic from file
        else:
            self.mosaic[0,:,:] = np.loadtxt('../../data/retina/mosaic/L_cone_mosaic.out')
            self.mosaic[1,:,:] = np.loadtxt('../../data/retina/mosaic/M_cone_mosaic.out')
            self.mosaic[2,:,:] = np.loadtxt('../../data/retina/mosaic/S_cone_mosaic.out')

            cell = 0
            for x in np.arange(self.newSimulation.Params['N']):
                for y in np.arange(self.newSimulation.Params['N']):
                    if (cell == self.selected_cell):
                        draw_cell = [y,x]
                    cell+=1

                    if(self.mosaic[0,x,y]==1):
                        mosaic_im[x,y] = 0.0
                    elif(self.mosaic[1,x,y]==1):
                        mosaic_im[x,y] = 1.0
                    else:
                        mosaic_im[x,y] = 0.075

        # Plot
        fig = plt.figure()
        Vax = plt.subplot2grid((1,1), (0,0))
        Vax.matshow(mosaic_im,cmap='prism')
    #        Vax.set_title('Cone distribution')
        Vax.axes.get_xaxis().set_ticks([])
        Vax.axes.get_yaxis().set_ticks([])
    #        Vax.annotate('x', xy=(draw_center[0], draw_center[1]), xycoords='data',
    #        xytext=(draw_center[0], draw_center[1]))
        plt.show()

    # Compute FFT and get first harmonic
    def firstHarmonic(self,freq,loaded_spikes = False):
        counter = 0
        sp_type = 0
        for population, model in self.FFT_recorded_models:

            # Spikes
            if (model == 'retina_parvo_ganglion_cell'):
                if(loaded_spikes == False):
                    [data,selected_senders,pop] = data_analysis.getData(population,model,self.spikes,[self.selected_cell])
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
                    sp_type+=1
                    # Fourier sample spacing
                    T = self.bin_size*0.001 # s

                response = PSTH_array

            # Membrane potentials
            else:
                [data,selected_senders,pop] = data_analysis.getData(population,model,self.potentials,[self.selected_cell])

                V_m = (data[0]['V_m'])[selected_senders[0]]
                V_m = V_m[int(self.start_time/self.newSimulation.Params['resolution']):len(V_m)]
                response = V_m

                # Fourier sample spacing
                T = self.newSimulation.Params['resolution']*0.001 # s

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

#            print "model = ",model," , freq = ",self.temporal_frequency," , ampl selected = ",np.max(ampl_possible_choices)
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
        self.PSTHs = np.zeros((len(self.sp_labels),int(self.simtime/self.bin_size)))

    # Create input stimulus and simulate photoreceptors' response
    def simulatePhotoreceptors(self,stim):

        print "\n--- Freq: %s cpd ---\n" % str(stim)

        if(self.newSimulation.Params['load_cone_from_file'] == False):

            center_row = int(self.newSimulation.Params['N']/2.0)
            center_col = int(self.newSimulation.Params['N']/2.0)

            print "\n--- Computing input ---"

            last_cell = 0

            for cell in np.arange(self.newSimulation.Params['N']*self.newSimulation.Params['N']):
                progress = 100*cell/len(self.L_cone_input)
                stdout.write("\r progress: %d %%"% progress)
                stdout.flush()

                for t in np.arange(0,int(self.newSimulation.Params['simtime']/self.newSimulation.Params['resolution'])):

                    row = int(cell/self.newSimulation.Params['N'])
                    col = np.remainder(cell,self.newSimulation.Params['N'])

                    # The array computed in the first col is re-used for the rest
                    # of cols
                    if col == 0.0:
                        [L_input,M_input,S_input] = self.getInput(stim,t,
                        self.newSimulation.Params['resolution'],row,col,center_row,center_col)
                        last_cell = cell
                    else:
                        L_input = self.L_cone_input[last_cell,t]
                        M_input = self.M_cone_input[last_cell,t]
                        S_input = self.S_cone_input[last_cell,t]

                    self.L_cone_input[cell,t] = L_input
                    self.M_cone_input[cell,t] = M_input
                    self.S_cone_input[cell,t] = S_input

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

    # Plot results
    def plotIntermediateResults(self):

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

        plt.show()


    def saveResults(self):

        for n in np.arange(len(self.FFT_labels)):
            np.savetxt('../../data/retina/data/ampl_'+self.FFT_labels[n], self.FFTamp[n])
            np.savetxt('../../data/retina/data/phase_'+self.FFT_labels[n], self.FFTph[n])
            if n==0:
                np.savetxt('../../data/retina/data/freq', self.spatial_frequency)


    def plotFinalResults(self):

        print "\n--- Plotting results ---\n"

        # Plot
        fig = plt.figure()

        data_analysis.spatialTuning(self.spatial_frequency,self.FFTamp,self.FFTph,
        self.FFT_labels,self.FFT_rows,self.FFT_cols,0,0,'Spatial frequency (cpd)',
       "Ampl. (mV or s^(-1))")

        plt.show()

    # Save spike times
    def saveSpikes(self,stim,folder,trial):
        data_analysis.saveSpikes(self.newSimulation.Params['N'],self.spikes,
        self.s_layers_to_record,self.sp_labels,folder,"retina",stim,trial,0)


#! =================
#! Main
#! =================

if __name__ == '__main__':

    ex3 = experiment_3()
    ex3.initializeFolders()

    for f in ex3.spatial_frequency:
        ex3.simulatePhotoreceptors(f)

        if(ex3.generate_random_mosaic):
            ex3.randomMosaic()
            ex3.generate_random_mosaic = False

        for trial in np.arange(ex3.trials):
            print "\n--- Trial %s ---\n" % trial
            ex3.NESTSimulation()
            ex3.saveSpikes('_sf_'+str(f),ex3.spike_folder,trial)

        ex3.firstHarmonic(f)
        ex3.plotIntermediateResults()
        ex3.resetPSTHs()

    # Plot freq. curves
    ex3.saveResults()
    ex3.plotFinalResults()
