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
# Name: mpi_run
#
# Description: run the different trials of the selected experiment in parallel
# taking advantage of MPI. Each MPI process is assigned an equal number of trials
# to be computed. Spike times are saved to file.
#
# References:
#
# [1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
# Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
# 2017
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import sys,os,os.path
import numpy as np
import time

import nest

# Install modules before importing mpi4py (to prevent errors with mpi4py)
nest.Install("parvo_neuron_module")
nest.Install("AII_amacrine_module")
nest.Install("ganglion_cell_module")

from mpi4py import MPI

# Initialize the MPI environment
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data_analysis
reload(data_analysis)

import run_network
reload(run_network)

import ex1_disk
reload(ex1_disk)

import ex2_square
reload(ex2_square)

import ex3_grating_spatial_freq
reload(ex3_grating_spatial_freq)

import ex4_disk_area_response
reload(ex4_disk_area_response)

import ex5_receptive_field
reload(ex5_receptive_field)

# Experiment to simulate
select_ex = 1

# Function to be executed by every MPI process
def worker(stim,select_ex,ex_value,aux_stimulus_value=0):
    print "\n--- Rank: %s, Trials: %s ---\n" % (rank,stim)

    # Initialize mosaic of the grating only once
    if select_ex == 3:
        ex = ex3_grating_spatial_freq.experiment_3()

        if(ex.generate_random_mosaic):
            ex.randomMosaic()
            ex.generate_random_mosaic = False

    # Create RF mask for experiment 5
    if select_ex == 5:
        ex = ex5_receptive_field.experiment_5()
        ex.createRFmask()

    for tr in stim:

        # Flashing spot
        if select_ex == 1:
            ex = ex1_disk.experiment_1()
            # Avoid simulation of photoreceptors
            ex.newSimulation.Params['load_cone_from_file'] = True
            ex.simulatePhotoreceptors()
            ex.NESTSimulation(False)
            ex.saveSpikes(ex.stim,ex.spike_folder,tr)

        # Square
        elif select_ex == 2:
            ex = ex2_square.experiment_2()
            ex.newSimulation.Params['load_cone_from_file'] = True
            ex.simulatePhotoreceptors()
            ex.NESTSimulation(False)
            ex.saveSpikes(ex.stim,ex.spike_folder,tr)

        # Grating
        elif select_ex == 3:
            ex.newSimulation.Params['load_cone_from_file'] = True
            ex.simulatePhotoreceptors(ex_value)
            ex.NESTSimulation(False)
            ex.saveSpikes('_sf_'+str(ex_value),ex.spike_folder,tr)

        # Area-response curves
        elif select_ex == 4:
            ex = ex4_disk_area_response.experiment_4()
            ex.newSimulation.Params['load_cone_from_file'] = True
            ex.simulatePhotoreceptors(ex_value)
            ex.NESTSimulation(False)
            ex.saveSpikes(ex.stim+str(ex_value),ex.spike_folder,tr)

        # RF
        elif select_ex == 5:
            ex.newSimulation.Params['load_cone_from_file'] = True
            ex.simulatePhotoreceptors(ex_value,aux_stimulus_value)
            ex.NESTSimulation(False)

            if aux_stimulus_value == 0:
                ex.saveSpikes(ex.stim+"_bright_"+str(ex_value),ex.spike_folder,tr)
            else:
                ex.saveSpikes(ex.stim+"_dark_"+str(ex_value),ex.spike_folder,tr)

    return 1


#! =================
#! Main
#! =================

def main():

    aux_stimulus_value = [0]

    # Start timer
    if rank == 0:
        start_c = time.time()

    # Initialize experiment object
    # ex_stimulus: range of stimulus values
    # aux_stimulus_value: only used by ex. 5 to differentiate bright (0) from
    # dark (1) stimuli
    if select_ex ==3:
        ex = ex3_grating_spatial_freq.experiment_3()
        ex_stimulus = ex.spatial_frequency
        aux_stimulus_value = np.zeros(len(ex_stimulus))
        if rank == 0:
            ex.initializeFolders()

    elif select_ex ==4:
        ex = ex4_disk_area_response.experiment_4()
        ex_stimulus = ex.disk_diameters
        aux_stimulus_value = np.zeros(len(ex_stimulus))
        if rank == 0:
            ex.initializeFolders()

    elif select_ex ==5:
        ex = ex5_receptive_field.experiment_5()
        ex.createRFmask()

        ex_stimulus = []
        aux_stimulus_value = []

        for nd in np.arange(len(ex.RF_mask)):
            ex_stimulus.append(int(nd))
            ex_stimulus.append(int(nd))
            aux_stimulus_value.append(0)
            aux_stimulus_value.append(1)

        if rank == 0:
            ex.initializeFolders()

    else:
        ex_stimulus = [0]

    c = 0
    for ex_value in ex_stimulus:
        # Create an instance of the experiment and simulate photoreceptors (only
        # once by process with rank = 0)
        if rank==0:
            print "\n--- Experiment value: %s ---\n" % ex_value
            if select_ex ==1:
                ex = ex1_disk.experiment_1()
                ex.initializeFolders()
                ex.simulatePhotoreceptors()

            elif select_ex ==2:
                ex = ex2_square.experiment_2()
                ex.initializeFolders()
                ex.simulatePhotoreceptors()

            elif select_ex ==3 or select_ex ==4:
                ex.simulatePhotoreceptors(ex_value)

            elif select_ex ==5:
                ex.simulatePhotoreceptors(ex_value,aux_stimulus_value[c])

        # Divide data into chunks
        if rank == 0:
            chunks = [[] for _ in range(size)]
            stimulus = np.arange(ex.trials)
            for i, chunk in enumerate(stimulus):
                chunks[(i) % size].append(chunk)
        else:
            chunks = None
            chunks_exp = None

        # Scatter data
        stim = []
        stim = comm.scatter(chunks,root=0)
        value_to_return = worker(stim,select_ex,ex_value,aux_stimulus_value[c])
        c+=1
        # Gather data
        results = comm.gather(value_to_return, root=0)

    # End of simulation
    if rank == 0:
        end_c = time.time()
        print "time elapsed (h): ",(end_c - start_c)/3600.0


if __name__ == '__main__':
    main()
