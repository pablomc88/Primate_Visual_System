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
# Name: mpi_run
#
# Description: run the different trials of the selected experiment in parallel
# taking advantage of MPI. Each MPI process is assigned an equal number of trials
# to be computed. Spike times are saved to file.
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

import sys,os,os.path
import numpy as np
import time

import nest
from mpi4py import MPI

# Initialize the MPI environment
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data_analysis
import run_network
import ex1_flash
import ex2_grating_spatial_freq
import ex3_disk_area_response
import ex4_receptive_field

# Experiment to simulate
select_ex = 1

# Function to be executed by every MPI process
def worker(stim,select_ex,ex_value,aux_stimulus_value=0):
    print ("\n--- Rank: %s, Trials: %s ---\n" % (rank,stim))

    # Create RF mask for experiment 4
    if select_ex == 4:
        ex = ex4_receptive_field.experiment_4()
        ex.createRFmask()

    for tr in stim:

        # Flashing spot
        if select_ex == 1:
            ex = ex1_flash.experiment_1()
            ex.NESTSimulation(tr)
            ex.saveSpikes(ex.stim,ex.spike_folder,tr)

        # Grating
        elif select_ex == 2:
            ex = ex2_grating_spatial_freq.experiment_2()
            ex.NESTSimulation(tr,'_sf_'+str(ex_value))
            ex.saveSpikes('_sf_'+str(ex_value),ex.spike_folder,tr)

        # Area-response curves
        elif select_ex == 3:
            ex = ex3_disk_area_response.experiment_3()
            ex.NESTSimulation(tr,'_disk_'+str(ex_value))
            ex.saveSpikes('_disk_'+str(ex_value),ex.spike_folder,tr)

        # RF
        elif select_ex == 4:

            if aux_stimulus_value == 0:
                ex.NESTSimulation(tr,"_disk_"+"_bright_"+str(ex_value))
                ex.saveSpikes("_disk_"+"_bright_"+str(ex_value),ex.spike_folder,tr)
            else:
                ex.NESTSimulation(tr,"_disk_"+"_dark_"+str(ex_value))
                ex.saveSpikes("_disk_"+"_dark_"+str(ex_value),ex.spike_folder,tr)


    return 1


#! =================
#! Main
#! =================

def main():

    aux_stimulus_value = [0]

    # Start timer
    if rank == 0:
        start_c = time.time()

    # Create an instance of the experiment.
    # ex_stimulus: range of stimulus values
    # aux_stimulus_value: only used by ex. 4 to differentiate bright (0) from
    # dark (1) stimuli
    if select_ex ==1:
        ex = ex1_flash.experiment_1()
        ex_stimulus = [0]
        aux_stimulus_value = [0]
        ex.initializeFolders()

    if select_ex ==2:
        ex = ex2_grating_spatial_freq.experiment_2()
        ex_stimulus = ex.spatial_frequency
        aux_stimulus_value = np.zeros(len(ex_stimulus))
        if rank == 0:
            ex.initializeFolders()

    elif select_ex ==3:
        ex = ex3_disk_area_response.experiment_3()
        ex_stimulus = ex.disk_diameters
        aux_stimulus_value = np.zeros(len(ex_stimulus))
        if rank == 0:
            ex.initializeFolders()

    elif select_ex ==4:
        ex = ex4_receptive_field.experiment_4()
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


    c = 0

    for ex_value in ex_stimulus:
        if rank==0:
            print ("\n--- Experiment value: %s ---\n" % ex_value)

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
        print ("time elapsed (h): ",(end_c - start_c)/3600.0)


if __name__ == '__main__':
    main()
