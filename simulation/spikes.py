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
# Name: spikes
#
# Description: script that loads files of spike times generated after simulation and
# plots results based on graphical parameters indicated in the corresponding
# simulation script.
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
import sys,os,os.path
import matplotlib.pyplot as plt

import data_analysis
reload(data_analysis)

## Parameters ##

# Experiments:
# 1: retina response to light flashes, which can be disk- or ring-shaped
# 2: retina response to flashing squares
# 3: retina response to sine-wave gratings of varying spatial frequency
# 4: retina response to flashing spots of varying diameter
# 5: estimation of the retina receptive fields
# 6: thalamocortical response to light flashes
# 7: thalamocortical response to sine-wave gratings of varying spatial frequency
# 8: thalamocortical response to flashing spots of varying diameter
# 9: estimation of the thalamocortical receptive fields
select_ex = 1

# Graphical parameters
plot_PSTHs = True
plot_topPSTHs = True
plot_spatial_tuning = False
plot_area_response_curves = False
plot_RF = False

## End of parameters ##

# Reset entries in sys.path
for s in sys.path:
    if 'retina' in s or 'thalamocortical_system' in s:
        sys.path.remove(s)

if select_ex < 6:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'retina'))

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

else:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'thalamocortical_system'))

    import ex1_flash
    reload(ex1_flash)

    import ex2_grating_spatial_freq
    reload(ex2_grating_spatial_freq)

    import ex3_disk_area_response
    reload(ex3_disk_area_response)

    import ex4_receptive_field
    reload(ex4_receptive_field)


def plotPSTHs(ex,N,stim,layer_sizes,path):

    fig = plt.figure()
    fig.subplots_adjust(hspace=1.5)
    fig.subplots_adjust(wspace=0.4)

    if(isinstance(layer_sizes, list) == False):
        data_analysis.initializePSTHs(ex,"retina",False,N,
        ex.sp_labels,stim,0,path)

        data_analysis.PSTH(ex.start_time,ex.newSimulation.Params['resolution'],
        ex.simtime,ex.spikes,ex.s_layers_to_record,ex.sp_labels,ex.selected_cell,
        ex.PSTH_rows,ex.PSTH_cols,ex.PSTH_starting_row,ex.PSTH_starting_col,
        ex.trials,ex.PSTHs,ex.bin_size,"retina","../data/")
    else:
        data_analysis.initializePSTHs(ex,"thalamocortical_system",False,N,
        ex.labels,stim,ex.layer_sizes,path)

        data_analysis.PSTH(ex.start_time,ex.newSimulation.Params['resolution'],
        ex.simtime,ex.spikes,ex.layers_to_record,ex.labels,ex.selected_cell,
        ex.PSTH_rows,ex.PSTH_cols,ex.PSTH_starting_row,ex.PSTH_starting_col,
        ex.trials,ex.PSTHs,ex.bin_size,"thalamocortical_system","../data/")

def plotTopographical(ex,N,stim,layer_sizes,path):

    fig = plt.figure()
    fig.subplots_adjust(hspace=1.5)
    fig.subplots_adjust(wspace=0.4)

    if(isinstance(layer_sizes, list) == False):
        data_analysis.initializePSTHs(ex,"retina",True,N,
        ex.sp_labels,stim,0,path)

        data_analysis.topographical(fig,ex.newSimulation.Params['N'],ex.topographical_time_intervals,
        ex.newSimulation.Params['resolution'],ex.simtime,ex.potentials,
        ex.top_layers_to_record,ex.top_labels,ex.topographical_rows,ex.topographical_cols,
        ex.topographical_V_mins,ex.topographical_V_maxs,ex.topographical_isSpikes,
        ex.trials,ex.top_PSTHs,ex.bin_size,ex.top_PSTH_index,0,False,'retina',"../data/")

    else:
        data_analysis.initializePSTHs(ex,"thalamocortical_system",True,N,
        ex.labels,stim,ex.layer_sizes,path)

        data_analysis.topographical(fig,ex.newSimulation.Params['N_cortex'],ex.topographical_time_intervals,
        ex.newSimulation.Params['resolution'],ex.simtime,ex.spikes,
        ex.top_layers_to_record,ex.top_labels,ex.topographical_rows,ex.topographical_cols,
        ex.topographical_V_mins,ex.topographical_V_maxs,ex.topographical_isSpikes,
        ex.trials,ex.top_PSTHs,ex.bin_size,ex.top_PSTH_index,ex.layer_sizes_top,False,
        'thalamocortical_system',"../data/")

        data_analysis.topographical(fig,ex.newSimulation.Params['N_cortex'],ex.topographical_time_intervals,
        ex.newSimulation.Params['resolution'],ex.simtime,ex.spikes,
        ex.pop_layers_to_record,ex.pop_labels,ex.topographical_rows,ex.topographical_cols,
        ex.pop_V_mins,ex.pop_V_maxs,ex.topographical_isSpikes,
        ex.trials,ex.top_PSTHs,ex.bin_size,ex.pop_PSTH_index,ex.layer_sizes_pop,True,
        'thalamocortical_system',"../data/")


def plotSpatialTuning(ex,N,stim,layer_sizes,path):

    # Compute harmonic responses
    for f in ex.spatial_frequency:
        if(isinstance(layer_sizes, list) == False):
            data_analysis.initializePSTHs(ex,"retina",False,N,
            ex.FFT_labels,'_sf_'+str(f),0,path)
        else:
            data_analysis.initializePSTHs(ex,"thalamocortical_system",False,N,
            ex.labels,'_sf_'+str(f),ex.layer_sizes,path)

        ex.firstHarmonic(f,True)
        ex.resetPSTHs()

    fig = plt.figure()

    data_analysis.spatialTuning(ex.spatial_frequency,ex.FFTamp,ex.FFTph,
    ex.FFT_labels,ex.FFT_rows,ex.FFT_cols,0,0,'Spatial frequency (cpd)',
   "Ampl. (mV or s^(-1))")


def plotAreaResponse(ex,N,stim,layer_sizes,path):

    # Compute responses to flashing spot
    for d in ex.disk_diameters:
        if(isinstance(layer_sizes, list) == False):
            data_analysis.initializePSTHs(ex,"retina",False,N,
            ex.area_labels,ex.stim+str(d),0,path)
        else:
            data_analysis.initializePSTHs(ex,"thalamocortical_system",False,N,
            ex.labels,'_disk_'+str(d),ex.layer_sizes,path)

        ex.areaResponseCurve(d,True)
        ex.resetPSTHs()

    fig = plt.figure()

    data_analysis.spatialTuning(ex.disk_diameters,ex.area_amp,ex.area_ph,
    ex.area_labels,ex.area_rows,ex.area_cols,0,0,'Disk diameter (deg)',
    "Ampl. (mV or s^(-1))")

def receptiveField(ex,N,stim,layer_sizes,path):

    ex.createRFmask()

    for pos in np.arange(len(ex.RF_mask)):
        if(isinstance(layer_sizes, list) == False):
            data_analysis.initializePSTHs(ex,"retina",False,N,
            ex.sp_labels,ex.stim+"_bright_"+str(pos),0,path)
        else:
            data_analysis.initializePSTHs(ex,"thalamocortical_system",False,N,
            ex.labels,"_disk_"+"_bright_"+str(pos),ex.layer_sizes,path)

        ex.updateRFmaps(pos,0,True)
        ex.resetPSTHs()

        if(isinstance(layer_sizes, list) == False):
            data_analysis.initializePSTHs(ex,"retina",False,N,
            ex.sp_labels,ex.stim+"_dark_"+str(pos),0,path)
        else:
            data_analysis.initializePSTHs(ex,"thalamocortical_system",False,N,
            ex.labels,"_disk_"+"_dark_"+str(pos),ex.layer_sizes,path)

        ex.updateRFmaps(pos,1,True)
        ex.resetPSTHs()

    fig = plt.figure()

    if(isinstance(layer_sizes, list) == False):
        data_analysis.receptiveField(fig,ex.newSimulation.Params['N'],
        ex.RF_intervals,ex.s_layers_to_record,
        ex.labels,ex.RF_bright,ex.RF_dark)
    else:
        data_analysis.receptiveField(fig,ex.newSimulation.Params['N_LGN'],
        ex.RF_intervals,ex.layers_to_record,
        ex.labels,ex.RF_bright,ex.RF_dark)

#! =================
#! Main
#! =================

if __name__ == '__main__':

    if select_ex == 1:
        ex = ex1_disk.experiment_1()
        stim = ex.stim
    elif select_ex == 2:
        ex = ex2_square.experiment_2()
        stim = ex.stim
    elif select_ex == 3:
        ex = ex3_grating_spatial_freq.experiment_3()
        stim = '_sf_'+str(ex.spatial_frequency[0])
    elif select_ex == 4:
        ex = ex4_disk_area_response.experiment_4()
        stim = '_disk_'+str(ex.disk_diameters[0])
    elif select_ex == 5:
        ex = ex5_receptive_field.experiment_5()
        stim = "_disk_"+"_bright_"+str(0)

    elif select_ex == 6:
        ex = ex1_flash.experiment_1()
        stim = ex.stim
    elif select_ex == 7:
        ex = ex2_grating_spatial_freq.experiment_2()
        stim = '_sf_'+str(ex.spatial_frequency[0])
    elif select_ex == 8:
        ex = ex3_disk_area_response.experiment_3()
        stim = '_disk_'+str(ex.disk_diameters[0])
    elif select_ex == 9:
        ex = ex4_receptive_field.experiment_4()
        stim = "_disk_"+"_bright_"+str(0)


    # Retina
    if select_ex < 6:

        # Pick center cell
        for cell in np.arange(ex.newSimulation.Params['N']**2):
            row = int(cell/ex.newSimulation.Params['N'])
            col = np.remainder(cell,ex.newSimulation.Params['N'])
            if row == int(ex.newSimulation.Params['N']/2.0) and\
            col == int(ex.newSimulation.Params['N']/2.0):
                ex.selected_cell = cell

        # To load layer IDs from NEST
        layer_IDs = ex.newSimulation.NESTSimulation(ex.mosaic,True)
        ex.loadLayers(layer_IDs)

        if plot_PSTHs:
            plotPSTHs(ex,ex.newSimulation.Params['N'],stim,0,"../data/")

        if plot_topPSTHs:
            plotTopographical(ex,ex.newSimulation.Params['N'],stim,0,"../data/")

        if plot_spatial_tuning:
            plotSpatialTuning(ex,ex.newSimulation.Params['N'],stim,0,"../data/")

        if plot_area_response_curves:
            plotAreaResponse(ex,ex.newSimulation.Params['N'],stim,0,"../data/")

        if plot_RF:
            receptiveField(ex,ex.newSimulation.Params['N'],stim,0,"../data/")

    # Thalamocortical system
    else:

        # To load layer IDs from NEST
        layer_IDs = ex.newSimulation.NESTSimulation(
        data_analysis.loadSpikes(ex.newSimulation.Params['N_LGN'],
        ex.retina_labels,ex.spike_folder,"retina",stim,0,0,"../data/"))
        ex.loadLayers(layer_IDs)

        if plot_PSTHs:
            plotPSTHs(ex,ex.newSimulation.Params['N_LGN'],stim,ex.layer_sizes,"../data/")

        if plot_topPSTHs:
            plotTopographical(ex,ex.newSimulation.Params['N_LGN'],stim,ex.layer_sizes,"../data/")

        if plot_spatial_tuning:
            plotSpatialTuning(ex,ex.newSimulation.Params['N_LGN'],stim,ex.layer_sizes,"../data/")

        if plot_area_response_curves:
            plotAreaResponse(ex,ex.newSimulation.Params['N_LGN'],stim,ex.layer_sizes,"../data/")

        if plot_RF:
            receptiveField(ex,ex.newSimulation.Params['N_LGN'],stim,ex.layer_sizes,"../data/")


    plt.show()
