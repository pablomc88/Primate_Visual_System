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
# Name: data_analysis
#
# Description: Library of functions to analyze and plot the results of simulation.
#
# References:
#
# [1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
# Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
# 2017
#
# [2] Nawrot, Martin, Ad Aertsen, and Stefan Rotter. "Single-trial estimation
# of neuronal firing rates: from single-neuron spike trains to population
# activity." Journal of neuroscience methods 94.1 (1999): 81-92.
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import nest
import numpy as np
import sys,os,os.path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

# Get events and senders from recorders
def getData(population,model,recorders,selected_cells):

    l = [nd for nd in np.arange(0,len(recorders)) if
    (recorders[nd][1] == population and recorders[nd][2] == model)][0]

    data = nest.GetStatus(recorders[l][0],keys='events')
    pop = [nd for nd in nest.GetLeaves(population)[0] if
    nest.GetStatus([nd], 'model')[0]==model]
    senders = data[0]['senders']

    if len(selected_cells) == 1:
        selected_senders = np.where(senders==pop[selected_cells[0]])
    else:
        selected_senders = senders

    return data,selected_senders,pop

# Update the spike-counts in the Post-Stimulus Time Histogram (PSTH) of every
# population during a trial
def updatePSTH(sim_time,recorders,recorded_models,cell,PSTHs,bin_size,
layer_sizes):

    j = 0

    for population, model in recorded_models:

        # There is only one PSTH (one cell) in every population
        if(PSTHs.ndim == 2):

            # The selected cell is the same in all layers
            if(isinstance(layer_sizes, list) == False):
                [data,selected_senders,pop] = getData(population,model,recorders,[cell])
            # The selected cell is different in every layer
            else:
                [data,selected_senders,pop] = getData(population,model,recorders,[cell[j]])

            spike_times = (data[0]['times'])[selected_senders[0]]

            for t in np.arange(0.0,sim_time,bin_size):
                spikes = spike_times[np.where((spike_times >= t) & (spike_times < t+\
                bin_size))[0]]
                PSTHs[j,int(t/bin_size)] += len(spikes)

        # There are as many PSTHs as cells in every population (topographical plot)
        else:

            # The cell list is the same in all layers
            if(isinstance(layer_sizes, list) == False):
                cell_list = cell
            # The cell list is different in every layer
            else:
                cell_list = list(np.arange(layer_sizes[j]))

            [data,senders,pop] = getData(population,model,recorders,cell_list)

            for c in cell_list:
                selected_senders = np.where(senders==pop[c])
                spike_times = (data[0]['times'])[selected_senders[0]]

                for t in np.arange(0.0,sim_time,bin_size):
                    spikes = spike_times[np.where((spike_times >= t) & (spike_times < t+\
                    bin_size))[0]]
                    PSTHs[c,j,int(t/bin_size)] += len(spikes)

        j+=1

# Gaussian kernel function for single-trial estimation of neuronal firing rates [2]
def kernel(t,sigma):
    return (1.0 / (np.sqrt(2.0*np.pi)*sigma)) * np.exp(-t**2 / (2.0 * sigma**2))

# Method to estimate the neuronal firing rate from single-trial spike trains [2]
def singleTrialPSTH(start_time,sim_time,spike_times):

    sigma = 0.020 # s
    time = np.arange(0.0,sim_time/1000.0,0.001)
    interval = time[np.where(time >= start_time/1000.0)[0]]

    lambda_func = np.zeros(len(interval))

    # Sum over kernel functions
    for ti in spike_times/1000.0:
        lambda_func += kernel(interval-ti,sigma)

    return interval*1000.0,lambda_func

# Save spike times to file
def saveSpikes(N,recorders,recorded_models,IDs,folder,visual_stage,stim,trial,
layer_sizes,path = '../../data/'):

    # Create a folder for every trial
    if os.path.isdir(path+visual_stage+"/spikes/"+folder+"/"+str(trial)) == False:
        os.system("mkdir "+path+visual_stage+"/spikes/"+folder+"/"+str(trial))

    k = 0

    for population, model in recorded_models:

        spike_times = []

        if(isinstance(layer_sizes, list) == False):
            cell_list = np.arange(N*N)
            [data,senders,pop] = getData(population,model,recorders,np.arange(N*N))

        else:
            cell_list = list(np.arange(layer_sizes[k]))
            [data,senders,pop] = getData(population,model,recorders,cell_list)

        for cell in cell_list:
            selected_senders = np.where(senders==pop[cell])
            spike_times.append( (data[0]['times'])[selected_senders[0]] )

        text_file = open(path+visual_stage+"/spikes/"+folder+"/"+str(trial)+"/"+IDs[k]+stim+".spikes", "w")
        k+=1

        for line in np.arange(0,len(spike_times)):
            for ch in spike_times[line]:
                text_file.write(str(ch))
                text_file.write(",")
            text_file.write(os.linesep)

        text_file.close()

# Load spike times from file
def loadSpikes(N,IDs,folder,visual_stage,stim,trial,layer_sizes,path="../../data/"):

    all_spikes = []

    k = 0

    for ID in IDs:
        spike_times = []
        lines = [line.rstrip('\n') for line in open(path+visual_stage+"/spikes/"+folder+"/"+str(trial)+"/"+ID+stim+".spikes", "r")]

        if(isinstance(layer_sizes, list) == False):
            new_N = N
        else:
            new_N = np.sqrt(layer_sizes[k])

        for n in np.arange(new_N*new_N):
            h = lines[int(n)].split(',')
            e = []
            for element in h[0:len(h)-1]:
                e.append(float(element))
            spike_times.append( e )

        all_spikes.append(spike_times)
        k+=1

    return all_spikes

# Initialize PSTHs with spike times loaded from file
def initializePSTHs(ex,visual_stage,topographical,N,labels,stim,layer_sizes,path):

    for tr in np.arange(ex.trials):
        if topographical:
            print("Trial = %s" % tr)
        # Read spikes
        if(isinstance(layer_sizes, list) == False):
            # The stimulus ID is given in the simulation script
            if stim=='':
                spike_file = loadSpikes(N,labels,ex.spike_folder,visual_stage,
                ex.stim,tr,0,path)
            # The stimulus ID is passed as an argument
            else:
                spike_file = loadSpikes(N,labels,ex.spike_folder,visual_stage,
                stim,tr,0,path)

        else:
            if stim=='':
                spike_file = loadSpikes(N,labels,ex.spike_folder,visual_stage,
                ex.stim,tr,layer_sizes,path)
            else:
                spike_file = loadSpikes(N,labels,ex.spike_folder,visual_stage,
                stim,tr,layer_sizes,path)

        # Update PSTHs
        for layer in np.arange(len(labels)):

            if topographical==False:
                if(isinstance(layer_sizes, list) == False):
                    cell_list = [ex.selected_cell]
                else:
                    cell_list = [ex.selected_cell[layer]]

            else:
                if(isinstance(layer_sizes, list) == False):
                    cell_list = np.arange(N*N)
                else:
                    cell_list = np.arange(layer_sizes[layer])

            for cell in cell_list:
                spike_times = np.array((spike_file[layer])[cell])

                for t in np.arange(0.0,ex.simtime,ex.bin_size):
                    spikes = spike_times[np.where((spike_times >= t) & (spike_times < t+\
                    ex.bin_size))[0]]

                    if topographical==False:
                        ex.PSTHs[layer,int(t/ex.bin_size)] += len(spikes)
                    else:
                        ex.top_PSTHs[cell,layer,int(t/ex.bin_size)] += len(spikes)

# Plot membrane potentials
def membranePotentials(start_time,time_step,sim_time,recorders,recorded_models,labels,
selected_cell,rows,cols,starting_row,starting_col,visual_stage,
path = '../../data/'):

    j = 0
    current_row = starting_row
    current_col = starting_col

    for population, model in recorded_models:

        if(isinstance(selected_cell, list) == False):
            [data,selected_senders,pop] = getData(population,model,recorders,[selected_cell])
        else:
            [data,selected_senders,pop] = getData(population,model,recorders,[selected_cell[j]])

        Vax = plt.subplot2grid((rows,cols), (current_row,current_col))
        V_m = (data[0]['V_m'])[selected_senders[0]] # membrane potential
        V_m = V_m[int(start_time/time_step):len(V_m)]
        time = np.arange(start_time,sim_time+time_step,time_step)

        Vax.plot( time[0:len(V_m)],V_m )
        Vax.set_title(labels[j])

        # save data
        np.savetxt(path+visual_stage+'/data/'+labels[j], V_m)
        if j==0:
            np.savetxt(path+visual_stage+'/data/time', time[0:len(V_m)])

        if(current_col<cols-1):
            current_col+=1
        else:
            current_col = 0
            current_row+=1

        j+=1

    Vax.set_xlabel('time (ms)')

# Plot PSTHs
def PSTH(start_time,time_step,sim_time,recorders,recorded_models,labels,
selected_cell,rows,cols,starting_row,starting_col,trials,PSTHs,bin_size,
visual_stage,path="../../data/"):

    j = 0
    current_row = starting_row
    current_col = starting_col

    for population, model in recorded_models:

        # Single-trial estimation of PSTH
        if trials ==1:

            if(isinstance(selected_cell, list) == False):
                [data,selected_senders,pop] = getData(population,model,recorders,[selected_cell])
            else:
                [data,selected_senders,pop] = getData(population,model,recorders,[selected_cell[j]])

            spike_times = (data[0]['times'])[selected_senders[0]]
            [PSTH_times,PSTH_array] = singleTrialPSTH(start_time,sim_time,spike_times)

        # Standard PSTH
        else:
            PSTH_array = (1000.0/bin_size) * PSTHs[j,int(start_time/bin_size):]/trials
            PSTH_times = np.arange(start_time,sim_time,bin_size)

        # Plot
        if(current_row<rows and current_col<cols):
            Vax = plt.subplot2grid((rows,cols), (current_row,current_col))
#            Vax.plot( PSTH_times, PSTH_array,'b')
            Vax.bar( PSTH_times, PSTH_array, bin_size, color="blue")
            if trials ==1:
                Vax.plot( spike_times , np.ones(len(spike_times)) ,"r*")
            Vax.set_title(labels[j])

        # save data
        np.savetxt(path+visual_stage+'/data/'+labels[j]+'_PSTH', PSTH_array[0::10])

        if j==0:
            np.savetxt(path+visual_stage+'/data/PSTH_times_interp', PSTH_times[0::10])

        if(current_col<cols-1):
            current_col+=1
        else:
            current_col = 0
            current_row+=1

        j+=1

    Vax.set_xlabel('time (ms)')

# Spatial tuning curve: it can be either spatial-frequency curve or area-response curve
def spatialTuning(x_axis,amplitude,phase,labels,rows,cols,starting_row,
starting_col,x_axis_label,y_axis_label):

    # x-array for interpolation
    x = np.arange(x_axis[0],x_axis[len(x_axis)-1],
    x_axis[len(x_axis)-1]/100.0)

    current_row = starting_row
    current_col = starting_col

    # Size of every subplot
    size_row = (0.9/rows)
    size_col = (0.9/cols)

    for n in np.arange(len(amplitude)):

        if current_row < rows:

            gs = gridspec.GridSpec(2, 1)

            gs.update(top=0.9-(size_row*current_row),
            bottom=0.9-size_row*(current_row + 0.5),
            left=0.1+size_col*current_col,
            right=0.1+size_col*(current_col + 0.5),
            hspace=0.05)

            Vax = plt.subplot(gs[0, 0])

            if len(amplitude[0,:]) > 3:
                f = interpolate.interp1d(x_axis, amplitude[n],'cubic')
                Vax.plot(x,f(x))
            Vax.plot(x_axis,amplitude[n],'ro')
            Vax.set_ylabel(y_axis_label)
            Vax.set_title(labels[n])
            Vax.axes.get_xaxis().set_ticks([])

            Vax = plt.subplot(gs[1, 0])

            if len(phase[0,:]) > 3:
                f = interpolate.interp1d(x_axis, phase[n],'cubic')
                Vax.plot(x,f(x))
            Vax.plot(x_axis,phase[n],'ro')
            Vax.set_ylabel('phase\n(deg)')
            Vax.set_xlabel(x_axis_label)

            x0, x1, y0, y1 = Vax.axis()
            Vax.axis((x0,x1,y0-5.0,y1+5.0))

        if(current_col<cols-1):
            current_col+=1
        else:
            current_col = 0
            current_row+=1


# 2D representation of the time-average activity of every population
def topographical(fig,N,time_intervals,time_step,sim_time,recorders,
recorded_models,labels,rows,cols,V_mins,V_maxs,spikes,trials,PSTHs,bin_size,
top_PSTH_index,layer_sizes,show_pop_avg = False,visual_stage='retina',
path="../../data/"):

    current_row = 0
    current_col = 0

    for n in np.arange(len(time_intervals)-1):

        # Corresponding index values for the time intervals
        intervals_Vm = [int(time_intervals[n]/time_step),int(time_intervals[n+1]/time_step)]
        intervals_PSTH_single_trial = [int(time_intervals[n]),int(time_intervals[n+1])]
        intervals_PSTH_standard = [int(time_intervals[n]/bin_size),int(time_intervals[n+1]/bin_size)]

        # Image used for the population average
        pop_im = np.zeros((N,N))

        j = 0
        for population, model in recorded_models:
            # Side length of the image
            if(isinstance(layer_sizes, list) == False):
                im_side = N
            else:
                im_side = int(np.sqrt(layer_sizes[j]))

            # Factor used in the population average for the upscaling of images
            # with im_side smaller than N
            pop_factor = N/im_side

            # Image used for the j-population
            im = np.zeros((im_side,im_side))

            # Create new subplot and update row and col positions
            if show_pop_avg ==False:
                Vax = plt.subplot2grid((rows+1,cols), (current_row,n))

            if(current_row<rows-1):
                current_row+=1
            else:
                current_row = 0

            # List of cell IDs
            if(isinstance(layer_sizes, list) == False):
                cell_list = np.arange(N*N)
            else:
                cell_list = np.arange(layer_sizes[j])

            # Only for single-trial estimation of PSTH and membrane potentials
            if trials ==1:
                [data,senders,pop] = getData(population,model,recorders,cell_list)
            if spikes==False:
                [data,senders,pop] = getData(population,model,recorders,cell_list)

            # Create 2D time-average plot

            # Standard PSTH (faster computation)
            if trials > 1 and spikes:
                PSTH_array = np.array((1000.0/bin_size) * PSTHs[0:im_side*im_side,
                top_PSTH_index[j],:]/trials)
                PSTH_array = np.sum(PSTH_array[:,intervals_PSTH_standard[0]:
                    intervals_PSTH_standard[1]],axis = 1)/\
                    ((intervals_PSTH_standard[1]-intervals_PSTH_standard[0]))
                im = np.reshape(PSTH_array,(im_side,im_side))

            # Membrane potential (tested only with retina simulations, not
            # thalamocortical ones)
            if spikes==False:
                number_samples = len(np.where(senders==pop[0])[0])
                V_m = np.array(data[0]['V_m'])
                V_m = (np.reshape(V_m,(number_samples,im_side*im_side))).T

                V_m = np.sum(V_m[:,intervals_Vm[0]:intervals_Vm[1]],axis = 1)/\
                                        ((intervals_Vm[1]-intervals_Vm[0]))
                im = np.reshape(V_m,(im_side,im_side))

            # Slower for-loop
            c=0
            for x in np.arange(0,im_side):
                for y in np.arange(0,im_side):

                    if(spikes):
                        # Single-trial estimation of PSTH
                        if trials ==1:
                            selected_senders = np.where(senders==pop[c])
                            spike_times = (data[0]['times'])[selected_senders[0]]

                            [PSTH_times,PSTH_array] = singleTrialPSTH(0.0,sim_time,spike_times)
                            im[int(x),int(y)] = np.sum(PSTH_array[intervals_PSTH_single_trial[0]:
                                intervals_PSTH_single_trial[1]])/\
                                ((intervals_PSTH_single_trial[1]-intervals_PSTH_single_trial[0]))

#                        # Standard PSTH
#                        else:
#                            PSTH_array = (1000.0/bin_size) * PSTHs[c,top_PSTH_index[j],:]/trials
#                            im[int(x),int(y)] = np.sum(PSTH_array[intervals_PSTH_standard[0]:
#                               intervals_PSTH_standard[1]])/\
#                               ((intervals_PSTH_standard[1]-intervals_PSTH_standard[0]))

#                    else:
#                        selected_senders = np.where(senders==pop[c])
#                        V_m = (data[0]['V_m'])[selected_senders[0]]

#                        im[int(x),int(y)] = np.sum(V_m[intervals_Vm[0]:intervals_Vm[1]])/\
#                        ((intervals_Vm[1]-intervals_Vm[0]))
                    c+=1

                    # Population average: upscaling of images of populations with
                    # im_side smaller than N. The larger pop_factor, the smaller the
                    # contribution of the j-population to the population average
                    if show_pop_avg:
                        for pop_avg_x in np.arange(int(pop_factor * x),int(pop_factor * x) + pop_factor):
                            for pop_avg_y in np.arange(int(pop_factor * y),int(pop_factor * y) + pop_factor):
                                pop_im[int(pop_avg_x),int(pop_avg_y)] += im[int(x),int(y)]/(pop_factor**2)

            # Plot of the 2D activity profile for the j-population
            if show_pop_avg ==False:

                im_plot=Vax.matshow(im,vmin=V_mins[j], vmax=V_maxs[j])
                cbar1 = fig.colorbar(im_plot,ticks=[V_mins[j],V_maxs[j]])

                if(n==0):
                    Vax.set_title(labels[j]+'\n'+str(intervals_Vm))
                else:
                    Vax.set_title(str(intervals_Vm))

                Vax.axes.get_xaxis().set_ticks([])
                Vax.axes.get_yaxis().set_ticks([])

                # Save image
#                fake_fig = plt.figure()
#                im_plot = plt.matshow(im,vmin=V_mins[j], vmax=V_maxs[j])
#                plt.axis('off')
#                plt.savefig(path+visual_stage+'/data/'+str(model)+str(n)+'.png')

            j+=1

        # Plot of the 2D activity profile for the population average
        if show_pop_avg:

            Vax = plt.subplot2grid((rows+1,cols), (rows,n))
            im_plot=Vax.matshow(pop_im/len(recorded_models),vmin=V_mins[0], vmax=V_maxs[0])
            cbar1 = fig.colorbar(im_plot,ticks=[V_mins[0],V_maxs[0]])
            Vax.axes.get_xaxis().set_ticks([])
            Vax.axes.get_yaxis().set_ticks([])
            if(n==0):
                Vax.set_title('Population average '+'\n'+str(intervals_Vm))
            else:
                Vax.set_title(str(intervals_Vm))


# 2D receptive-field profile
def receptiveField(fig,N,RF_intervals,s_layers_to_record,labels,RF_bright,RF_dark):

    RF_index = 0

    for interval in RF_intervals:
        counter = 0

        for population, model in s_layers_to_record:

            # 3D plot
            Vax = fig.add_subplot(len(s_layers_to_record),len(RF_intervals),1 +\
            RF_index + counter*len(RF_intervals), projection='3d')
            X = np.arange(N)
            Y = X
            X, Y = np.meshgrid(X, Y)
            Z = RF_bright[RF_index,counter,:,:] - RF_dark[RF_index,counter,:,:]
            im_plot = Vax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm',
                                   linewidth=0, antialiased=False)

            # 2D plot
#            Vax = plt.subplot2grid((len(s_layers_to_record),len(RF_intervals)),(counter,RF_index))
#            im_plot = Vax.matshow(RF_bright[RF_index,counter,:,:] - RF_dark[RF_index,counter,:,:])

            Vax.axes.get_xaxis().set_ticks([])
            Vax.axes.get_yaxis().set_ticks([])

            if RF_index == 0:
                Vax.set_title(str(labels[counter] +'\n'+ str(RF_intervals[RF_index])))
            else:
                Vax.set_title(str(RF_intervals[RF_index]))

            cbar = fig.colorbar(im_plot)

            counter+=1

        RF_index+=1

# Show a video sequence of the input stimulus
def videoSeq(number_cells,inputIm,simtime,resolution,video_step):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(number_cells)
    Y = X
    X, Y = np.meshgrid(X, Y)
    Z = inputIm[:,:,0]
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm',
                           linewidth=0, antialiased=False)
    ax.set_title('Input stimulus. Time = 0.0 ms',y=1.08)
    fig.show()
    ax.axes.figure.canvas.draw()

    for t in np.arange(0.0,int(simtime/resolution),video_step/resolution):
        surf.remove()
        Z = inputIm[:,:,t]
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm',
        linewidth=0, antialiased=False)
        ax.set_title('Input stimulus. Time = %s ms'%str(t),y=1.08)
        ax.axes.figure.canvas.draw()
