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
# Name: run_network
#
# Description: creation and simulation of the retina network
#
# Warning: when using more than 1 thread in NEST, topology connections are not
# correctly configured between the model types 'parvo_neuron' since they have no
# proxies and are local receivers.
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

import sys,os
import nest
import nest.topology as tp
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process, Pipe
from sys import stdout
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','networks'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..','models'))

import cone_VanHateren

# Multiprocessing computation of cone responses: target function
def parallel_cone(pipe,cells,time,cone_input,cone_layer,Vis_dark,Vis_resting_potential):
    # Initialize array of cone_response copying cone_input
    cone_response = cone_input

    for cell in cells:
        if multiprocessing.current_process().name=="root":
            progress = 100*(cell-cells[0])/len(cells)
            stdout.write("\r progress: %d %%"% progress)
            stdout.flush()

        # Time-driven simulation
        for t in np.arange(0,time):
            # Update dynamics of the model
            cone_layer[cell].feedInput(cone_input[cell,t])
            cone_layer[cell].update()
            # Record response
            cone_response[cell,t] = (cone_layer[cell].LF_taum.last_values[0] -\
            cone_layer[cell].LF_tauh.last_values[0] - Vis_dark - Vis_resting_potential)

    pipe.send(cone_response[cells,:])
    pipe.close()

#! ================
#! Class runNetwork
#! ================

class runNetwork(object):

    # Sigmoid activation functions of the cation channels of ON and OFF bipolar
    # cells (synapse photoreceptors -> bipolar cells)
    def sigmoid_OFF_Bip(self,V_pre,a,b):
        return ( 1.0 / (1.0 + np.exp(-(V_pre-a)/b)) )
    def sigmoid_ON_Bip(self,V_pre,a,b):
        return ( 1.0 / (1.0 + np.exp((V_pre-a)/b)) )

    def __init__(self,simtime):

        # Simulation parameters
        self.Params = {
            'N': 40, # number of cells per row
            'visSize': 2.0, # visual angle (degrees)
            'NEST_threads': 1, # threads used in NEST simulation (must be 1)
            'photo_processes': 10, # processes used to compute the response of one
                                  # type of photoreceptor (the total number of
                                  # processes is this parameter multiplied by 3,
                                  # the number of cone types). This parameter must
                                  # be a multiple of the number of cells (N*N)
            'resolution': 1.0, # simulation step (in ms)
            'simtime': simtime, # ms
            'load_cone_from_file': False # load cone responses from file
        }

        # Cone layers
        self.cone_layers = []

        # Records of photoreceptor response
        self.cone_response = np.zeros((3,self.Params['N']*self.Params['N'],
        int(self.Params['simtime']/self.Params['resolution'])))

        # Time array
        self.time = np.zeros(int(self.Params['simtime']/self.Params['resolution']))
        for t in np.arange(0,int(self.Params['simtime']/self.Params['resolution'])):
            self.time[t] = (t*self.Params['resolution'])

        # Records of the temporal evolution of cation channels in ON and OFF
        # bipolar cells
        self.L_cone_metabotropic = np.zeros((self.Params['N']*self.Params['N'],
        int(self.Params['simtime']/self.Params['resolution'])))
        self.M_cone_metabotropic = np.zeros((self.Params['N']*self.Params['N'],
        int(self.Params['simtime']/self.Params['resolution'])))
        self.S_cone_metabotropic = np.zeros((self.Params['N']*self.Params['N'],
        int(self.Params['simtime']/self.Params['resolution'])))
        self.L_cone_ionotropic = np.zeros((self.Params['N']*self.Params['N'],
        int(self.Params['simtime']/self.Params['resolution'])))
        self.M_cone_ionotropic = np.zeros((self.Params['N']*self.Params['N'],
        int(self.Params['simtime']/self.Params['resolution'])))
        self.S_cone_ionotropic = np.zeros((self.Params['N']*self.Params['N'],
        int(self.Params['simtime']/self.Params['resolution'])))

        # Constants calculated by using a dark stimulus
        # (bkg_illuminance = pulse_amplitude = 0)
        self.Vis_dark = 30.39 # mV
        # Constant to force a resting potential at the cone's terminal that
        # ranges from −35 to − 45 mV
        self.Vis_resting_potential = 30.0 # mV


    #! ========================
    #! Photoreceptors' response
    #! ========================

    def simulatePhotoreceptors(self,cone_input):
        # Parameters (default from Van Hateren's model)
        tauR = 0.49
        tauE = 16.8
        cb = 2.8 * 10**(-3)
        kb = 1.63 * 10**(-4)
        nX = 1.0
        tauC = 2.89
        ac = 9.08 * 10**(-2)
        nc = 4.0
        taum = 4.0
        ais = 7.09 * 10**(-2)
        gamma = 0.678
        tauis = 56.9
        gs = 0.5 # smaller than in the original model
        tau1 = 4.0
        tau2 = 4.0
        tauh = 20.0
        githr = 0.4

        # Create layer of cones
        L_cones = []
        M_cones = []
        S_cones = []
        for x in np.arange(self.Params['N']):
            for y in np.arange(self.Params['N']):
                cone_1 = cone_VanHateren.cone(
                self.Params['resolution'],tauR,tauE,cb,kb,nX,tauC,ac,nc,taum,ais,
                gamma,tauis,gs,tau1,tau2,tauh,githr,True)
                cone_2 = cone_VanHateren.cone(
                self.Params['resolution'],tauR,tauE,cb,kb,nX,tauC,ac,nc,taum,ais,
                gamma,tauis,gs,tau1,tau2,tauh,githr,True)
                cone_3 = cone_VanHateren.cone(
                self.Params['resolution'],tauR,tauE,cb,kb,nX,tauC,ac,nc,taum,ais,
                gamma,tauis,gs,tau1,tau2,tauh,githr,True)
                L_cones.append(cone_1)
                M_cones.append(cone_2)
                S_cones.append(cone_3)

        self.cone_layers.append(L_cones)
        self.cone_layers.append(M_cones)
        self.cone_layers.append(S_cones)

        # Compute response
        if(self.Params['load_cone_from_file'] == False):
            print ("\n\n--- Computing photoreceptors' response ---\n")
            start_c = time.time()

            # Distribute simulation of cell's responses among workers
            all_cells = np.arange(len(self.cone_layers[0]))
            cells_per_process = int(len(all_cells)/self.Params['photo_processes'])
            print ("cells/process: ", cells_per_process)
            print ("Processes created: ",self.Params['photo_processes']*3)

            # Process and pipe arrays
            pr_array = []
            pi_array = []

            for pr in np.arange(self.Params['photo_processes']):
                sel_cells = all_cells[pr*cells_per_process:(pr+1)*cells_per_process]
                # Create process and send a pipe connection
                for cone_id in np.arange(3):
                    parent_conn, child_conn = Pipe()
                    if(pr==0 and cone_id ==0):
                        id = "root"
                    else:
                        id = "worker"
                    p = Process(name=id,target=parallel_cone,
                    args=(child_conn,sel_cells,int(self.Params['simtime']/self.Params['resolution']),
                    cone_input[cone_id],self.cone_layers[cone_id],self.Vis_dark,self.Vis_resting_potential))
                    p.start()
                    pr_array.append(p)
                    pi_array.append(parent_conn)

            # Gather results
            cn = 0
            for pr in np.arange(self.Params['photo_processes']):
                sel_cells = all_cells[pr*cells_per_process:(pr+1)*cells_per_process]

                for cone_id in np.arange(3):
                    self.cone_response[cone_id,sel_cells,:] = pi_array[cn].recv()
                    pr_array[cn].join()
                    cn+=1

            end_c = time.time()
            print ("\n time elapsed (s): ",(end_c - start_c))

            # Save records to file
            np.savetxt('../../data/retina/results/L_cone_response.out', self.cone_response[0])
            np.savetxt('../../data/retina/results/M_cone_response.out', self.cone_response[1])
            np.savetxt('../../data/retina/results/S_cone_response.out', self.cone_response[2])
            np.savetxt('../../data/retina/results/time.out', self.time)

        # Load records from file
        else:
            self.cone_response[0] = np.loadtxt('../../data/retina/results/L_cone_response.out')
            self.cone_response[1] = np.loadtxt('../../data/retina/results/M_cone_response.out')
            self.cone_response[2] = np.loadtxt('../../data/retina/results/S_cone_response.out')
            self.time = np.loadtxt('../../data/retina/results/time.out')


    #! =================
    #! NEST simulation
    #! =================

    def NESTSimulation(self,mosaic,load_mod = True):
        if load_mod:
            # Install modules just once
            model = nest.Models(mtype='nodes',sel='parvo_neuron')
            if not model:
                nest.Install("parvo_neuron_module")

            model = nest.Models(mtype='nodes',sel='AII_amacrine')
            if not model:
                nest.Install("AII_amacrine_module")

            model = nest.Models(mtype='nodes',sel='ganglion_cell')
            if not model:
                nest.Install("ganglion_cell_module")

        # Seeds
        np.random.seed(int(time.time()))
        self.seeds = np.arange(self.Params['NEST_threads']) + int((time.time()*100)%2**32)

        # NEST Kernel and Network settings
        nest.ResetKernel()
        nest.ResetNetwork()
        nest.SetKernelStatus(
        {"local_num_threads": self.Params['NEST_threads'],
        'resolution': self.Params['resolution'], "rng_seeds": list(self.seeds)})

        # import network description
        import retina

        # get network info
        models, layers, conns  = retina.get_Network(self.Params)

        # Create models
        for m in models:
                nest.CopyModel(m[0], m[1], m[2])

        print ("\n---Creating layers---\n")
        # Create layers, store layer info in Python variable
        layer_IDs = []
        for l in layers:
            exec ("%s = tp.CreateLayer(%s)" % (l[0],l[1]),globals())
            exec ("copy_var = %s" % l[0],globals())
            layer_IDs.append([l[0],copy_var,l[1]['elements']])
#            print (l[0])

        print ("\n---Connecting layers---\n")
        # Create connections, need to insert variable names
        for c in conns:
                eval('tp.ConnectLayers(%s,%s,c[2])' % (c[0], c[1]))
#                print ('tp.ConnectLayers(%s,%s)' % (c[0], c[1]))

        # Initialize generators with the synaptic conductance values for ON and OFF
        # bipolar cells according to the photoreceptor mosaic
        cell = 0
        for x in np.arange(self.Params['N']):
            for y in np.arange(self.Params['N']):
                # parameters of the sigmoid: ensure that glutamate release of the cone cell
                # is about 1.0 in dark and approaches to 0.0 with inputs > 1000.0 trolands
                if(mosaic[0,x,y]==1):
                    self.L_cone_metabotropic[cell,:] = self.sigmoid_ON_Bip(self.cone_response[0,cell,:],-50.0,4.0)
                else:
                    self.L_cone_metabotropic[cell,:] = np.zeros(len(self.cone_response[0,cell,:]))
                nest.SetStatus([tp.GetElement(L_cones_metabotropic,(x,y))[0]],
                [{'amplitude_times':self.time,'amplitude_values':list(self.L_cone_metabotropic[cell,:])}])

                if(mosaic[1,x,y]==1):
                    self.M_cone_metabotropic[cell,:] = self.sigmoid_ON_Bip(self.cone_response[1,cell,:],-50.0,4.0)
                else:
                    self.M_cone_metabotropic[cell,:] = np.zeros(len(self.cone_response[1,cell,:]))
                nest.SetStatus([tp.GetElement(M_cones_metabotropic,(x,y))[0]],
                [{'amplitude_times':self.time,'amplitude_values':list(self.M_cone_metabotropic[cell,:])}])

                if(mosaic[2,x,y]==1):
                    self.S_cone_metabotropic[cell,:] = self.sigmoid_ON_Bip(self.cone_response[2,cell,:],-50.0,4.0)
                else:
                    self.S_cone_metabotropic[cell,:] = np.zeros(len(self.cone_response[2,cell,:]))
                nest.SetStatus([tp.GetElement(S_cones_metabotropic,(x,y))[0]],
                [{'amplitude_times':self.time,'amplitude_values':list(self.S_cone_metabotropic[cell,:])}])

                if(mosaic[0,x,y]==1):
                    self.L_cone_ionotropic[cell,:] = self.sigmoid_OFF_Bip(self.cone_response[0,cell,:],-50.0,4.0)
                else:
                    self.L_cone_ionotropic[cell,:] = np.zeros(len(self.cone_response[0,cell,:]))
                nest.SetStatus([tp.GetElement(L_cones_ionotropic,(x,y))[0]],
                [{'amplitude_times':self.time,'amplitude_values':list(self.L_cone_ionotropic[cell,:])}])

                if(mosaic[1,x,y]==1):
                    self.M_cone_ionotropic[cell,:] = self.sigmoid_OFF_Bip(self.cone_response[1,cell,:],-50.0,4.0)
                else:
                    self.M_cone_ionotropic[cell,:] = np.zeros(len(self.cone_response[1,cell,:]))
                nest.SetStatus([tp.GetElement(M_cones_ionotropic,(x,y))[0]],
                [{'amplitude_times':self.time,'amplitude_values':list(self.M_cone_ionotropic[cell,:])}])

                if(mosaic[2,x,y]==1):
                    self.S_cone_ionotropic[cell,:] = self.sigmoid_OFF_Bip(self.cone_response[2,cell,:],-50.0,4.0)
                else:
                    self.S_cone_ionotropic[cell,:] = np.zeros(len(self.cone_response[2,cell,:]))
                nest.SetStatus([tp.GetElement(S_cones_ionotropic,(x,y))[0]],
                [{'amplitude_times':self.time,'amplitude_values':list(self.S_cone_ionotropic[cell,:])}])

                cell+=1


        ## Check-point: Visualization functions
#        fig = tp.PlotLayer(H1_Horizontal_cells,nodesize =80)
#        ctr = tp.FindCenterElement(H1_Horizontal_cells)
#        tp.PlotTargets(ctr,Midget_bipolar_cells_L_ON,fig = fig,mask=conns[12][2]['mask'],
#        kernel=conns[12][2]['kernel'],src_size=250,tgt_color='red',tgt_size=20,
#        kernel_color='green')
#        plt.show()

        return layer_IDs

    #! ================================
    #! Recording devices and simulation
    #! ================================

    def runSimulation(self,recorded_models,recorded_spikes):

        nest.CopyModel('multimeter', 'RecordingNode',
                {'interval'   : self.Params['resolution'],
                'record_from': ['V_m'],
                'record_to'  : ['memory'],
                'withgid'    : True,
                'withtime'   : False})

        recorders = []

        for population, model in recorded_models:

                rec = nest.Create('RecordingNode')

                recorders.append([rec,population,model])
                tgts = [nd for nd in nest.GetLeaves(population)[0] if nest.GetStatus([nd],
                'model')[0]==model]
                nest.Connect(rec, tgts)

        nest.CopyModel('spike_detector', 'RecordingSpikes',
                {"withtime": True,
                "withgid": True,
                "to_file": False})

        spike_detectors = []

        for population, model in recorded_spikes:
                rec = nest.Create('RecordingSpikes')
                spike_detectors.append([rec,population,model])
                tgts = [nd for nd in nest.GetLeaves(population)[0] if nest.GetStatus([nd],
                'model')[0]==model]
                nest.Connect(tgts, rec)


        print ("\n--- Simulation ---\n")
        nest.SetStatus([0],{'print_time': True})
        nest.Simulate(self.Params['simtime'])

        return recorders,spike_detectors
