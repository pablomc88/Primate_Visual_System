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
# Name: retina
#
# Description:
#
# This script defines the network model using the Topology module of NEST. It
# specifies neuron models, two-dimensional neuronal layers and connections.
#
# We made certain assumptions about the model:
#
# 1) Photoreceptors release only one type of neurotransmitter, glutamate.
# However, bipolar cells react to this stimulus with two different responses,
# ON-center and OFF-center responses [2,3]. Ionotropic glutamate receptors are
# positively coupled to the synaptic cation channel of OFF bipolar cells, which
# is opened with an increase of glutamate. On the contrary, ON bipolar cells
# are negatively coupled to the synaptic cation channel and glutamate acts
# essentially as an inhibitory transmitter, closing the cation channel. To
# simulate the activation function of this cation channel based on the cone
# membrane potential, we used a sigmoid function whose exponent is negative for
# OFF bipolar cells (standard sigmoid) and positive for ON bipolar cells
# (inverted sigmoid). These functions are implemented in the script
# 'simulation/retina/run_network.py'. Two populations of
# 'step_current_generator' are used to feed ON and OFF bipolar cells,
# respectively. Each population connect to one of the two sigmoid activation
# functions.
#
# 2) In the synapse horizontal cell-bipolar cell, although both bipolar cell
# types express the same ionotropic GABA receptors, GABA release from
# horizontal cells can evoke opposite responses. One evidence suggests that
# GABA evokes opposite responses if chloride equilibrium potentials of the
# synaptic chloride channel in the two bipolar cell types are on opposite sides
# of the bipolar cell’s resting potential [4]. In our model, ON bipolar cells
# receive excitatory synapses from horizontal cells, which have a positive
# reversal potential taking as a reference the bipolar cell’s resting potential
# (0 mV), and OFF bipolar cells receive inhibitory synapses, which have a
# negative reversal potential (-70 mV).
#
# 3) Among all types of amacrine cells, the model includes only the AII
# amacrine cell (narrow-field, bistratified) since it is the most studied
# amacrine cell and the most numerous type in the mammalian retina [5]. It is
# shown that the AII amacrine functionality extends to cone-mediated (photopic)
# vision [6,7]. Under cone-driven conditions, ON cone bipolar cells excite AII
# amacrine cells through gap junctions and, in turn, AII amacrince cells
# release inhibitory neurotransmitters onto OFF bipolar cells and OFF ganglion
# cells.
#
# 4) Parameter values of neuron models were chosen as generic as possible (see,
# for example, values of C_m , g_L , E_ex and E_in). The leak reversal
# potential, E_L , was adjusted in horizontal cells and bipolar cells to force
# a resting potential in the dark of about -45 mV, as observed experimentally
# [8,9], and in amacrine cells for a resting potential of about -65 mV. For
# ganglion cells, we chose values of the leak reversal potential and the
# threshold potential, V_th , to keep the cell constantly depolarized,
# resulting in a spontaneous firing rate of about 40 spikes/s. Values of the
# synaptic activation functions, θ_syn and k_syn , were set to force a synaptic
# threshold below resting potential [9].
#
# 5) Synaptic dictionaries include a circular mask of radius R and a kernel of
# constant probability. Weights of synaptic connections are generated according
# to a Gaussian distribution of standard deviation sigma. The value of sigma in
# midget cells corresponds to the radius of the receptive field center of P
# cells (0.03 deg) [10]. The surround of the receptive field is accounted for by
# horizontal cells (0.1 deg). To create the spatially coextensive receptive
# field of the blue-yellow pathway, the value of sigma of S-ON bipolar cells is
# the same as that of diffuse bipolar cells. To approximate experimental
# results, both values are set to 0.05 degrees.
#
# References:
#
# [1] Martinez-Cañada, P., Morillas, C., Pelayo, F. (2017). A Conductance-Based
# Neuronal Network Model for Color Coding in the Primate Foveal Retina. In IWINAC
# 2017
# [2] Snellman, J., Kaur, T., Shen, Y., Nawy, S.: Regulation of on bipolar cell
# activity. Progress in retinal and eye research 27(4), 450–463 (2008)
# [3] Nawy, S., Jahr, C.E.: Suppression by glutamate of cgmp-activated
# conductance in retinal bipolar cells. Nature 346(6281), 269 (1990)
# [4] Vardi, N., Zhang, L.L., Payne, J.A., Sterling, P.: Evidence that
# different cation chloride cotransporters in retinal neurons allow opposite
# responses to gaba. Journal of Neuroscience 20(20), 7657–7663 (2000)
# [5] Masland, R.H.: The fundamental plan of the retina. Nature neuroscience
# 4(9),877–886 (2001)
# [6] Demb, J.B., Singer, J.H.: Intrinsic properties and functional circuitry
# of the aii amacrine cell. Visual neuroscience 29(01), 51–60 (2012)
# [7] Manookin, M.B., Beaudoin, D.L., Ernst, Z.R., Flagel, L.J., Demb, J.B.:
# Disinhibition combines with excitation to extend the operating range of the
# off visual pathway in daylight. Journal of Neuroscience 28(16), 4136–4150
# (2008)
# [8] Arman, A.C., Sampath, A.P.: Dark-adapted response threshold of off
# ganglion cells is not set by off bipolar cells in the mouse retina. Journal
# of neurophysiology 107(10), 2649–2659 (2012)
# [9] Smith, R.G.: Simulation of an anatomically defined local circuit: the cone-
# horizontal cell network in cat retina. Visual neuroscience 12(03), 545–561
# (1995)
# [10] Croner, L.J., Kaplan, E.: Receptive fields of p and m ganglion cells
# across the primate retina. Vision research 35(1), 7–24 (1995)
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)
#

import nest
import nest.topology as tp
import numpy as np

# Update dictionaries
def updateDicts(dict1, dict2):
    assert(isinstance(dict1, dict))
    assert(isinstance(dict2, dict))

    tmp = dict1.copy()
    tmp.update(dict2)
    return tmp

# Return lists of layers, models and connections
def get_Network(params):
    models = get_Models()
    layers = get_Layers(params)
    conns = get_Connections(params)
    return models,layers,conns

# Parameters of neuron models for each layer
def get_Models():

    # Horizontal cells
    retina_parvo_horizontal_cell  = 'parvo_neuron'
    retina_parvo_horizontal_cell_params = {
    "C_m": 100.0,
    "g_L": 10.0,
    "E_L": -60.0,
    "E_ex": 0.0,
    "E_in": -70.0,
    "a" : -50.0,
    "b" : 4.0
    }

    # ON Midget bipolar cell
    retina_parvo_ON_bipolar_cell  = 'parvo_neuron'
    retina_parvo_ON_bipolar_cell_params = {
    "C_m": 100.0,
    "g_L": 10.0,
    "E_L": -60.0,
    "E_ex": 0.0,
    "E_in": -70.0,
    "a" : -35.0,
    "b" : 3.0
    }

    # OFF Midget bipolar cell
    retina_parvo_OFF_bipolar_cell  = 'parvo_neuron'
    retina_parvo_OFF_bipolar_cell_params = {
    "C_m": 100.0,
    "g_L": 10.0,
    "E_L": -50.0,
    "E_ex": 0.0,
    "E_in": -70.0,
    "a" : -35.0,
    "b" : 3.0
    }

    # AII amacrine cell
    retina_parvo_amacrine_cell  = 'AII_amacrine'
    retina_parvo_amacrine_cell_params = {
    "C_m": 100.0,
    "g_L": 10.0,
    "E_L": -60.0,
    "a" : -55.0,
    "b" : 3.0,
    "g_ex": 20.0
    }

    # Midget ganglion cell
    retina_parvo_ganglion_cell  = 'ganglion_cell'
    retina_parvo_ganglion_cell_params = {
    "C_m": 100.0,
    "g_L": 10.0,
    "E_ex": 0.0,
    "E_in": -70.0,
    "E_L": -60.0,
    "V_th": -55.0,
    "V_reset": -60.0,
    "t_ref":  2.0,
    "rate": 0.0
    }

    # gaussian noise current
    retina_noise = 'noise_generator'
    retina_noise_params = {
    'mean': 0.0, # pA
    'std': 1.0 # pA
    }


    models = [(retina_parvo_horizontal_cell,'retina_parvo_horizontal_cell',
            retina_parvo_horizontal_cell_params),
            (retina_parvo_ON_bipolar_cell,'retina_parvo_ON_bipolar_cell',
            retina_parvo_ON_bipolar_cell_params),
            (retina_parvo_OFF_bipolar_cell,'retina_parvo_OFF_bipolar_cell',
            retina_parvo_OFF_bipolar_cell_params),
            (retina_parvo_amacrine_cell,'retina_parvo_amacrine_cell',
            retina_parvo_amacrine_cell_params),
            (retina_parvo_ganglion_cell,'retina_parvo_ganglion_cell',
            retina_parvo_ganglion_cell_params),
            ('step_current_generator','generator',{}),
            (retina_noise,'retina_noise', retina_noise_params)]

    return models

# Definition of neuronal layers
def get_Layers(params):

    layerProps = {
    'rows'     : params['N'],
    'columns'  : params['N'],
    'extent'   : [params['visSize'], params['visSize']],
    'edge_wrap': True
    }

    # Create layer dictionaries
    layers = [('L_cones_ionotropic',updateDicts(layerProps, {'elements': 'generator'})),
    ('M_cones_ionotropic',updateDicts(layerProps, {'elements': 'generator'})),
    ('S_cones_ionotropic',updateDicts(layerProps, {'elements': 'generator'})),
    ('L_cones_metabotropic',updateDicts(layerProps, {'elements': 'generator'})),
    ('M_cones_metabotropic',updateDicts(layerProps, {'elements': 'generator'})),
    ('S_cones_metabotropic',updateDicts(layerProps, {'elements': 'generator'})),
    ('H1_Horizontal_cells',updateDicts(layerProps, {'elements': 'retina_parvo_horizontal_cell'})),
    ('H2_Horizontal_cells',updateDicts(layerProps, {'elements': 'retina_parvo_horizontal_cell'})),
    ('Midget_bipolar_cells_L_ON',updateDicts(layerProps, {'elements': 'retina_parvo_ON_bipolar_cell'})),
    ('Midget_bipolar_cells_L_OFF',updateDicts(layerProps, {'elements': 'retina_parvo_OFF_bipolar_cell'})),
    ('Midget_bipolar_cells_M_ON',updateDicts(layerProps, {'elements': 'retina_parvo_ON_bipolar_cell'})),
    ('Midget_bipolar_cells_M_OFF',updateDicts(layerProps, {'elements': 'retina_parvo_OFF_bipolar_cell'})),
    ('Diffuse_bipolar_cells_S_ON',updateDicts(layerProps, {'elements': 'retina_parvo_OFF_bipolar_cell'})),
    ('S_cone_bipolar_cells_S_ON',updateDicts(layerProps, {'elements': 'retina_parvo_ON_bipolar_cell'})),
    ('AII_amacrine_cells',updateDicts(layerProps, {'elements': 'retina_parvo_amacrine_cell'})),
    ('Midget_ganglion_cells_L_ON',updateDicts(layerProps, {'elements': 'retina_parvo_ganglion_cell'})),
    ('Midget_ganglion_cells_L_OFF',updateDicts(layerProps, {'elements': 'retina_parvo_ganglion_cell'})),
    ('Midget_ganglion_cells_M_ON',updateDicts(layerProps, {'elements': 'retina_parvo_ganglion_cell'})),
    ('Midget_ganglion_cells_M_OFF',updateDicts(layerProps, {'elements': 'retina_parvo_ganglion_cell'})),
    ('Small_bistratified_ganglion_cells_S_ON',updateDicts(layerProps, {'elements': 'retina_parvo_ganglion_cell'})),
    ('Noise_generators',updateDicts(layerProps, {'elements': 'retina_noise'}))
    ]

    return layers

# Weights are scaled with the size of the network so that the sum of the weights
# of all incoming synapses is always equal to a constant value
def get_Relative_Weight(params,radius):

    # Create a fictional network and count the number of target connections
    layerProps = {
    'rows'     : params['N'],
    'columns'  : params['N'],
    'extent'   : [params['visSize'], params['visSize']],
    'edge_wrap': True,
    'elements': 'iaf_cond_exp'
    }

    l = tp.CreateLayer(layerProps)

    dict = {
    "connection_type":"convergent",
    "mask": {"circular": {"radius": radius}},
    "kernel": 1.0,
    "weights": {"gaussian": {"p_center": 1.0, "sigma": radius/3.0}}
    }

    tp.ConnectLayers(l,l,dict)
    ctr = tp.FindCenterElement(l)
    targets = tp.GetTargetNodes(ctr,l)

#    print ("Number of targets = ",len(targets[0]))

    conn = nest.GetConnections(ctr,targets[0])
    st = nest.GetStatus(conn)

    w = 0.0
    for n in np.arange(len(st)):
        w += st[n]['weight']

#    print ("Total weight = ",w)

    return w,len(targets[0])

# Create connections between layers
def get_Connections(params):

    # Synapse model
    nest.CopyModel('static_synapse','syn')

    # Build complete list of connections
    allconns = []

    # ----- Dictionary for the center RF of midget cells ----- #

    P_Center_dict = {
    "connection_type":"convergent",
    "mask": {"circular": {"radius": 0.09}}, # 3 x sigma
    "kernel": 1.0,
    "delays" : {"normal": {"mean": 1.0, "std": 0.25, "min": params['resolution']}},
    "synapse_model": "syn",
    "allow_autapses":False,
    "allow_multapses":False
    }

    # ----- Dictionary for the surround RF of midget cells ----- #

    P_Surround_dict = {
    "connection_type":"convergent",
    "mask": {"circular": {"radius": 0.3}},
    "kernel": 1.0,
    "delays" : {"normal": {"mean": 1.0, "std": 0.25, "min": params['resolution']}},
    "synapse_model": "syn",
    "allow_autapses":False,
    "allow_multapses":False
    }

    # ----- Dictionary for the S-ON pathway ----- #

    # Diffuse types contact multiple cones (mask a bit larger than center RF but
    # smaller than surround RF)
    P_S_ON_dict = {
    "connection_type":"convergent",
    "mask": {"circular": {"radius": 0.15}},
    "kernel": 1.0,
    "delays" : {"normal": {"mean": 1.0, "std": 0.25, "min": params['resolution']}},
    "synapse_model": "syn",
    "allow_autapses":False,
    "allow_multapses":False
    }

    # to scale weights with the size of the network
    [conn_P_Center,targets_P_Center] = get_Relative_Weight(params,P_Center_dict['mask']['circular']['radius'])
    [conn_P_Surround,targets_P_Surround] = get_Relative_Weight(params,P_Surround_dict['mask']['circular']['radius'])
    [conn_P_S_ON,targets_P_S_ON] = get_Relative_Weight(params,P_S_ON_dict['mask']['circular']['radius'])

    # ----- Cones -> Bipolar cells ----- #

    Cones_Bipolar_dict = P_Center_dict.copy()
    Cones_Bipolar_dict.update({
    "sources": {"model": "generator"}
    })

    Cones_Bipolar_L_ON_dict = Cones_Bipolar_dict.copy()
    # from [4], center radius of P cells
    Cones_Bipolar_L_ON_dict.update({"weights": {"gaussian": {"p_center": 5.0/conn_P_Center, "sigma": 0.03}},
    "targets": {"model": "retina_parvo_ON_bipolar_cell"}})

    Cones_Bipolar_L_OFF_dict = Cones_Bipolar_dict.copy()
    Cones_Bipolar_L_OFF_dict.update({"weights": {"gaussian": {"p_center": 6.5/conn_P_Center, "sigma": 0.03}},
    "targets": {"model": "retina_parvo_OFF_bipolar_cell"}})

    Cones_Bipolar_M_ON_dict = Cones_Bipolar_dict.copy()
    Cones_Bipolar_M_ON_dict.update({"weights": {"gaussian": {"p_center": 5.0/conn_P_Center, "sigma": 0.03}},
    "targets": {"model": "retina_parvo_ON_bipolar_cell"}})

    Cones_Bipolar_M_OFF_dict = Cones_Bipolar_dict.copy()
    Cones_Bipolar_M_OFF_dict.update({"weights": {"gaussian": {"p_center": 6.5/conn_P_Center, "sigma": 0.03}},
    "targets": {"model": "retina_parvo_OFF_bipolar_cell"}})

    Cones_Diffuse_bipolar_dict = P_S_ON_dict.copy()
    Cones_Diffuse_bipolar_dict.update({"weights": {"gaussian": {"p_center": 3.25/conn_P_S_ON, "sigma": 0.05}},
    "sources": {"model": "generator"},
    "targets": {"model": "retina_parvo_OFF_bipolar_cell"}})

    Cones_S_Bipolar_dict = P_S_ON_dict.copy()
    Cones_S_Bipolar_dict.update({"weights": {"gaussian": {"p_center": 5.0/conn_P_S_ON, "sigma": 0.05}},
    "sources": {"model": "generator"},
    "targets": {"model": "retina_parvo_ON_bipolar_cell"}})


    [allconns.append(['L_cones_metabotropic','Midget_bipolar_cells_L_ON',Cones_Bipolar_L_ON_dict])]
    [allconns.append(['L_cones_ionotropic','Midget_bipolar_cells_L_OFF',Cones_Bipolar_L_OFF_dict])]
    [allconns.append(['M_cones_metabotropic','Midget_bipolar_cells_M_ON',Cones_Bipolar_M_ON_dict])]
    [allconns.append(['M_cones_ionotropic','Midget_bipolar_cells_M_OFF',Cones_Bipolar_M_OFF_dict])]
    [allconns.append(['S_cones_metabotropic','S_cone_bipolar_cells_S_ON',Cones_S_Bipolar_dict])]
    [allconns.append(['L_cones_ionotropic','Diffuse_bipolar_cells_S_ON',Cones_Diffuse_bipolar_dict])]
    [allconns.append(['M_cones_ionotropic','Diffuse_bipolar_cells_S_ON',Cones_Diffuse_bipolar_dict])]


    # ----- Cones -> Horizontal cells ----- #

    # I follow results shown by Helga Kolb in Webvision, with the same amplitude
    # of H1 and H2 cells to luminance flashes. In any case, S cone elicits
    # larger response in H2 than L and M cones.
    Cones_Horizontal_dict = P_Surround_dict.copy()
    Cones_Horizontal_dict.update({
    "sources": {"model": "generator"},
    "targets": {"model": "retina_parvo_horizontal_cell"}
    })

    Cones_Horizontal_H1_dict = Cones_Horizontal_dict.copy()
    # from [4], surround radius of P cells
    Cones_Horizontal_H1_dict.update({
    "weights": {"gaussian": {"p_center": 2.0/conn_P_Surround, "sigma": 0.1}}})

    Cones_Horizontal_H2_dict_S = Cones_Horizontal_dict.copy()
    Cones_Horizontal_H2_dict_S.update({
    "weights": {"gaussian": {"p_center": 2.0/conn_P_Surround, "sigma": 0.1}}})

    Cones_Horizontal_H2_dict_LM = Cones_Horizontal_dict.copy()
    Cones_Horizontal_H2_dict_LM.update({
    "weights": {"gaussian": {"p_center": 1.0/conn_P_Surround, "sigma": 0.1}}})

    # Horizontal cell type H1: receives input from L and M cones (but not S cones)
    [allconns.append(['L_cones_ionotropic','H1_Horizontal_cells',Cones_Horizontal_H1_dict])]
    [allconns.append(['M_cones_ionotropic','H1_Horizontal_cells',Cones_Horizontal_H1_dict])]

    # Horizontal cell type H2: receives input from L, M cones and S cones.
    [allconns.append(['L_cones_ionotropic','H2_Horizontal_cells',Cones_Horizontal_H2_dict_LM])]
    [allconns.append(['M_cones_ionotropic','H2_Horizontal_cells',Cones_Horizontal_H2_dict_LM])]
    [allconns.append(['S_cones_ionotropic','H2_Horizontal_cells',Cones_Horizontal_H2_dict_S])]


    # ----- Horizontal cells -> Bipolar cells ----- #

    # Additional delay of 5-15 ms
    Horizontal_Bipolar_dict = P_Surround_dict.copy()
    Horizontal_Bipolar_dict.update({
    "delays" : {"normal": {"mean": 5.0, "std": 0.25, "min": params['resolution']}},
    "sources": {"model": "retina_parvo_horizontal_cell"}
    })

    # ON bipolar cells receive excitatory synapses from horizontal cells
    Horizontal_H1_Bipolar_L_ON_dict = Horizontal_Bipolar_dict.copy()
    Horizontal_H1_Bipolar_L_ON_dict.update({"weights": {"gaussian": {"p_center": 3.5/conn_P_Surround, "sigma": 0.1}},
    "targets": {"model": "retina_parvo_ON_bipolar_cell"} })
    Horizontal_H1_Bipolar_M_ON_dict = Horizontal_Bipolar_dict.copy()
    Horizontal_H1_Bipolar_M_ON_dict.update({"weights": {"gaussian": {"p_center": 3.5/conn_P_Surround, "sigma": 0.1}},
    "targets": {"model": "retina_parvo_ON_bipolar_cell"} })
    Horizontal_H2_Bipolar_S_ON_dict = Horizontal_Bipolar_dict.copy()
    Horizontal_H2_Bipolar_S_ON_dict.update({"weights": {"gaussian": {"p_center": 3.5/conn_P_Surround, "sigma": 0.1}},
    "targets": {"model": "retina_parvo_ON_bipolar_cell"} })

    # OFF bipolar cells receive inhibitory synapses from horizontal cells
    Horizontal_H1_Bipolar_L_OFF_dict = Horizontal_Bipolar_dict.copy()
    Horizontal_H1_Bipolar_L_OFF_dict.update({"weights": {"gaussian": {"p_center": -5.0/conn_P_Surround, "sigma": 0.1}},
    "targets": {"model": "retina_parvo_OFF_bipolar_cell"}  })
    Horizontal_H1_Bipolar_M_OFF_dict = Horizontal_Bipolar_dict.copy()
    Horizontal_H1_Bipolar_M_OFF_dict.update({"weights": {"gaussian": {"p_center": -5.0/conn_P_Surround, "sigma": 0.1}},
    "targets": {"model": "retina_parvo_OFF_bipolar_cell"} })
    Horizontal_H1_Diffuse_bipolar_dict = Horizontal_Bipolar_dict.copy()
    Horizontal_H1_Diffuse_bipolar_dict.update({"weights": {"gaussian": {"p_center": -5.0/conn_P_Surround, "sigma": 0.1}},
    "targets": {"model": "retina_parvo_OFF_bipolar_cell"} })

    [allconns.append(['H1_Horizontal_cells','Midget_bipolar_cells_L_ON',Horizontal_H1_Bipolar_L_ON_dict])]
    [allconns.append(['H1_Horizontal_cells','Midget_bipolar_cells_L_OFF',Horizontal_H1_Bipolar_L_OFF_dict])]
    [allconns.append(['H1_Horizontal_cells','Midget_bipolar_cells_M_ON',Horizontal_H1_Bipolar_M_ON_dict])]
    [allconns.append(['H1_Horizontal_cells','Midget_bipolar_cells_M_OFF',Horizontal_H1_Bipolar_M_OFF_dict])]
    [allconns.append(['H2_Horizontal_cells','S_cone_bipolar_cells_S_ON',Horizontal_H2_Bipolar_S_ON_dict])]
    [allconns.append(['H1_Horizontal_cells','Diffuse_bipolar_cells_S_ON',Horizontal_H1_Diffuse_bipolar_dict])]


    # ----- ON Bipolar cells -> AII amacrine cells (gap-junction) ----- #

    Bipolar_Amacrine_dict = P_Center_dict.copy()
    Bipolar_Amacrine_dict.update({
    "sources": {"model": "retina_parvo_ON_bipolar_cell"},
    "targets": {"model": "retina_parvo_amacrine_cell"},
    "weights": 1.0/targets_P_Center # divisive factor of g_gap
    })

    [allconns.append(['Midget_bipolar_cells_L_ON','AII_amacrine_cells',Bipolar_Amacrine_dict])]
    [allconns.append(['Midget_bipolar_cells_M_ON','AII_amacrine_cells',Bipolar_Amacrine_dict])]

    # ----- AII amacrine cells -> OFF Bipolar cells ----- #

    Amacrine_Bipolar_dict = P_Center_dict.copy()
    Amacrine_Bipolar_dict.update({
    "sources": {"model": "retina_parvo_amacrine_cell"},
    "targets": {"model": "retina_parvo_OFF_bipolar_cell"},
    "weights": {"gaussian": {"p_center": -1.0/conn_P_Center, "sigma": 0.03}}
    })

    [allconns.append(['AII_amacrine_cells','Midget_bipolar_cells_L_OFF',Amacrine_Bipolar_dict])]
    [allconns.append(['AII_amacrine_cells','Midget_bipolar_cells_M_OFF',Amacrine_Bipolar_dict])]

    # ----- AII amacrine cells -> OFF Ganglion cells ----- #

    Amacrine_Ganglion_dict = P_Center_dict.copy()
    Amacrine_Ganglion_dict.update({
    "sources": {"model": "retina_parvo_amacrine_cell"},
    "targets": {"model": "retina_parvo_ganglion_cell"},
    "weights": {"gaussian": {"p_center": -1.0/conn_P_Center, "sigma": 0.03}}
    })

    [allconns.append(['AII_amacrine_cells','Midget_ganglion_cells_L_OFF',Amacrine_Bipolar_dict])]
    [allconns.append(['AII_amacrine_cells','Midget_ganglion_cells_M_OFF',Amacrine_Bipolar_dict])]


    # ----- Bipolar cells -> Ganglion cells ----- #

    Bipolar_Ganglion_dict = P_Center_dict.copy()
    Bipolar_Ganglion_dict.update({
    "targets": {"model": "retina_parvo_ganglion_cell"}
    })

    L_ON_Bipolar_L_ON_Ganglion_dict = Bipolar_Ganglion_dict.copy()
    L_ON_Bipolar_L_ON_Ganglion_dict.update({"weights": {"gaussian": {"p_center": 20.0/conn_P_Center, "sigma": 0.03}},
    "sources": {"model": "retina_parvo_ON_bipolar_cell"}})

    L_OFF_Bipolar_L_OFF_Ganglion_dict = Bipolar_Ganglion_dict.copy()
    L_OFF_Bipolar_L_OFF_Ganglion_dict.update({"weights": {"gaussian": {"p_center":20.0/conn_P_Center, "sigma": 0.03}},
    "sources": {"model": "retina_parvo_OFF_bipolar_cell"}})

    M_ON_Bipolar_M_ON_Ganglion_dict = Bipolar_Ganglion_dict.copy()
    M_ON_Bipolar_M_ON_Ganglion_dict.update({"weights": {"gaussian": {"p_center": 20.0/conn_P_Center, "sigma": 0.03}},
    "sources": {"model": "retina_parvo_ON_bipolar_cell"}})

    M_OFF_Bipolar_M_OFF_Ganglion_dict = Bipolar_Ganglion_dict.copy()
    M_OFF_Bipolar_M_OFF_Ganglion_dict.update({"weights": {"gaussian": {"p_center": 20.0/conn_P_Center, "sigma": 0.03}},
    "sources": {"model": "retina_parvo_OFF_bipolar_cell"}})

    Diffuse_Bipolar_Small_bistratified_Ganglion_dict = Bipolar_Ganglion_dict.copy()
    Diffuse_Bipolar_Small_bistratified_Ganglion_dict.update({"weights": {"gaussian": {"p_center": 10.0/conn_P_Center, "sigma": 0.03}},
    "sources": {"model": "retina_parvo_OFF_bipolar_cell"}})

    S_cone_bipolar_Small_bistratified_Ganglion_dict = Bipolar_Ganglion_dict.copy()
    S_cone_bipolar_Small_bistratified_Ganglion_dict.update({"weights": {"gaussian": {"p_center": 10.0/conn_P_Center, "sigma": 0.03}},
    "sources": {"model": "retina_parvo_ON_bipolar_cell"}})

    [allconns.append(['Midget_bipolar_cells_L_ON','Midget_ganglion_cells_L_ON',L_ON_Bipolar_L_ON_Ganglion_dict])]
    [allconns.append(['Midget_bipolar_cells_L_OFF','Midget_ganglion_cells_L_OFF',L_OFF_Bipolar_L_OFF_Ganglion_dict])]
    [allconns.append(['Midget_bipolar_cells_M_ON','Midget_ganglion_cells_M_ON',M_ON_Bipolar_M_ON_Ganglion_dict])]
    [allconns.append(['Midget_bipolar_cells_M_OFF','Midget_ganglion_cells_M_OFF',M_OFF_Bipolar_M_OFF_Ganglion_dict])]
    [allconns.append(['Diffuse_bipolar_cells_S_ON','Small_bistratified_ganglion_cells_S_ON',
    Diffuse_Bipolar_Small_bistratified_Ganglion_dict])]
    [allconns.append(['S_cone_bipolar_cells_S_ON','Small_bistratified_ganglion_cells_S_ON',
    S_cone_bipolar_Small_bistratified_Ganglion_dict])]


    # ----- Noise generators -> Ganglion cells ----- #

#    Noise_Ganglion_dict = P_Center_dict.copy()
#    Noise_Ganglion_dict.update({
#    "weights": {"gaussian": {"p_center": 1.0/conn_P_Center, "sigma": 0.03}},
#    "sources": {"model": "retina_noise"},
#    "targets": {"model": "retina_parvo_ganglion_cell"}
#    })

    Noise_Ganglion_dict = {
    "connection_type":"convergent",
    "mask": {"circular": {"radius": 0.001}}, # one-to-one connection
    "kernel": 1.0,
    "delays" : params['resolution'],
    "synapse_model": "syn",
    "weights": 1.0,
    "sources": {"model": "retina_noise"},
    "targets": {"model": "retina_parvo_ganglion_cell"},
    "allow_autapses":False,
    "allow_multapses":False
    }

    [allconns.append(['Noise_generators','Midget_ganglion_cells_L_ON',Noise_Ganglion_dict])]
    [allconns.append(['Noise_generators','Midget_ganglion_cells_L_OFF',Noise_Ganglion_dict])]
    [allconns.append(['Noise_generators','Midget_ganglion_cells_M_ON',Noise_Ganglion_dict])]
    [allconns.append(['Noise_generators','Midget_ganglion_cells_M_OFF',Noise_Ganglion_dict])]
    [allconns.append(['Noise_generators','Small_bistratified_ganglion_cells_S_ON',
    Noise_Ganglion_dict])]


    return allconns
