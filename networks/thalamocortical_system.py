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
# Name: thalamocortical_system
#
# Description:
#
# This script defines the network model of the thalamocortical system. It
# specifies its neuron models, two-dimensional neuronal layers and connections.
#
# We made certain assumptions about the model:
#
# *** LGN ***
#
# 1) In the cat, the receptive-field center of relay cells is driven by excitation of
# ganglion cells of the same sign (ON or OFF) and the surround apparently emerges
# through inhibition of interneurons that receive input from ganglion cells of the
# opposite sign [3]
#
# 2) In primate, there are about as many relay cells as ganglion cells and there is
# nearly a one-to-one anatomical mapping from retina to relay cells [3]
#
# 3) In the monkey LGN, the four dorsal layers (layers 3–6) receive input primarily
# from midget ganglion cells, and contain cells with smaller, colour-opponent,
# poorly contrast sensitive receptive ﬁelds [4]
#
# 4) A number of studies have revealed that almost all neurons in laminae 5 and 6 are
# ON-center cells, whereas layers 3 and 4 contain mostly OFF-center neurons [5,6]
#
# 5) Parvocellular interneurons have electrophysiological properties similar to the
# relay neurons in their laminae. Although there are some recordings of interneurons
# in the parvocellular laminae having center responses that are of the opposite sign
# of the relay cells around them, evidence suggests that same-sign inhibition of relay
# cells is rather the general trend in the primate [5,6]
#
# 6) GABAergic neurons of the reticular nucleus are not included in the model because
# the neural density in the reticular thalamic nucleus is relatively low enough in the
# primate thalamus [7] to consider that the reticular nucleus plays a key role in the
# visual processing
#
# 7) I assume only axonal inhibition by interneurons: "At present, unfortunately,
# there remains no clear-cut empirical evidence about the physiological role of
# the triads in vision" [3]. Also, "In macaque monkeys, trias appear
# to be common in the magnocellular layers and much rarer in the parvocellular
# layers." [8]
#
# *** Thalamocortical connections ***
#
# 1) The feedforward input of cortical-cell receptive fields is built upon rectangular
# masks. This type of mask facilitated the fitting of model parameters to the type of
# measurements used to estimate the receptive-field size of cortical cells [9,10]
#
# 2) Cells with oriented RFs (color-luminance and luminance-preferring cells) have RFs
# with odd-symmetry (in the model: 2 subregions, ON and OFF). Odd symmetry or mixed
# symmetry fits better with the role of double-opponent cells’ responses to color edges [11].
# Moreover, neurons that are well tuned in orientation and spatial frequency tend to have
# odd-symmetric receptive fields [12]
#
# 3) The postsynaptic spiny stellate dendritic arbor in 4C-beta is 200 um in diameter,
# matching the diameter of individual parvocellular thalamic axon arbors [13]. The total diameter
# of the field of terminations of parvocellular axons overlapping at least half the dendritic
# field of single spiny stellate neurons is therefore about 400 um. Assuming a cortical
# magnification factor of 4 mm/deg, 400 um would correspond to 0.1 degree of visual angle.
# Therefore we restricted the width and length of the mask and the separation distance D to an
# approximate value of 0.1 degrees.
#
# 4) The separation between subregions, D, was set to 0.1 degrees. The distribution of
# receptive-field length/width ratios is seen to be nearly invariant with eccentricity in foveal
# striate cortex and the average value to be about 2 [9]. Accordingly, we set the width of the
# rectangular mask to 0.052 degrees and the length to 0.12 degrees
#
# *** V1 layer 4C-Beta ***
#
# 1) Three-quarters of cortical cells are excitatory cells and the rest are inhibitory
# cells
#
# 2) Parameter values of intracortical connections: for connections from excitatory cells, axonal
# arborizations originating from spiny stellate cells are confined to within 2 dendritic arbor radii
# of the cell body (about 200 um, 0.05 degrees). Inhibitory connections are assigned a value of 0.025
# degrees (100 um)
#
# 3) Probability of connection among adjacent neurons does not depend strongly on the interneuron
# distance [14].
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
# [3] J. A. Hirsch, X. Wang, F. T. Sommer and L. M. Martinez, How inhibitory circuits
# in the thalamus serve vision, Annual review of neuroscience 38 (2015) 309–329.
#
# [4] Callaway, E. M. (2005). Structure and function of parallel pathways in the primate
# early visual system. The Journal of physiology, 566(1), 13-19.
#
# [5] J. R. Wilson, Synaptic organization of individual neurons in the macaque lateral
# geniculate nucleus, Journal of Neuroscience 9(8) (1989) 2931–2953.
#
# [6] C. R. Michael, Retinal afferent arborization patterns, dendritic field orientations,
# and the segregation of function in the lateral geniculate nucleus of the monkey,
# Proceedings of the National Academy of Sciences 85(13) (1988) 4914–4918.
#
# [7] P. Arcelli, C. Frassoni, M. Regondi, S. De Biasi and R. Spreafico, Gabaergic neurons
# in mammalian thalamus: a marker of thalamic complexity?, Brain research bulletin 42(1)
# (1997) 27–37.
#
# [8] V. A. Casagrande, D. W. Royal and G. Sáry, Geniculate nucleus (lgn), The primate
# visual system: a comparative approach (2005) p. 191.
#
# [9] B. Dow, A. Snyder, R. Vautin and R. Bauer, Magnification factor and receptive field
# size in foveal striate cortex of the monkey, Experimental brain research 44(2) (1981) 213–228.
#
# [10] D. C. Van Essen, W. T. Newsome and J. H. Maunsell, The visual field representation in
# striate cortex of the macaque monkey: asymmetries, anisotropies, and individual variability,
# Vision research 24(5) (1984) 429–448.
#
# [11] R. Shapley and M. J. Hawken, Color in the cortex: single-and double-opponent cells,
# Vision research 51(7) (2011) 701–717.
#
# [12] Ringach, D. L. (2002). Spatial structure and symmetry of simple-cell receptive fields in
# macaque primary visual cortex. Journal of neurophysiology, 88(1), 455-463.
#
# [13] J. S. Lund, A. Angelucci and P. C. Bressloff, Anatomical substrates for functional columns
# in macaque monkey primary visual cortex, Cerebral Cortex 13(1) (2003) 15–24.
#
# [14] Song, S., Sjöström, P. J., Reigl, M., Nelson, S., & Chklovskii, D. B. (2005). Highly
# nonrandom features of synaptic connectivity in local cortical circuits. PLoS biology, 3(3), e68.
#
#
# Author: Martinez-Cañada, P. (pablomc@ugr.es)



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

## Population size of each group of cortical cells ##
# N = maximum number of cortical neurons per row (params['N_cortex'])
#
# Total number of excitatory cells = (N^2/4) * 8 + N^2 * 4 + (N^2/4) * 2 = N^2 * (13/2)
# Percentage of Color-luminance: 4/13 = 30 %
# Percentage of Luminance-preferring: 8/13 = 62 %
# Percentage of Color-preferring: 1/13 = 8 %
#
# Total number of inhibitory cells = (N^2/16) * 8 + (N^2/4) * 4 + (N^2/16) * 2 = N^2 * (7/4)
# Total number of cortical cells = N^2 * (13/2 + 7/4) = N^2 * (33/4)
# Percentage of excitatory cells = 26/33 (79 %)
# Percentage of inhibitory cells = 7/33 (21 %)

def getSizes(params):
    rows_color_luminance_exc = params['N_cortex']//2
    rows_luminance_preferring_exc = params['N_cortex']
    rows_color_preferring_exc = params['N_cortex']//2

    rows_color_luminance_inh = params['N_cortex']//4
    rows_luminance_preferring_inh = params['N_cortex']//2
    rows_color_preferring_inh = params['N_cortex']//4

    return [rows_color_luminance_exc,
            rows_luminance_preferring_exc,
            rows_color_preferring_exc,
            rows_color_luminance_inh,
            rows_luminance_preferring_inh,
            rows_color_preferring_inh]

# Parameters of neuron models for each layer
def get_Models():

    # Retinal ganglion cells (Spikes are given to the spike generator as an array)
    Retinal_ganglion_cell  = 'spike_generator'
    Retinal_ganglion_cell_params = {
    "origin": 0.0,
    "start": 0.0
    }

    # LGN relay cells
    LGN_relay_cell  = 'iaf_cond_alpha'
    LGN_relay_cell_params = {
    "C_m": 100.0,
    "g_L": 10.0,
    "E_L": -60.0,
    "V_th": -55.0,
    "V_reset": -60.0,
    "t_ref":  2.0,
    "E_ex": 0.0, # AMPA, from Hill-Tononi 2005
    "E_in": -80.0, # GABA-A of thalamocortical cells, from Hill-Tononi 2005
    "tau_syn_ex": 1.0, # it approximates Hill-Tononi's diff. of exp. response, also Casti 2008
    "tau_syn_in": 3.0 # it approximates Hill-Tononi's diff. of exp. response
    }

    # LGN interneurons
    LGN_interneuron  = 'iaf_cond_alpha'
    LGN_interneuron_params = {
    "C_m": 100.0,
    "g_L": 10.0,
    "E_L": -60.0,
    "V_th": -55.0,
    "V_reset": -60.0,
    "t_ref":  2.0,
    "E_ex": 0.0,
    "E_in": -80.0,
    "tau_syn_ex": 1.0,
    "tau_syn_in": 3.0
    }

    # Cortical excitatory cells
    Cortex_excitatory_cell  = 'iaf_cond_alpha'
    Cortex_excitatory_cell_params = {
    "C_m": 100.0,
    "g_L": 10.0,
    "E_L": -60.0,
    "V_th": -55.0,
    "V_reset": -60.0,
    "t_ref":  2.0,
    "E_ex": 0.0,
    "E_in": -70.0, # GABA-A of cortical cells, from Hill-Tononi 2005
    "tau_syn_ex": 1.0,
    "tau_syn_in": 3.0
    }

    # Cortical inhibitory cells
    Cortex_inhibitory_cell  = 'iaf_cond_alpha'
    Cortex_inhibitory_cell_params = {
    "C_m": 100.0,
    "g_L": 10.0,
    "E_L": -60.0,
    "V_th": -55.0,
    "V_reset": -60.0,
    "t_ref":  2.0,
    "E_ex": 0.0,
    "E_in": -70.0,
    "tau_syn_ex": 1.0,
    "tau_syn_in": 3.0
    }

    # Gaussian noise current
    thalamo_noise = 'noise_generator'
    thalamo_noise_params = {
    'mean': 0.0, # pA
    'std': 1.0 # pA
    }

    models = [(Retinal_ganglion_cell,'Retinal_ganglion_cell',
            Retinal_ganglion_cell_params),
            (LGN_relay_cell,'LGN_relay_cell', LGN_relay_cell_params),
            (LGN_interneuron,'LGN_interneuron', LGN_interneuron_params),
            (Cortex_excitatory_cell,'Cortex_excitatory_cell',
            Cortex_excitatory_cell_params),
            (Cortex_inhibitory_cell,'Cortex_inhibitory_cell',
            Cortex_inhibitory_cell_params),
             (thalamo_noise,'thalamo_noise', thalamo_noise_params)]

    return models

# Definition of neuronal layers
def get_Layers(params):

    layerProps_LGN = {
    'rows'     : params['N_LGN'],
    'columns'  : params['N_LGN'],
    'extent'   : [params['visSize'], params['visSize']],
    'edge_wrap': True
    }

    layerProps_cortex = {
    'rows'     : params['N_cortex'],
    'columns'  : params['N_cortex'],
    'extent'   : [params['visSize'], params['visSize']],
    'edge_wrap': True
    }

    [rows_color_luminance_exc,
    rows_luminance_preferring_exc,
    rows_color_preferring_exc,
    rows_color_luminance_inh,
    rows_luminance_preferring_inh,
    rows_color_preferring_inh] = getSizes(params)

    # Create layer dictionaries
    layers = [
            ## LGN
            ('Midget_ganglion_cells_L_ON',updateDicts(layerProps_LGN, {'elements': 'Retinal_ganglion_cell'})),
            ('Midget_ganglion_cells_L_OFF',updateDicts(layerProps_LGN, {'elements': 'Retinal_ganglion_cell'})),
            ('Midget_ganglion_cells_M_ON',updateDicts(layerProps_LGN, {'elements': 'Retinal_ganglion_cell'})),
            ('Midget_ganglion_cells_M_OFF',updateDicts(layerProps_LGN, {'elements': 'Retinal_ganglion_cell'})),
            ('Parvo_LGN_relay_cell_L_ON',updateDicts(layerProps_LGN, {'elements': 'LGN_relay_cell'})),
            ('Parvo_LGN_relay_cell_L_OFF',updateDicts(layerProps_LGN, {'elements': 'LGN_relay_cell'})),
            ('Parvo_LGN_relay_cell_M_ON',updateDicts(layerProps_LGN, {'elements': 'LGN_relay_cell'})),
            ('Parvo_LGN_relay_cell_M_OFF',updateDicts(layerProps_LGN, {'elements': 'LGN_relay_cell'})),
            ('Parvo_LGN_interneuron_ON',updateDicts(layerProps_LGN, {'elements': 'LGN_interneuron'})),
            ('Parvo_LGN_interneuron_OFF',updateDicts(layerProps_LGN, {'elements': 'LGN_interneuron'})),

            ## Cortex
            ## Excitatory
            ('Color_Luminance_L_ON_L_OFF_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_luminance_exc,'columns'  : rows_color_luminance_exc})),
            ('Color_Luminance_L_ON_L_OFF_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_luminance_exc,'columns'  : rows_color_luminance_exc})),
            ('Color_Luminance_L_OFF_L_ON_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_luminance_exc,'columns'  : rows_color_luminance_exc})),
            ('Color_Luminance_L_OFF_L_ON_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_luminance_exc,'columns'  : rows_color_luminance_exc})),
            ('Color_Luminance_M_ON_M_OFF_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_luminance_exc,'columns'  : rows_color_luminance_exc})),
            ('Color_Luminance_M_ON_M_OFF_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_luminance_exc,'columns'  : rows_color_luminance_exc})),
            ('Color_Luminance_M_OFF_M_ON_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_luminance_exc,'columns'  : rows_color_luminance_exc})),
            ('Color_Luminance_M_OFF_M_ON_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_luminance_exc,'columns'  : rows_color_luminance_exc})),

            ('Luminance_preferring_ON_OFF_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_luminance_preferring_exc,'columns'  : rows_luminance_preferring_exc})),
            ('Luminance_preferring_ON_OFF_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_luminance_preferring_exc,'columns'  : rows_luminance_preferring_exc})),
            ('Luminance_preferring_OFF_ON_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_luminance_preferring_exc,'columns'  : rows_luminance_preferring_exc})),
            ('Luminance_preferring_OFF_ON_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_luminance_preferring_exc,'columns'  : rows_luminance_preferring_exc})),

            ('Color_preferring_L_ON_M_OFF',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_preferring_exc,'columns'  : rows_color_preferring_exc})),
            ('Color_preferring_M_ON_L_OFF',updateDicts(layerProps_cortex, {'elements': 'Cortex_excitatory_cell',
            'rows': rows_color_preferring_exc,'columns'  : rows_color_preferring_exc})),

            ## Inhibitory
            ('Color_Luminance_inh_L_ON_L_OFF_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_luminance_inh,'columns'  : rows_color_luminance_inh})),
            ('Color_Luminance_inh_L_ON_L_OFF_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_luminance_inh,'columns'  : rows_color_luminance_inh})),
            ('Color_Luminance_inh_L_OFF_L_ON_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_luminance_inh,'columns'  : rows_color_luminance_inh})),
            ('Color_Luminance_inh_L_OFF_L_ON_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_luminance_inh,'columns'  : rows_color_luminance_inh})),
            ('Color_Luminance_inh_M_ON_M_OFF_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_luminance_inh,'columns'  : rows_color_luminance_inh})),
            ('Color_Luminance_inh_M_ON_M_OFF_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_luminance_inh,'columns'  : rows_color_luminance_inh})),
            ('Color_Luminance_inh_M_OFF_M_ON_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_luminance_inh,'columns'  : rows_color_luminance_inh})),
            ('Color_Luminance_inh_M_OFF_M_ON_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_luminance_inh,'columns'  : rows_color_luminance_inh})),

            ('Luminance_preferring_inh_ON_OFF_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_luminance_preferring_inh,'columns'  : rows_luminance_preferring_inh})),
            ('Luminance_preferring_inh_ON_OFF_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_luminance_preferring_inh,'columns'  : rows_luminance_preferring_inh})),
            ('Luminance_preferring_inh_OFF_ON_vertical',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_luminance_preferring_inh,'columns'  : rows_luminance_preferring_inh})),
            ('Luminance_preferring_inh_OFF_ON_horizontal',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_luminance_preferring_inh,'columns'  : rows_luminance_preferring_inh})),

            ('Color_preferring_inh_L_ON_M_OFF',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_preferring_inh,'columns'  : rows_color_preferring_inh})),
            ('Color_preferring_inh_M_ON_L_OFF',updateDicts(layerProps_cortex, {'elements': 'Cortex_inhibitory_cell',
            'rows': rows_color_preferring_inh,'columns'  : rows_color_preferring_inh})),

            ## Noise generators
            ('Noise_generators_LGN',updateDicts(layerProps_LGN, {'elements': 'thalamo_noise'})),

            ('Noise_generators_Color_Luminance',updateDicts(layerProps_cortex, {'elements': 'thalamo_noise',
            'rows': rows_color_luminance_exc,'columns'  : rows_color_luminance_exc})),
            ('Noise_generators_Luminance_preferring',updateDicts(layerProps_cortex, {'elements': 'thalamo_noise',
            'rows': rows_luminance_preferring_exc,'columns'  : rows_luminance_preferring_exc})),
            ('Noise_generators_Color_preferring',updateDicts(layerProps_cortex, {'elements': 'thalamo_noise',
            'rows': rows_color_preferring_exc,'columns'  : rows_color_preferring_exc})),

            ('Noise_generators_Color_Luminance_inh',updateDicts(layerProps_cortex, {'elements': 'thalamo_noise',
            'rows': rows_color_luminance_inh,'columns'  : rows_color_luminance_inh})),
            ('Noise_generators_Luminance_preferring_inh',updateDicts(layerProps_cortex, {'elements': 'thalamo_noise',
            'rows': rows_luminance_preferring_inh,'columns'  : rows_luminance_preferring_inh})),
            ('Noise_generators_Color_preferring_inh',updateDicts(layerProps_cortex, {'elements': 'thalamo_noise',
            'rows': rows_color_preferring_inh,'columns'  : rows_color_preferring_inh}))
    ]

    return layers

# Weights are scaled with the size of the network so that the sum of the weights
# of all incoming synapses is always equal to a constant value
def get_Relative_Weight(params,mask_params,number_rows_source,number_rows_target):

    # Create a fictional network and count the number of target connections
    layerProps = {
    'rows'     : number_rows_source,
    'columns'  : number_rows_source,
    'extent'   : [params['visSize'], params['visSize']],
    'edge_wrap': True,
    'elements': 'iaf_cond_exp'
    }

    l_source = tp.CreateLayer(layerProps)

    layerProps = {
    'rows'     : number_rows_target,
    'columns'  : number_rows_target,
    'extent'   : [params['visSize'], params['visSize']],
    'edge_wrap': True,
    'elements': 'iaf_cond_exp'
    }

    l_target = tp.CreateLayer(layerProps)

    # Circular mask
    if (isinstance(mask_params, list) == False):
        dict = {
        "connection_type":"divergent",
        "mask": {"circular": {"radius": mask_params}},
        "kernel": 1.0,
        "weights": {"gaussian": {"p_center": 1.0, "sigma": mask_params/3.0}}
        }
    # Rectangular mask
    else:
        dict = {
        "connection_type":"convergent",
        "mask": {'rectangular': {'lower_left':[mask_params[0],mask_params[1]],
                                'upper_right':[mask_params[2],mask_params[3]]}},
        "kernel": 1.0,
        "weights": 1.0
        }

    tp.ConnectLayers(l_source,l_target,dict)
    ctr = tp.FindCenterElement(l_target)
    conn = nest.GetConnections(target = [ctr[0]])

#    print ("Number of sources = ",len(conn))

    st = nest.GetStatus(conn)

    w = 0.0
    for n in np.arange(len(st)):
        w += st[n]['weight']

#    print ("Total weight = ",w)

    if w == 0.0:
        print ("-- Warning: found w = 0.0. Changed to 1.0. --")
        w = 1.0

    return w,len(conn)


# Make dictionary of circular connections
def make_dict_circular(params,mask_radius,center_weight,sigma,mean_delay,std_delay,
sources,targets,number_rows_source,number_rows_target,kernel=1.0):

    # to scale weights with the size of the network
    [total_weight,t] = get_Relative_Weight(params,mask_radius,number_rows_source,number_rows_target)

    dict = {
    "connection_type":"divergent",
    "mask": {"circular": {"radius": mask_radius}},
    "kernel": kernel,
    "delays" : {"normal": {"mean": mean_delay, "std": std_delay, "min": params['resolution']}},
    "synapse_model": "syn",
    "weights": {"gaussian": {"p_center": center_weight/total_weight, "sigma": sigma}},
    "sources": {"model": sources},
    "targets": {"model": targets},
    "allow_autapses":False,
    "allow_multapses":False
    }

    return dict

# Make dictionary of rectangular connections
def make_dict_rectangular(params,mask_points,center_weight,mean_delay,std_delay,
sources,targets,number_rows_source,number_rows_target):

    # to scale weights with the size of the network
    [total_weight,t] = get_Relative_Weight(params,mask_points,number_rows_source,number_rows_target)

    dict = {
    "connection_type":"convergent",
    "mask": {'rectangular': {'lower_left':[mask_points[0],mask_points[1]],
                            'upper_right':[mask_points[2],mask_points[3]]}},
    "kernel": 1.0,
    "delays" : {"normal": {"mean": mean_delay, "std": std_delay, "min": params['resolution']}},
    "synapse_model": "syn",
    "weights": center_weight/total_weight,
    "sources": {"model": sources},
    "targets": {"model": targets},
    "allow_autapses":False,
    "allow_multapses":False
    }

    return dict

# Cortical intracortical connections: all-to-all connections
def make_cortex_horizontal_connections(allconns,exc_model_IDs,inh_model_IDs,
Exc_Exc_dict,Exc_Inh_dict,Inh_Inh_dict,Inh_Exc_dict):

    # Exc -> Exc
    for source_model in exc_model_IDs:
        for target_model in exc_model_IDs:
            [allconns.append([source_model,target_model,Exc_Exc_dict])]

    # Exc -> Inh
    for source_model in exc_model_IDs:
        for target_model in inh_model_IDs:
            [allconns.append([source_model,target_model,Exc_Inh_dict])]

    # Inh -> Inh
    for source_model in inh_model_IDs:
        for target_model in inh_model_IDs:
            [allconns.append([source_model,target_model,Inh_Inh_dict])]

    # Inh -> Exc
    for source_model in inh_model_IDs:
        for target_model in exc_model_IDs:
            [allconns.append([source_model,target_model,Inh_Exc_dict])]

    return allconns


# Create connections between layers
def get_Connections(params):

    # Synapse model
    nest.CopyModel('static_synapse','syn')

    # Build complete list of connections
    allconns = []

    # Cortical layer sizes
    [rows_color_luminance_exc,
    rows_luminance_preferring_exc,
    rows_color_preferring_exc,
    rows_color_luminance_inh,
    rows_luminance_preferring_inh,
    rows_color_preferring_inh] = getSizes(params)

    ###################
    ####### LGN #######
    ###################

    # ----- Retinal ganglion cells -> Relay cells ----- #

    center_weight = 4.0 # nS
    sigma = 0.03 # degrees
    delay = 3.0 # ms (Somers 1995)
    std_delay = 1.0 # ms (Somers 1995)
    mask_radius = sigma * 3.0 # degrees

    Retina_Relay_cell_dict = make_dict_circular(params,mask_radius,center_weight,sigma,delay,
    std_delay,"Retinal_ganglion_cell","LGN_relay_cell",params['N_LGN'],params['N_LGN'])

    # Add connection
    [allconns.append(['Midget_ganglion_cells_L_ON','Parvo_LGN_relay_cell_L_ON',Retina_Relay_cell_dict])]
    [allconns.append(['Midget_ganglion_cells_L_OFF','Parvo_LGN_relay_cell_L_OFF',Retina_Relay_cell_dict])]
    [allconns.append(['Midget_ganglion_cells_M_ON','Parvo_LGN_relay_cell_M_ON',Retina_Relay_cell_dict])]
    [allconns.append(['Midget_ganglion_cells_M_OFF','Parvo_LGN_relay_cell_M_OFF',Retina_Relay_cell_dict])]

    # ----- Retinal ganglion cells -> Interneurons ----- #

    center_weight = 2.0
    sigma = 0.06 # (2 x dend. size of parvo cells)
    delay = 3.0
    std_delay = 1.0
    mask_radius = sigma * 3.0

    Retina_Interneuron_dict = make_dict_circular(params,mask_radius,center_weight,sigma,delay,
    std_delay,"Retinal_ganglion_cell","LGN_interneuron",params['N_LGN'],params['N_LGN'])

    # Add connection
    [allconns.append(['Midget_ganglion_cells_L_ON','Parvo_LGN_interneuron_ON',Retina_Interneuron_dict])]
    [allconns.append(['Midget_ganglion_cells_M_ON','Parvo_LGN_interneuron_ON',Retina_Interneuron_dict])]
    [allconns.append(['Midget_ganglion_cells_L_OFF','Parvo_LGN_interneuron_OFF',Retina_Interneuron_dict])]
    [allconns.append(['Midget_ganglion_cells_M_OFF','Parvo_LGN_interneuron_OFF',Retina_Interneuron_dict])]

    # ----- Interneurons -> Relay cells ----- #
    # ----- Axonal inhibition ----- #

    center_weight = -2.0
    sigma = 0.06
    delay = 2.0 # (Hill 2005)
    std_delay = 0.25 # (Hill 2005)
    mask_radius = sigma * 3.0

    Interneuron_Relay_cell_dict = make_dict_circular(params,mask_radius,center_weight,sigma,delay,
    std_delay,"LGN_interneuron","LGN_relay_cell",params['N_LGN'],params['N_LGN'])

    # Add connection
    [allconns.append(['Parvo_LGN_interneuron_ON','Parvo_LGN_relay_cell_L_ON',Interneuron_Relay_cell_dict])]
    [allconns.append(['Parvo_LGN_interneuron_ON','Parvo_LGN_relay_cell_M_ON',Interneuron_Relay_cell_dict])]
    [allconns.append(['Parvo_LGN_interneuron_OFF','Parvo_LGN_relay_cell_L_OFF',Interneuron_Relay_cell_dict])]
    [allconns.append(['Parvo_LGN_interneuron_OFF','Parvo_LGN_relay_cell_M_OFF',Interneuron_Relay_cell_dict])]

    # ----- Interneurons -> Interneurons ----- #

    center_weight = -2.0
    sigma = 0.06
    delay = 2.0
    std_delay = 0.25
    mask_radius = sigma * 3.0

    Interneuron_Interneuron_dict = make_dict_circular(params,mask_radius,center_weight,sigma,delay,
    std_delay,"LGN_interneuron","LGN_interneuron",params['N_LGN'],params['N_LGN'])

    # Add connection
    [allconns.append(['Parvo_LGN_interneuron_ON','Parvo_LGN_interneuron_ON',Interneuron_Interneuron_dict])]
    [allconns.append(['Parvo_LGN_interneuron_OFF','Parvo_LGN_interneuron_OFF',Interneuron_Interneuron_dict])]

    #############################################
    ## Thalamocortical connections to L4C-beta ##
    #############################################

    ## Excitatory ##

    # ----- Relay cells -> Color-Luminance cells ----- #

    center_weight_ON = 2.5
    center_weight_OFF = 2.5
    delay = 3.0 # (Hill_2005)
    std_delay = 0.25 # (Hill_2005)
    D = 0.1 # Distance between subregions
    mask_points_vertical_ON = [-0.026, -0.06, 0.026, 0.06]
    mask_points_vertical_OFF = [-0.026 + D, -0.06, 0.026 + D, 0.06]
    mask_points_horizontal_ON = [-0.06, -0.026, 0.06, 0.026]
    mask_points_horizontal_OFF = [-0.06, -0.026 + D, 0.06, 0.026 + D]

    Color_Luminance_vertical_ON_dict = make_dict_rectangular(params,
    mask_points_vertical_ON,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_color_luminance_exc)

    Color_Luminance_vertical_OFF_dict = make_dict_rectangular(params,
    mask_points_vertical_OFF,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_color_luminance_exc)

    Color_Luminance_horizontal_ON_dict = make_dict_rectangular(params,
    mask_points_horizontal_ON,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_color_luminance_exc)

    Color_Luminance_horizontal_OFF_dict = make_dict_rectangular(params,
    mask_points_horizontal_OFF,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_color_luminance_exc)

    # Add connections
    # L_ON_L_OFF
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_Luminance_L_ON_L_OFF_vertical',
    Color_Luminance_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_Luminance_L_ON_L_OFF_vertical',
    Color_Luminance_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_Luminance_L_ON_L_OFF_horizontal',
    Color_Luminance_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_Luminance_L_ON_L_OFF_horizontal',
    Color_Luminance_horizontal_OFF_dict])]
    # L_OFF_L_ON
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_Luminance_L_OFF_L_ON_vertical',
    Color_Luminance_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_Luminance_L_OFF_L_ON_vertical',
    Color_Luminance_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_Luminance_L_OFF_L_ON_horizontal',
    Color_Luminance_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_Luminance_L_OFF_L_ON_horizontal',
    Color_Luminance_horizontal_OFF_dict])]
    # M_ON_M_OFF
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_Luminance_M_ON_M_OFF_vertical',
    Color_Luminance_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_Luminance_M_ON_M_OFF_vertical',
    Color_Luminance_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_Luminance_M_ON_M_OFF_horizontal',
    Color_Luminance_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_Luminance_M_ON_M_OFF_horizontal',
    Color_Luminance_horizontal_OFF_dict])]
    # M_OFF_M_ON
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_Luminance_M_OFF_M_ON_vertical',
    Color_Luminance_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_Luminance_M_OFF_M_ON_vertical',
    Color_Luminance_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_Luminance_M_OFF_M_ON_horizontal',
    Color_Luminance_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_Luminance_M_OFF_M_ON_horizontal',
    Color_Luminance_horizontal_OFF_dict])]

#    # ----- Relay cells -> Luminance-preferring cells ----- #

    center_weight_ON = 1.25
    center_weight_OFF = 1.25
    delay = 3.0
    std_delay = 0.25
    D = 0.1
    mask_points_vertical_ON = [-0.026, -0.06, 0.026, 0.06]
    mask_points_vertical_OFF = [-0.026 + D, -0.06, 0.026 + D, 0.06]
    mask_points_horizontal_ON = [-0.06, -0.026, 0.06, 0.026]
    mask_points_horizontal_OFF = [-0.06, -0.026 + D, 0.06, 0.026 + D]

    Luminance_preferring_vertical_ON_dict = make_dict_rectangular(params,
    mask_points_vertical_ON,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_luminance_preferring_exc)

    Luminance_preferring_vertical_OFF_dict = make_dict_rectangular(params,
    mask_points_vertical_OFF,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_luminance_preferring_exc)

    Luminance_preferring_horizontal_ON_dict = make_dict_rectangular(params,
    mask_points_horizontal_ON,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_luminance_preferring_exc)

    Luminance_preferring_horizontal_OFF_dict = make_dict_rectangular(params,
    mask_points_horizontal_OFF,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_luminance_preferring_exc)

    # Add connections

    # ON_OFF
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Luminance_preferring_ON_OFF_vertical',
    Luminance_preferring_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Luminance_preferring_ON_OFF_vertical',
    Luminance_preferring_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Luminance_preferring_ON_OFF_vertical',
    Luminance_preferring_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Luminance_preferring_ON_OFF_vertical',
    Luminance_preferring_vertical_OFF_dict])]

    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Luminance_preferring_ON_OFF_horizontal',
    Luminance_preferring_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Luminance_preferring_ON_OFF_horizontal',
    Luminance_preferring_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Luminance_preferring_ON_OFF_horizontal',
    Luminance_preferring_horizontal_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Luminance_preferring_ON_OFF_horizontal',
    Luminance_preferring_horizontal_OFF_dict])]

    # OFF_ON
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Luminance_preferring_OFF_ON_vertical',
    Luminance_preferring_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Luminance_preferring_OFF_ON_vertical',
    Luminance_preferring_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Luminance_preferring_OFF_ON_vertical',
    Luminance_preferring_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Luminance_preferring_OFF_ON_vertical',
    Luminance_preferring_vertical_OFF_dict])]

    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Luminance_preferring_OFF_ON_horizontal',
    Luminance_preferring_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Luminance_preferring_OFF_ON_horizontal',
    Luminance_preferring_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Luminance_preferring_OFF_ON_horizontal',
    Luminance_preferring_horizontal_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Luminance_preferring_OFF_ON_horizontal',
    Luminance_preferring_horizontal_OFF_dict])]


    # ----- Relay cells -> Color-preferring cells ----- #

    center_weight_ON = 2.5
    center_weight_OFF = 2.5
    delay = 3.0
    std_delay = 0.25

    mask_points = [-0.06,-0.06,0.06,0.06]

    Non_oriented_color_ON_dict = make_dict_rectangular(params,
    mask_points,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_color_preferring_exc)

    Non_oriented_color_OFF_dict = make_dict_rectangular(params,
    mask_points,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_excitatory_cell",params['N_LGN'],rows_color_preferring_exc)

    # Add connection
    # L_ON_M_OFF
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_preferring_L_ON_M_OFF',
    Non_oriented_color_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_preferring_L_ON_M_OFF',
    Non_oriented_color_OFF_dict])]
    # M_ON_L_OFF
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_preferring_M_ON_L_OFF',
    Non_oriented_color_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_preferring_M_ON_L_OFF',
    Non_oriented_color_ON_dict])]

    ## Inhibitory ##

    # ----- Relay cells -> Color-Luminance cells ----- #

    center_weight_ON = 3.0
    center_weight_OFF = 3.0

    Color_Luminance_vertical_ON_dict = make_dict_rectangular(params,
    mask_points_vertical_ON,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_color_luminance_inh)

    Color_Luminance_vertical_OFF_dict = make_dict_rectangular(params,
    mask_points_vertical_OFF,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_color_luminance_inh)

    Color_Luminance_horizontal_ON_dict = make_dict_rectangular(params,
    mask_points_horizontal_ON,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_color_luminance_inh)

    Color_Luminance_horizontal_OFF_dict = make_dict_rectangular(params,
    mask_points_horizontal_OFF,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_color_luminance_inh)

    # L_ON_L_OFF
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_Luminance_inh_L_ON_L_OFF_vertical',
    Color_Luminance_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_Luminance_inh_L_ON_L_OFF_vertical',
    Color_Luminance_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_Luminance_inh_L_ON_L_OFF_horizontal',
    Color_Luminance_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_Luminance_inh_L_ON_L_OFF_horizontal',
    Color_Luminance_horizontal_OFF_dict])]
    # L_OFF_L_ON
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_Luminance_inh_L_OFF_L_ON_vertical',
    Color_Luminance_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_Luminance_inh_L_OFF_L_ON_vertical',
    Color_Luminance_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_Luminance_inh_L_OFF_L_ON_horizontal',
    Color_Luminance_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_Luminance_inh_L_OFF_L_ON_horizontal',
    Color_Luminance_horizontal_OFF_dict])]
    # M_ON_M_OFF
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_Luminance_inh_M_ON_M_OFF_vertical',
    Color_Luminance_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_Luminance_inh_M_ON_M_OFF_vertical',
    Color_Luminance_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_Luminance_inh_M_ON_M_OFF_horizontal',
    Color_Luminance_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_Luminance_inh_M_ON_M_OFF_horizontal',
    Color_Luminance_horizontal_OFF_dict])]
    # M_OFF_M_ON
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_Luminance_inh_M_OFF_M_ON_vertical',
    Color_Luminance_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_Luminance_inh_M_OFF_M_ON_vertical',
    Color_Luminance_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_Luminance_inh_M_OFF_M_ON_horizontal',
    Color_Luminance_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_Luminance_inh_M_OFF_M_ON_horizontal',
    Color_Luminance_horizontal_OFF_dict])]

    # ----- Relay cells -> Luminance-preferring cells ----- #

    center_weight_ON = 1.5
    center_weight_OFF = 1.5

    Luminance_preferring_vertical_ON_dict = make_dict_rectangular(params,
    mask_points_vertical_ON,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_luminance_preferring_inh)

    Luminance_preferring_vertical_OFF_dict = make_dict_rectangular(params,
    mask_points_vertical_OFF,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_luminance_preferring_inh)

    Luminance_preferring_horizontal_ON_dict = make_dict_rectangular(params,
    mask_points_horizontal_ON,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_luminance_preferring_inh)

    Luminance_preferring_horizontal_OFF_dict = make_dict_rectangular(params,
    mask_points_horizontal_OFF,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_luminance_preferring_inh)

    # ON_OFF
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Luminance_preferring_inh_ON_OFF_vertical',
    Luminance_preferring_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Luminance_preferring_inh_ON_OFF_vertical',
    Luminance_preferring_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Luminance_preferring_inh_ON_OFF_vertical',
    Luminance_preferring_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Luminance_preferring_inh_ON_OFF_vertical',
    Luminance_preferring_vertical_OFF_dict])]

    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Luminance_preferring_inh_ON_OFF_horizontal',
    Luminance_preferring_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Luminance_preferring_inh_ON_OFF_horizontal',
    Luminance_preferring_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Luminance_preferring_inh_ON_OFF_horizontal',
    Luminance_preferring_horizontal_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Luminance_preferring_inh_ON_OFF_horizontal',
    Luminance_preferring_horizontal_OFF_dict])]

    # OFF_ON
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Luminance_preferring_inh_OFF_ON_vertical',
    Luminance_preferring_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Luminance_preferring_inh_OFF_ON_vertical',
    Luminance_preferring_vertical_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Luminance_preferring_inh_OFF_ON_vertical',
    Luminance_preferring_vertical_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Luminance_preferring_inh_OFF_ON_vertical',
    Luminance_preferring_vertical_OFF_dict])]

    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Luminance_preferring_inh_OFF_ON_horizontal',
    Luminance_preferring_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Luminance_preferring_inh_OFF_ON_horizontal',
    Luminance_preferring_horizontal_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Luminance_preferring_inh_OFF_ON_horizontal',
    Luminance_preferring_horizontal_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Luminance_preferring_inh_OFF_ON_horizontal',
    Luminance_preferring_horizontal_OFF_dict])]


    # ----- Relay cells -> Color-preferring cells ----- #

    center_weight_ON = 3.0
    center_weight_OFF = 3.0

    Non_oriented_color_ON_dict = make_dict_rectangular(params,
    mask_points,center_weight_ON,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_color_preferring_inh)

    Non_oriented_color_OFF_dict = make_dict_rectangular(params,
    mask_points,center_weight_OFF,delay,
    std_delay,"LGN_relay_cell","Cortex_inhibitory_cell",params['N_LGN'],rows_color_preferring_inh)

    # L_ON_M_OFF
    [allconns.append(['Parvo_LGN_relay_cell_L_ON','Color_preferring_inh_L_ON_M_OFF',
    Non_oriented_color_ON_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_OFF','Color_preferring_inh_L_ON_M_OFF',
    Non_oriented_color_OFF_dict])]
    # M_ON_L_OFF
    [allconns.append(['Parvo_LGN_relay_cell_L_OFF','Color_preferring_inh_M_ON_L_OFF',
    Non_oriented_color_OFF_dict])]
    [allconns.append(['Parvo_LGN_relay_cell_M_ON','Color_preferring_inh_M_ON_L_OFF',
    Non_oriented_color_ON_dict])]

    #############################################
    ### Horizontal connections within L4C-beta ##
    #############################################

#    Exc_Exc_dict = {
#    "connection_type":"convergent",
#    "mask": {"circular": {"radius": 0.15}},
#    "kernel": {"gaussian": {"p_center": 0.2, "sigma": 0.05}},
#    "delays" : {"normal": {"mean": 2.0, "std": 0.25, "min": params['resolution']}},
#    "synapse_model": "syn",
#    "weights": 2.0 * params['visSize'] / params['N_cortex'],
#    "sources": {"model": "Cortex_excitatory_cell"},
#    "targets": {"model": "Cortex_excitatory_cell"},
#    "allow_autapses":False,
#    "allow_multapses":False
#    }

    sigma = 0.05
    mask_radius = sigma * 3.0
    kernel = 1.0

    center_weight = 0.5

    delay = 2.0
    std_delay = 0.25

    Exc_Exc_dict = make_dict_circular(params,mask_radius,center_weight,sigma,delay,
        std_delay,"Cortex_excitatory_cell","Cortex_excitatory_cell",params['N_cortex'],params['N_cortex'],
        kernel)

#    Exc_Inh_dict = {
#    "connection_type":"convergent",
#    "mask": {"circular": {"radius": 0.15}},
#    "kernel": {"gaussian": {"p_center": 0.6, "sigma": 0.05}},
#    "delays" : {"normal": {"mean": 2.0, "std": 0.25, "min": params['resolution']}},
#    "synapse_model": "syn",
#    "weights": 0.0 * params['visSize'] / params['N_cortex'],
#    "sources": {"model": "Cortex_excitatory_cell"},
#    "targets": {"model": "Cortex_inhibitory_cell"},
#    "allow_autapses":False,
#    "allow_multapses":False
#    }

    sigma = 0.05
    mask_radius = sigma * 3.0
    kernel = 1.0

    center_weight = 0.5

    delay = 2.0
    std_delay = 0.25

    Exc_Inh_dict = make_dict_circular(params,mask_radius,center_weight,sigma,delay,
        std_delay,"Cortex_excitatory_cell","Cortex_inhibitory_cell",params['N_cortex'],params['N_cortex'],
        kernel)

#    Inh_Inh_dict = {
#    "connection_type":"convergent",
#    "mask": {"circular": {"radius": 0.075}},
#    "kernel": {"gaussian": {"p_center": 0.6, "sigma": 0.025}},
#    "delays" : {"normal": {"mean": 2.0, "std": 0.25, "min": params['resolution']}},
#    "synapse_model": "syn",
#    "weights": 0.0 * params['visSize'] / params['N_cortex'],
#    "sources": {"model": "Cortex_inhibitory_cell"},
#    "targets": {"model": "Cortex_inhibitory_cell"},
#    "allow_autapses":False,
#    "allow_multapses":False
#    }

    sigma = 0.025
    mask_radius = sigma * 3.0
    kernel = 1.0

    center_weight = -3.0

    delay = 2.0
    std_delay = 0.25

    Inh_Inh_dict = make_dict_circular(params,mask_radius,center_weight,sigma,delay,
        std_delay,"Cortex_inhibitory_cell","Cortex_inhibitory_cell",params['N_cortex'],params['N_cortex'],
        kernel)

#    Inh_Exc_dict = {
#    "connection_type":"convergent",
#    "mask": {"circular": {"radius": 0.075}},
#    "kernel": {"gaussian": {"p_center": 0.6, "sigma": 0.025}},
#    "delays" : {"normal": {"mean": 2.0, "std": 0.25, "min": params['resolution']}},
#    "synapse_model": "syn",
#    "weights": -2.0 * params['visSize'] / params['N_cortex'],
#    "sources": {"model": "Cortex_inhibitory_cell"},
#    "targets": {"model": "Cortex_excitatory_cell"},
#    "allow_autapses":False,
#    "allow_multapses":False
#    }

    sigma = 0.025
    mask_radius = sigma * 3.0
    kernel = 1.0

    center_weight = -3.0

    delay = 2.0
    std_delay = 0.25

    Inh_Exc_dict = make_dict_circular(params,mask_radius,center_weight,sigma,delay,
        std_delay,"Cortex_inhibitory_cell","Cortex_excitatory_cell",params['N_cortex'],params['N_cortex'],
        kernel)


    exc_model_IDs = [
    'Color_Luminance_L_ON_L_OFF_vertical',
    'Color_Luminance_L_ON_L_OFF_horizontal',
    'Color_Luminance_L_OFF_L_ON_vertical',
    'Color_Luminance_L_OFF_L_ON_horizontal',
    'Color_Luminance_M_ON_M_OFF_vertical',
    'Color_Luminance_M_ON_M_OFF_horizontal',
    'Color_Luminance_M_OFF_M_ON_vertical',
    'Color_Luminance_M_OFF_M_ON_horizontal',

    'Luminance_preferring_ON_OFF_vertical',
    'Luminance_preferring_ON_OFF_horizontal',
    'Luminance_preferring_OFF_ON_vertical',
    'Luminance_preferring_OFF_ON_horizontal',

    'Color_preferring_L_ON_M_OFF',
    'Color_preferring_M_ON_L_OFF'
    ]

    inh_model_IDs = [
    'Color_Luminance_inh_L_ON_L_OFF_vertical',
    'Color_Luminance_inh_L_ON_L_OFF_horizontal',
    'Color_Luminance_inh_L_OFF_L_ON_vertical',
    'Color_Luminance_inh_L_OFF_L_ON_horizontal',
    'Color_Luminance_inh_M_ON_M_OFF_vertical',
    'Color_Luminance_inh_M_ON_M_OFF_horizontal',
    'Color_Luminance_inh_M_OFF_M_ON_vertical',
    'Color_Luminance_inh_M_OFF_M_ON_horizontal',

    'Luminance_preferring_inh_ON_OFF_vertical',
    'Luminance_preferring_inh_ON_OFF_horizontal',
    'Luminance_preferring_inh_OFF_ON_vertical',
    'Luminance_preferring_inh_OFF_ON_horizontal',

    'Color_preferring_inh_L_ON_M_OFF',
    'Color_preferring_inh_M_ON_L_OFF'
    ]

    allconns = make_cortex_horizontal_connections(allconns,exc_model_IDs,inh_model_IDs,
    Exc_Exc_dict,Exc_Inh_dict,Inh_Inh_dict,Inh_Exc_dict)


    #############################################
    ### Gaussian white noise current ############
    #############################################

    # ----- Noise generators -> Relay cells ----- #

    Noise_LGN_Relay_cell_dict = {
    "connection_type":"convergent",
    "mask": {"circular": {"radius": 0.001}}, # one-to-one connection
    "kernel": 1.0,
    "delays" : params['resolution'],
    "synapse_model": "syn",
    "weights": 1.0,
    "sources": {"model": "thalamo_noise"},
    "targets": {"model": "LGN_relay_cell"},
    "allow_autapses":False,
    "allow_multapses":False
    }

    [allconns.append(['Noise_generators_LGN','Parvo_LGN_relay_cell_L_ON',Noise_LGN_Relay_cell_dict])]
    [allconns.append(['Noise_generators_LGN','Parvo_LGN_relay_cell_L_OFF',Noise_LGN_Relay_cell_dict])]
    [allconns.append(['Noise_generators_LGN','Parvo_LGN_relay_cell_M_ON',Noise_LGN_Relay_cell_dict])]
    [allconns.append(['Noise_generators_LGN','Parvo_LGN_relay_cell_M_OFF',Noise_LGN_Relay_cell_dict])]

    # ----- Noise generators -> LGN interneurons ----- #

    Noise_LGN_interneuron_dict = {
    "connection_type":"convergent",
    "mask": {"circular": {"radius": 0.001}}, # one-to-one connection
    "kernel": 1.0,
    "delays" : params['resolution'],
    "synapse_model": "syn",
    "weights": 1.0,
    "sources": {"model": "thalamo_noise"},
    "targets": {"model": "LGN_interneuron"},
    "allow_autapses":False,
    "allow_multapses":False
    }

    [allconns.append(['Noise_generators_LGN','Parvo_LGN_interneuron_ON',Noise_LGN_interneuron_dict])]
    [allconns.append(['Noise_generators_LGN','Parvo_LGN_interneuron_OFF',Noise_LGN_interneuron_dict])]

    # ----- Noise generators -> cortex excitatory ----- #

    Noise_cortex_exc_dict = {
    "connection_type":"convergent",
    "mask": {"circular": {"radius": 0.001}}, # one-to-one connection
    "kernel": 1.0,
    "delays" : params['resolution'],
    "synapse_model": "syn",
    "weights": 1.0,
    "sources": {"model": "thalamo_noise"},
    "targets": {"model": "Cortex_excitatory_cell"},
    "allow_autapses":False,
    "allow_multapses":False
    }

    [allconns.append(['Noise_generators_Color_Luminance',
    'Color_Luminance_L_ON_L_OFF_vertical',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Color_Luminance',
    'Color_Luminance_L_ON_L_OFF_horizontal',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Color_Luminance',
    'Color_Luminance_L_OFF_L_ON_vertical',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Color_Luminance',
    'Color_Luminance_L_OFF_L_ON_horizontal',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Color_Luminance',
    'Color_Luminance_M_ON_M_OFF_vertical',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Color_Luminance',
    'Color_Luminance_M_ON_M_OFF_horizontal',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Color_Luminance',
    'Color_Luminance_M_OFF_M_ON_vertical',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Color_Luminance',
    'Color_Luminance_M_OFF_M_ON_horizontal',Noise_cortex_exc_dict])]

    [allconns.append(['Noise_generators_Luminance_preferring',
    'Luminance_preferring_ON_OFF_vertical',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Luminance_preferring',
    'Luminance_preferring_ON_OFF_horizontal',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Luminance_preferring',
    'Luminance_preferring_OFF_ON_vertical',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Luminance_preferring',
    'Luminance_preferring_OFF_ON_horizontal',Noise_cortex_exc_dict])]

    [allconns.append(['Noise_generators_Color_preferring',
    'Color_preferring_L_ON_M_OFF',Noise_cortex_exc_dict])]
    [allconns.append(['Noise_generators_Color_preferring',
    'Color_preferring_M_ON_L_OFF',Noise_cortex_exc_dict])]

    # ----- Noise generators -> cortex inhibitory ----- #

    Noise_cortex_inh_dict = {
    "connection_type":"convergent",
    "mask": {"circular": {"radius": 0.001}}, # one-to-one connection
    "kernel": 1.0,
    "delays" : params['resolution'],
    "synapse_model": "syn",
    "weights": 1.0,
    "sources": {"model": "thalamo_noise"},
    "targets": {"model": "Cortex_inhibitory_cell"},
    "allow_autapses":False,
    "allow_multapses":False
    }

    [allconns.append(['Noise_generators_Color_Luminance_inh',
    'Color_Luminance_inh_L_ON_L_OFF_vertical',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Color_Luminance_inh',
    'Color_Luminance_inh_L_ON_L_OFF_horizontal',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Color_Luminance_inh',
    'Color_Luminance_inh_L_OFF_L_ON_vertical',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Color_Luminance_inh',
    'Color_Luminance_inh_L_OFF_L_ON_horizontal',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Color_Luminance_inh',
    'Color_Luminance_inh_M_ON_M_OFF_vertical',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Color_Luminance_inh',
    'Color_Luminance_inh_M_ON_M_OFF_horizontal',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Color_Luminance_inh',
    'Color_Luminance_inh_M_OFF_M_ON_vertical',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Color_Luminance_inh',
    'Color_Luminance_inh_M_OFF_M_ON_horizontal',Noise_cortex_inh_dict])]

    [allconns.append(['Noise_generators_Luminance_preferring_inh',
    'Luminance_preferring_inh_ON_OFF_vertical',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Luminance_preferring_inh',
    'Luminance_preferring_inh_ON_OFF_horizontal',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Luminance_preferring_inh',
    'Luminance_preferring_inh_OFF_ON_vertical',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Luminance_preferring_inh',
    'Luminance_preferring_inh_OFF_ON_horizontal',Noise_cortex_inh_dict])]

    [allconns.append(['Noise_generators_Color_preferring_inh',
    'Color_preferring_inh_L_ON_M_OFF',Noise_cortex_inh_dict])]
    [allconns.append(['Noise_generators_Color_preferring_inh',
    'Color_preferring_inh_M_ON_L_OFF',Noise_cortex_inh_dict])]


    return allconns
