# Primate Visual System

Descriptive models of the visual system have been essential to understand how retinal neurons convert visual stimuli into a neural response. With recent advancements of neuroimaging techniques, availability of an increasing amount of physiological data and current computational capabilities, we now have powerful resources for developing biologically more realistic models of the brain. In this project, we implemented a two-dimensional network model of the primate retina, LGN and layer 4C in V1 that uses conductance-based neurons. The model aims to provide neuroscientists with a realistic model whose parameters have been carefully tuned based on data from the primate fovea and whose response at every stage of the model adequately reproduces neuronal behavior.

The retina model is the result of a research work and its associated publication is:

* Martinez-Cañada, P., Morillas, C., Pelayo, F. “A Conductance-Based Neuronal Network Model for Color Coding in the Primate Foveal Retina”. In IWINAC 2017

The rest of the code, corresponding to the thalamocortical system, will be also released after publication.

The model was implemented using the well-known neural simulation tool NEST (https://github.com/nest/nest-simulator) and Python. Download, build and install NEST (version 2.11 or later). Install also CMake (version 2.8.12 or later).

Instructions to compile the neuron models in NEST follow the tutorial about “Writing an extension module” (https://nest.github.io/nest-simulator/extension_modules). Neuron models have been written to work with NEST 2.11. We will describe how to build the model 'retina_parvo'. A similar approach can be followed to compile the other 2 models ('retina_AII_amacrine' and 'retina_ganglion_cell').

# Building neuron models:

Define the environment variable 'NEST_INSTALL_DIR' to contain the path to which you have installed NEST, e.g. using bash:

```
export NEST_INSTALL_DIR=/Users/pablo/NEST/ins
```

Create a build directory on the same level as 'retina_parvo' folder:

```
cd models/retina_parvo
mkdir build
cd build
```

Configure. The configure process uses the script 'nest-config' to find out where NEST is installed, where the source code resides, and which compiler options were used for compiling NEST. If 'nest-config' is not in your path, you need to provided it explicitly like this (don't forget to add '..' at the end):

```
cmake -Dwith-nest=${NEST_INSTALL_DIR}/bin/nest-config ..
```

Compile:

```
make
make install
```

# Running a simulation:

Simulation scripts are in folder 'simulation/retina' and are 5:

* 'ex1_disk.py': retina response to light flashes, which can be disk- or ring-shaped

* 'ex2_square.py': retina response to flashing squares

* 'ex3_grating_spatial_freq.py': retina response to sine-wave gratings of varying spatial frequency

* 'ex4_disk_area_response.py': retina response to flashing spots of varying diameter

* 'ex5_receptive_field.py': estimation of the retina receptive fields

Adjust the simulation parameters in the script and execute it using the Python interpreter, e.g.:

```
python ex1_disk.py
```

The size of the network and other simulation parameters, such as the number of threads, can be set in the file 'run_network.py'. It is also possible to  run the different repetitions ('trials') of the simulation scripts in parallel taking advantage of MPI. The parallelization of the code is included in the script 'mpi_run.py'. You can use the 'mpirun' command to specify the number of processes to be created, e.g.:

```
mpirun -np 8 python mpi_run.py
```

would execute a simulation script with 8 MPI processes. 
