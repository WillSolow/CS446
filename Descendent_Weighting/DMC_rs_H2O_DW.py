# Will Solow, Skye Rhomberg
# CS446 Spring 2021
# Diffusion Monte Carlo (DMC) Simulation w/ Descendent Weighting
# Script Style
# Last Updated 02/28/2021

# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. In this particular implementation, a 4D
# array is utilized as a main data structure for storing the coordinates of each atom 
# in each molecule in each walker. 

# To Run: Navigate to file in terminal directory and type 'python dmc_rs_4D_H2O.py'

# We add Descendent Weighting: a process for producing more balanced histograms
# TODO: doc this further

# Output: Graphs for the reference energy, the n-step rolling average, and the number 
# of walkers at each time step, as well as a density histogram of the walker distance
# from equilibrium and the corresponding wave function

# Imports
import numpy as np
import DMC_rs_lib as lib
import DMC_rs_print_xyz_lib as out
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time
import sys

# Set print options to suppress scientific notation
np.set_printoptions(suppress=True)

# Ignore runtime divide by zero erros which can occur when the distances between two
# atoms are equal in the intermolecular potential energy function
np.seterr(divide='ignore')


###################################################################################
# Scientific Constants

# Create a random seed for the number generator, can be changed to a constant value
# for the purpose of replicability
seed = np.random.randint(1000000)
# Set the seed manually for replicability purposes over multiple simulations
#seed = 

# Set the seed for the pseudo-random number generator. 
np.random.seed(seed)
print('Seed used: ' + str(seed))

####################################################################################
# Simulation Loop Constants


# Time step 
# Used to calculate the distance an atom moves in a time step
# Smaller time step means less movement in a given time step
dt = 1


# Length of the equilibriation phase in time steps. The below data is for the water molecule
# If dt = 1.0, equilibriation phase should be greater than 1500
# If dt = 0.5, equilibriation phase should be greater than 2000
# If dt = 0.1, equilibriation phase should be greater than 5000
equilibriation_phase = 1500


# Number of time steps in a simulation
sim_length = 1000

# Number of initial walkers
n_walkers = 1000

# Number of time steps for rolling average calculation
rolling_avg = 1000


# Number of bins for histogram. More bins is more precise
n_bins = 50


####################################################################################
# Descendent Weighting Constants

# Interval at which we save `snapshots' of the walkers for use in DW simulation
prop_interval = 100

# Time period for which walkers are propogated during DW simulation loop
prop_period = 100
prop_steps = int(prop_period / dt)

# Number of times we run DW simulation loop
prop_reps = 10

####################################################################################
# Molecule Model Constants


# Number of molecules in each walker
# Used to initialize the walker array
num_molecules = 1

# Filename (string)
# Used to initialize system. Should be a .xyz filename with the xyz positions of 
# one walker in the system.
filename = 'm_water.xyz'

print(out.gen_walker_array(filename))
sys.exit(0)


# Initial 4D walker array
# Returns a uniform distribution cenetered at the given bond length
# Array axes are walkers, molecules, atoms, coordinates
walkers = (np.random.rand(n_walkers, num_molecules, lib.atomic_masses.shape[0],lib.coord_const) - .5) 
#walkers = np.load('5000_walker.npy')


#######################################################################################
# Simulation

# Equilibriate Walkers
walkers = lib.sim_loop(walkers,equilibriation_phase,dt)['w']
	
# Simulation loop
sim_out = lib.sim_loop(walkers,sim_length,dt,prop_interval)
walkers, reference_energy, num_walkers, snapshots = [sim_out[k] for k in 'wrns']


################################################################################
# Output - DW

#TODO avoid figure clashes
#TODO change output listcomp to support xyz printing


# Uncomment the below line to avoid graphing 
# sys.exit(0)

# For every snapshot produced in the main simulation loop
# Run a (usually shorter) simulation keeping track of descendants
# These ancestor weights directly weight each walker when plotted
# In the histogram of positions
for i,walkers in enumerate(snapshots):
    # print(f'Snapshot: {i}')

    # Run each %prop-reps% simulations on each snapshot and average 
    # Producing a histogram for that snapshot based on a larger dataset
    ancestor_weights = np.mean(np.stack( \
            [lib.sim_loop(snapshots[i],prop_steps,dt,do_dw=True)['a'] \
            for j in range(prop_reps)],axis=-1),axis=1)

    # print(ancestor_weights.shape)


    # Calculate the distance between one of the OH vectors
    # Used in the histogram and wave function plot	
    OH_positions = np.linalg.norm(walkers[:,0,0]-walkers[:,0,1], axis = 1)

    plt.figure(i)
    plt.hist(OH_positions,weights=ancestor_weights,bins=n_bins,density=True)
    plt.xlabel('Walker OH Bond Length')
    plt.ylabel('Density')
    plt.title(f'Density of OH Bond Length at Snapshot {i*prop_interval}')


plt.show()