# Will Solow, Skye Rhomberg
# CS446 Spring 2021
# Diffusion Monte Carlo (DMC) Simulation w/ Descendent Weighting
# Script Style
# Last Updated 4/27/2021

# This file has been running all the big sims on NSCC with command line input

# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. In this particular implementation, a 4D
# array is utilized as a main data structure for storing the coordinates of each atom 
# in each molecule in each walker. 

# To Run: Navigate to file in terminal directory and type 'python dmc_rs_H2O.py'

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
dt = .1


# Length of the equilibriation phase in time steps. The below data is for the water molecule
# If dt = 1.0, equilibriation phase should be greater than 1500
# If dt = 0.5, equilibriation phase should be greater than 2000
# If dt = 0.1, equilibriation phase should be greater than 5000
equilibriation_phase = 10000


# Number of time steps in a simulation
sim_length = 1000000

# Number of initial walkers
n_walkers = 20000

# Number of time steps for rolling average calculation
rolling_avg = 1000

# Wave function snapshots. Sample the wave function every x time steps
wave_func_interval = 1000


# Number of bins for histogram. More bins is more precise
n_bins = 50


#######################################################################################
# Import Simulation Constants from the command line

# Only run if all inputted constants are available

if len(sys.argv) < 5:
    print('\n\nUsage: DMC_rs_H2O_DW.py prop_range<int>, init_type<random, normal, one_atom, equil> input_file \
          output_file')
    sys.exit(0)


# Assign simulation constants
dt = .1
sim_length = 1000000
n_walkers = 10000
wave_func_interval = 10000
prop_range = float(sys.argv[1])
method = sys.argv[2]
input_file = sys.argv[3]
output_filename = sys.argv[4]

print(f'\n\nAssigned values are: \ndt: {dt} \nsim_length: {sim_length}\nn_walkers: {n_walkers}\nwave_func_interval: {wave_func_interval}\n\n')



####################################################################################
# Descendent Weighting Constants

# Interval at which we save `snapshots' of the walkers for use in DW simulation
prop_interval = 500

# Time period for which walkers are propogated during DW simulation loop
prop_period = 20
prop_steps = int(prop_period / dt)

# Number of times we run DW simulation loop
prop_reps = 5

####################################################################################
# Molecule Model Constants


# Number of molecules in each walker
# Used to initialize the walker array
num_molecules = 2


# Initialize the walker array depending on the method inputted 
if method == 'one_atom_random':
    walkers = lib.init_one_atom_random(n_walkers, prop_range, num_molecules, input_file)

elif method == 'one_atom_normal':
    walkers = lib.init_one_atom_normal(n_walkers, prop_range, num_molecules, input_file)

elif method == 'equil_random': 
    walkers = lib.init_equil_random(n_walkers, prop_range, num_molecules, input_file)

elif method == 'equil_normal':
    walkers = lib.init_equil_normal(n_walkers, prop_range, num_molecules, input_file)

elif method == 'normal': 
    walkers = lib.init_normal(n_walkers, num_molecules, prop_range)

else: 
    walkers = lib.init_random(n_walkers, num_molecules, prop_range)


#######################################################################################
# Simulation

start = time.time()

# Run the DMC simulation with the initailized walker array, all code code can be found in DMC_rs_lib.py
# Output is a dictionary of values which we access below
output_dict = lib.sim_loop(walkers,sim_length,dt,wf_save=wave_func_interval,output_filename=output_filename)

# Reference energy array of values at each iteration
ref_energy = output_dict['r']

# Number of walkers array of values at each iteration
num_walkers = output_dict['n']

np.savetxt(f'{output_filename}_dimer_{method}_{prop_range}_ref.txt', ref_energy)
np.savetxt(f'{output_filename}_dimer_{method}_{prop_range}_walkers.txt', num_walkers)





