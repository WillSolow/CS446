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
'''
if len(sys.argv) < 4:
    print('\n\nUsage: DMC_rs_H2O_DW.py dt sim_length n_walkers wave_func_interval filename')
    print(f'\nDefault is: \ndt: {dt} \nsim_length: {sim_length}\nn_walkers: {n_walkers}\nwave_func_interval: {wave_func_interval}\n\n')
    sys.exit(0)
'''

# Assign simulation constants
dt = float(sys.argv[1])
sim_length = int(sys.argv[2])
n_walkers = int(sys.argv[3])
wave_func_interval = int(sys.argv[4])
output_filename = sys.argv[5]

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
num_molecules = 3

# Filename (string)
# Used to initialize system. Should be a .xyz filename with the xyz positions of 
# one walker in the system.
filename = 'm_trimer.xyz'

# If using WebMO intitialization, uncomment this line below
# Reads in a .xyz file of a 1 walker system and broadcasts to n_walker array with 
# some amount of propagation simulate some randomness but not so much that the system
# cannot equilibrate

# Propagation amount
prop_amount = .5
walkers, num_molecules = out.gen_walker_array(filename, n_walkers, prop_amount, num_molecules)


# Uncomment the code below if doing an initialization within a random range
# Initial 4D walker array
# Returns a uniform distribution cenetered at the given bond length
# Array axes are walkers, molecules, atoms, coordinates
#walkers = (np.random.rand(n_walkers, num_molecules, lib.atomic_masses.shape[0],lib.coord_const) - .5) 


# Uncomment this line if loading in an already equliibrated walker array
#walkers = np.load('5000_walker.npy')


#######################################################################################
# Simulation

start = time.time()
# Equilibriate Walkers
#walkers = lib.sim_loop(walkers,equilibriation_phase,dt)['w']

ref_energy = lib.sim_loop(walkers,sim_length,dt,wf_save=wave_func_interval,output_filename=output_filename)['r']

np.savetxt(f'{output_filename}_cr_ref',[np.mean(ref_energy[int(sim_length/2):sim_length])])
#np.save(f'dt{dt}_sim{sim_length}_walk{n_walkers}',wave_func_out)

sys.exit(0)



# Simulation loop for descentdent weighting
'''
sim_out = lib.sim_loop(walkers,sim_length,dt,dw_save=prop_interval)
walkers, reference_energy, num_walkers, snapshots = [sim_out[k] for k in 'wrns']
'''

################################################################################
# Output - DW
'''
#TODO avoid figure clashes
#TODO change output listcomp to support xyz printing

# Uncomment the below line to avoid graphing for DW 
sys.exit(0)

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
    #OH_positions = np.linalg.norm(walkers[:,0,0]-walkers[:,0,1], axis = 1)



    # Uncomment below for OOO angles
    oxy = walkers[:,:,0]
    oxy_vec_10 = oxy[:,1]-oxy[:,0]
    oxy_vec_20 = oxy[:,2]-oxy[:,0]
    oxy_vec_21 = oxy[:,2]-oxy[:,1]
    oxy_ln_10 = np.linalg.norm(oxy_vec_10, axis=1)
    oxy_ln_20 = np.linalg.norm(oxy_vec_20, axis=1)
    oxy_ln_21 = np.linalg.norm(oxy_vec_21, axis=1)

    o_ang_1 = (180/np.pi)*np.arccos(np.sum(oxy_vec_10*oxy_vec_20, axis=1) / \
        (oxy_ln_10*oxy_ln_20))
    o_ang_2 = (180/np.pi)*np.arccos(np.sum(-oxy_vec_10*oxy_vec_21, axis=1) / \
            (oxy_ln_10*oxy_ln_21))
    o_ang_3 = (180/np.pi)*np.arccos(np.sum(-oxy_vec_20*-oxy_vec_21, axis=1) / \
        (oxy_ln_20*oxy_ln_21))

      
    o_angles = np.concatenate((o_ang_1,o_ang_2,o_ang_3),axis=0)


    plt.figure(i)

    # Uncomment below for OOO angles
    
    plt.hist(o_angles,weights=np.tile(ancestor_weights,3),bins=n_bins,density=True)
    plt.xlabel('Walker Oxygen Bond Angle')
    plt.ylabel('Density')
    plt.title(f'Density of Oxygen Bond Angles at Snapshot {i*prop_interval}')
    

    # Uncomment below for OH bond
    
    #plt.hist(OH_positions,weights=ancestor_weights,bins=n_bins,density=True)
    #plt.xlabel('Walker OH Bond Length')
    #plt.ylabel('Density')
    #plt.title(f'Density of OH Bond Length at Snapshot {i*prop_interval}')
    


plt.show()

'''