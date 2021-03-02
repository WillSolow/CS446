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

# Initial 4D walker array
# Returns a uniform distribution cenetered at the given bond length
# Array axes are walkers, molecules, coordinates, and atoms
walkers = (np.random.rand(n_walkers, num_molecules, lib.atomic_masses.shape[0], \
    lib.coord_const) - .5) 
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


################################################################################
# Output

# Uncomment the below line to avoid graphing 
sys.exit(0)


# Calculate the rolling average for rolling_avg time steps
ref_rolling_avg = np.zeros(sim_length)
for i in range(rolling_avg, sim_length):
    # Calculate the rolling average by looping over the past rolling_avg time steps 
    for j in range(rolling_avg):
        ref_rolling_avg[i] = (ref_rolling_avg[i]-(ref_rolling_avg[i] / (j+1))) \
                             + ( reference_energy[i-j] / (j+1) )
			
			
# Calculate the average reference convergence energy based on reference energy 
# after the equilibriation phase
ref_converge_num = np.mean(ref_rolling_avg[rolling_avg:])


# Create walker num array for plotting
init_walkers = (np.zeros(sim_length) + 1 ) * n_walkers

# Create Zero Point Energy array for plotting
zp_energy = (np.zeros(sim_length) + 1) * ref_converge_num	

			

# Calculate the distance between one of the OH vectors
# Used in the histogram and wave function plot	
OH_positions = np.linalg.norm(walkers[:,0,0]-walkers[:,0,1], axis = 1)



# Get the range to graph the wave function in
# Step is .001, which is usually a good smooth value
x = np.arange(OH_positions.min(), OH_positions.max(), step = .001)

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
	

# Plot the reference energy throughout the simulation
plt.figure(1)
plt.plot(reference_energy, label= 'Reference Energy')
plt.plot(zp_energy, label='ZP Energy (' + str.format('{0:.6f}', ref_converge_num) + ')')
plt.axis([0,sim_length,.17,.195])
plt.xlabel('Simulation Iteration')
plt.ylabel('Reference Energy')
plt.title('Reference Energy')
plt.legend()

# Plot the rolling average of the reference energy throughout the simulation
plt.figure(2)
plt.plot(np.arange(rolling_avg,sim_length),ref_rolling_avg[rolling_avg:], label= 'Reference Energy')
plt.plot(zp_energy, label='ZP Energy (' + str.format('{0:.6f}', ref_converge_num) + ')')
plt.axis([0,sim_length,.183,.192])
plt.xlabel('Simulation Iteration')
plt.ylabel('Reference Energy')
plt.title(str(rolling_avg) + ' Step Rolling Average')
plt.legend()


# Plot the number of walkers throughout the simulation
plt.figure(3)
plt.plot(num_walkers, label='Current Walkers')
plt.plot(init_walkers, label='Initial Walkers')
plt.xlabel('Simulation Iteration')
plt.ylabel('Number of Walkers')
plt.title('Number of Walkers Over Time')
plt.legend()


# Plot a density histogram of the walkers at the final iteration of the simulation
plt.figure(4)
plt.hist(OH_positions, bins=n_bins, density=True)
plt.plot(x, N*np.exp(-((x-eq_bond_length)**2)*np.sqrt(kOH*reduced_mass)/2), label = 'Wave Function (Norm Constant ' + str.format('{0:.4f}' + ')', N))
plt.xlabel('Walker Position')
plt.ylabel('Density of Walkers')
plt.title('Density of Walker Positions')
plt.legend()

# Plot a density histogram of the angles that the oxygen molecules form at the 
# final iteration of the simulation
plt.figure(5)
plt.hist(o_ang_1, bins=n_bins, density=True)
plt.xlabel('Oxygen Angle 1 in a Walker')
plt.ylabel('Density of Walkers')
plt.title('Density of Walker Oxygen Angle 1')

plt.figure(6)
plt.hist(o_ang_2, bins=n_bins, density=True)
plt.xlabel('Oxygen Angle 2 in a Walker')
plt.ylabel('Density of Walkers')
plt.title('Density of Walker Oxygen Angle 2')

plt.figure(7)
plt.hist(o_ang_3, bins=n_bins, density=True)
plt.xlabel('Oxygen Angle 3 in a Walker')
plt.ylabel('Density of Walkers')
plt.title('Density of Walker Oxygen Angle 3')

plt.figure(8)
plt.hist(o_angles, bins=n_bins, density=True)
plt.xlabel('Oxygen Angle in a Walker')
plt.ylabel('Density of Walkers')
plt.title('Density of Walker Oxygen Angles')

plt.show()

