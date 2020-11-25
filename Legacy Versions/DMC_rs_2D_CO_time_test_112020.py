# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style


# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. In this particular implementation, 
# the system is the Carbon Monoxide bond, represented in a 2D array of coordinates in 3D space
# for each walker.

# To Run: Navigate to file in terminal directory and type 'python dmc_rs_2D_CO.py'

# Output: The elapsed time taken to run a number of simulations. Used in the 
# comparison of our implementation compared to Object oriented implementation.

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time

###################################################################################
# Scientific Constants


# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e+23

# create a random seed for the number generator, can be changed to a constant value
# for the purpose of replicability
seed = np.random.randint(100000)

# Set the seed for the pseudo-random number generator. 
np.random.seed(seed)
print('Seed used: ' + str(seed))


####################################################################################
# Simulation Loop Constants


# Time step 
# Used to calculate the distance an atom moves in a time step
# Smaller time step means less movement in a given time step
dt = 10.0

# Number of time steps in a simulation
sim_length = 5000

# Number of initial walkers
n_walkers = 1000

# Number of simulations rand
num_sims = 10

# Number of time steps for rolling average calculation
rolling_avg = 1000

# Number of bins for histogram. More bins is more precise
n_bins = 50

# Dimensions in system. Two walkers is 6 dimensional as it takes 6 coordinates to
# simulate the walker in 3 space
system_dimensions = 6


####################################################################################
# Molecule Model Constants

# Atomic masses of atoms in system
# Used to calculate the atomic mass in Atomic Mass Unites
carbon_mass = 12.000
oxygen_mass = 15.995


# Equilibrium position of the system in atomic units
eq_bond_length = 0.59707


# Spring constant
k = 1.2216

# Experimentally calculated reference energy
ref_converge_num = .00494317


# Calculates the atomic masses in Atomic Mass Units
atomic_mass_carbon = carbon_mass / (avogadro * electron_mass)
atomic_mass_oxygen = oxygen_mass / (avogadro * electron_mass)

# Calculates the reduced mass of the system
# Used when graphing the wave fuction
reduced_mass = (atomic_mass_carbon*atomic_mass_oxygen) / (atomic_mass_carbon+atomic_mass_oxygen)



# Initial walker array
# Returns a uniform distriubtion centered at the equilibrium 
walkers = eq_bond_length + (np.random.rand(n_walkers, system_dimensions) - 0.5)


#######################################################################################
# Simulation


# Create arrays to store values for plotting at each time step
reference_energy = np.zeros(sim_length)
num_walkers = np.zeros(sim_length)

# Input: Array of walkers
# Output: Array of potential energies for each walker
# Calculates the potential energy of a walker based on its distance from the equilibrium
# bond length
def potential_energy(x):
	# Calculate the distance in 3D space between the two atoms in each walker
    distance = np.sqrt( (x[:,0]-x[:,3])**2 + (x[:,1]-x[:,4])**2 + (x[:,2]-x[:,5])**2)
    return .5 * k * (distance - eq_bond_length)**2
	
def sim_loop():

    # Initial walker array
    # Returns a uniform distriubtion centered at the equilibrium 
    walkers = eq_bond_length + (np.random.rand(n_walkers, system_dimensions) - 0.5)
    start_time = time.time()
    # Simulation loop
    # Iterates over the walkers array, propogating each walker. Deletes and replicates those 
    # walkers based on their potential energies with respect to the calculated reference energy
    
    for i in range(sim_length):
    
        # Calculate the Reference Energy
        # Energy is calculated based on the average of all potential energies of walkers.
        # Is adjusted by a statistical value to account for large or small walker populations.
        reference_energy[i] = np.mean( potential_energy(walkers) ) \
        + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
    
	
        # Current number of walkers
        num_walkers[i] = walkers.shape[0]
    
	
	
        # Propagates each atom in a normal distribution about its current position
        propagate_carbon = np.random.normal(0, np.sqrt(dt/atomic_mass_carbon), \
                (walkers.shape[0], int(system_dimensions/2)))
        propagate_oxygen = np.random.normal(0, np.sqrt(dt/atomic_mass_oxygen), \
                (walkers.shape[0], int(system_dimensions/2)))
        
        # Adds the propagation lengths to the walker array
        walkers = walkers + np.append(propagate_carbon, propagate_oxygen, axis=1)
	
	
    
        # Calculates the potential energy of each walker in the system
        potential_energies = potential_energy(walkers)

    
	
        # Gives a uniform distribution in the range [0,1) associated with each walker
        # in the system
        # Used to calculate the chance that a walker is deleted or replicated	
        thresholds = np.random.rand(walkers.shape[0])
	
	
        # Calculates a probability for each walker that it is deleted
        # This is actually the probability that a walker is not deleted
        prob_delete = np.exp(-(potential_energies-reference_energy[i])*dt)

        # Calculates a probability for each walker that it is replicated
        # In the model it is based off of prob_delete
        prob_replicate = prob_delete - 1
    
	
	
        # Returns a boolean array of which walkers have a chance of surviving or being deleted
        # Based on the above probabilities and thresholds calculated for each walker
        # calculate which walkers actually have the necessary potential energies.
        # These two arrays are not mutually exclusive, but the calculations below ensure
        # that no walker is both deleted and replicated in the same time step.
        to_delete = prob_delete < thresholds
        to_replicate = prob_replicate > thresholds
    
	
	
        # Gives a boolean array of indices of the walkers that are not deleted
        # Calculates if a walker is deleted by if its potential energy is greater than
        # the reference energy and if its threshold is above the prob_delete threshold.
        # Notice that walkers_to_remain is mutually exclusive from walkers_to_replicate
        # as the potential energy calculate is exclusive.
        walkers_to_remain = np.invert( (potential_energies > reference_energy[i]) * to_delete )
	
        # Returns the walkers that remain after deletion
        walkers_after_delete = walkers[walkers_to_remain]

	
	
        # Gives a boolean array of indices of the walkres that are replicated
        # Calculates if a walker is replicated by if its potential energy is less than
        # the reference energy and if its threshold is below the prob_replicate threshold.
        walkers_to_replicate = (potential_energies < reference_energy[i])*to_replicate
	
        # Returns the walkers that are to be replicated
        walkers_after_replication = walkers[walkers_to_replicate]
	
	
	
        # Returns the new walker array
        # Concatenates the walkers that were not deleted with the walkers that are to be 
        # replicated. Since a replicated walker was not deleted, concatenating these two 
        # arrays serves to replicated a walker. 
        # Notice that if the potential energy is equal the reference energy, the walker 
        # will appear in the walkers_after_delete array but not in the 
        # walkers_after_replication array. This serves to ensure that in the unlikely case 
        # of equal potential and reference energy, the walker is neither replicated nor deleted. 
        walkers = np.append(walkers_after_delete, walkers_after_replication, axis=0)
    return time.time()-start_time
	
	
#####################################################################################
# Output


time_steps = [1000, 2000, 5000, 10000]
number_walkers = [100, 200, 500, 1000]

for i in time_steps:
    sim_length = i
    for j in number_walkers:
        n_walkers = j
        print('\n\nSimulation average with '+str(i)+' steps with '+str(j)+' walkers:')
        elapsed_time = []
        for k in range(num_sims):
            elapsed_time.append(sim_loop())
        print('Average time: '+str(np.mean(elapsed_time)))





