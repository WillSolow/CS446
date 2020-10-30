# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style


# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. In this test file, a print statement follows
# every change to the walkers array so that the user can verify its correctness

# Output: Graphs for the reference energy, the n-step rolling average, and the number 
# of walkers at each time step, as well as a density histogram of the walker distance
# from equilibrium and the corresponding wave function

# Imports

import sys
import platform
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

###################################################################################
# Scientific Constants


# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e+23

# create a random seed for the number generator, can be changed to a constant value
# for the purpose of replicability
seed = np.random.randint(100000)
seed = 46508

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
sim_length = 2

# Number of initial walkers
n_walkers = 10

# Number of time steps for rolling average calculation
rolling_avg = 1000

# Number of bins for histogram. More bins is more precise
n_bins = 50


####################################################################################
# Molecule Model Constants


# Atomic masses of atoms in system
# Used to calculate the atomic mass in Atomic Mass Unites
atom_mass = 10


# Equilibrium position of the system in atomic units
eq_length = 5



# Spring constant
k = 1.0

# Constant for covergence of reference energy
h = 1



# Calculate the reduced mass of the system
atomic_mass = atom_mass / (avogadro * electron_mass) 
reduced_mass = (atomic_mass * atomic_mass) / (atomic_mass + atomic_mass)


# Calculate the convergence reference energy based on the given equation.
ref_converge_num = .5*h*np.sqrt(k/reduced_mass)


np.set_printoptions(suppress = True)
np.set_printoptions(precision=8)
print('Python Version: ', sys.version_info.major)
print('Python Release: ', platform.python_version())
# Initial walker array
# Returns a uniform distriubtion centered at the equilibrium 
walkers = eq_length + (np.random.rand(n_walkers) - 0.5)
print('Initial Walkers: ',walkers)

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
    return .5 * k * (x - eq_length)**2

	
	
# Simulation loop
# Iterates over the walkers array, propogating each walker. Deletes and replicates those 
# walkers based on their potential energies with respect to the calculated reference energy
for i in range(sim_length):
    print('\n\nInitial Potential Energy: ', potential_energy(walkers))
	# Calculate the Reference Energy
	# Energy is calculated based on the average of all potential energies of walkers.
	# Is adjusted by a statistical value to account for large or small walker populations.
    print('\n\nAverage of Walker PE: ', np.mean(potential_energy(walkers)))
    
    reference_energy[i] = np.mean(potential_energy(walkers) ) \
        + (1.0 - (walkers.shape[0] / (1.0*n_walkers)) ) / ( 2.0*dt )
    print('\n\nReference Energy: ', reference_energy[i])
	
    # Current number of walkers
    num_walkers[i] = walkers.shape[0]
    
	
    # Propogates each atom in a normal distribution about its current position
    propogation_lengths = np.random.normal(0, np.sqrt(dt / reduced_mass), walkers.shape[0])
	
	# Adds the propogation lengths to the walker array
    walkers = walkers + propogation_lengths
    print('\n\nWalkers after Propagations: ', walkers)
	
	
    
    # Calculates the potential energy of each walker in the system
    potential_energies = potential_energy(walkers)
    print('\n\nPE after Prop: ', potential_energies)

    
	# Gives a uniform distribution in the range [0,1) associated with each walker
    # in the system
    # Used to calculate the chance that a walker is deleted or replicated	
    thresholds = np.random.rand(walkers.shape[0])
    print('\n\nThresholds: ', thresholds)
	
	
	# Calculates a probability for each walker that it is deleted
    # This is actually the probability that a walker is not deleted
    prob_delete = np.exp(-(potential_energies-reference_energy[i])*dt)
    print('\n\nProb Delete: ', prob_delete)

	# Calculates a probability for each walker that it is replicated
	# In the model it is based off of prob_delete
    prob_replicate = prob_delete - 1
    print('\n\nProb Replicate', prob_replicate)
	
	
	
	# Returns a boolean array of which walkers have a chance of surviving or being deleted
	# Based on the above probabilities and thresholds calculated for each walker
    # calculate which walkers actually have the necessary potential energies.
	# These two arrays are not mutually exclusive, but the calculations below ensure
	# that no walker is both deleted and replicated in the same time step.
    to_delete = prob_delete < thresholds
    print('\n\nBoolean Delete: ', to_delete)
    to_replicate = prob_replicate > thresholds
    print('\n\nBoolean Replicate: ', to_replicate)
    
	
	
	# Gives a boolean array of indices of the walkers that are not deleted
	# Calculates if a walker is deleted by if its potential energy is greater than
	# the reference energy and if its threshold is above the prob_delete threshold.
	# Notice that walkers_to_remain is mutually exclusive from walkers_to_replicate
	# as the potential energy calculate is exclusive.
    walkers_to_remain = np.invert( (potential_energies > reference_energy[i]) * to_delete )
    print('\n\nWalkers to remain: ', walkers_to_remain)
	
	# Returns the walkers that remain after deletion
    walkers_after_delete = walkers[walkers_to_remain]
    print('\n\nRemaining Walkers: ', walkers_after_delete)
	
	
	# Gives a boolean array of indices of the walkres that are replicated
	# Calculates if a walker is replicated by if its potential energy is less than
	# the reference energy and if its threshold is below the prob_replicate threshold.
    walkers_to_replicate = (potential_energies < reference_energy[i])*to_replicate
    print('\n\nWalkers to replicate: ', walkers_to_replicate)
	
	# Returns the walkers that are to be replicated
    walkers_after_replication = walkers[walkers_to_replicate]
    print('\n\nReplicated Walkers: ', walkers_after_replication)
	
	
	# Returns the new walker array
	# Concatenates the walkers that were not deleted with the walkers that are to be 
	# replicated. Since a replicated walker was not deleted, concatenating these two 
	# arrays serves to replicated a walker. 
	# Notice that if the potential energy is equal the reference energy, the walker 
	# will appear in the walkers_after_delete array but not in the 
	# walkers_after_replication array. This serves to ensure that in the unlikely case 
	# of equal potential and reference energy, the walker is neither replicated nor deleted. 
    walkers = np.append(walkers_after_delete, walkers_after_replication)
    print('\n\nWalkers after simulation ' + str(i) + ': ', walkers)
    print('\n\n\n##################################################\n\n\n')


