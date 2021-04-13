# Will Solow, Skye Rhomberg
# CS446 Spring 2021
# Diffusion Monte Carlo (DMC) Simulation w/ Descendent Weighting
# Script Style
# Last Updated 02/28/2021

# This is a library of the scientific constants and functions used in our DMC simulations
# Everything in here should be constant across all simulations which import this file
# This file is specifically for the CO bond simulation as the results can be verified
# against a known wave function

# Imports
import numpy as np
import itertools as it
import scipy.integrate as integrate

###################################################################################
# Scientific Constants


# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e23

# Number of coordinates
# Always 3, used for clarity
coord_const = 3

####################################################################################
# Simulation Loop Constants

# Set the dimensions of the 4D array of which the walkers, molecules, atoms, and positions 
# reside. Used for clarity in the simulation loop
walker_axis = 0
molecule_axis = 1
atom_axis = 2
coord_axis = 3


####################################################################################
# Molecule Model Constants


# Atomic masses of atoms in system
# Used to calculate the atomic masses in Atomic Mass Units
oxygen_mass = 15.99491461957
carbon_mass = 12.000



# Equilibrium length of CO Bond
# Input as equation to avoid rounding errors
eq_bond_length = 0.59707


# Spring constant of the OH Bond
# Input as equation to avoid rounding errors 
kCO = 1.2216



# Returns an array of masses in Atomic Mass Units, Oxygen is first followed by both 
# Hydrogens
atomic_masses = np.array([carbon_mass, oxygen_mass]) / (avogadro * electron_mass)



# Calculate the reduced mass of the system
# Note that as the wave function is being graphed for an OH vector, we only consider the
# reduced mass of the OH vector system
reduced_mass = np.prod(atomic_masses)/np.sum(atomic_masses)


####################################################################################
# Wave Function Calculations

# Part of the wave function. Used in integration to solve for the normalization constant
# under the assumption that the integral should be 1.
wave_func = lambda x: np.exp(-(x**2)*np.sqrt(kCO*reduced_mass)/2)

# Get the integral of the wave function and the error
integral_value, error = integrate.quad(wave_func, -np.inf, np.inf)

# Calculate the Normalization constant
norm_constant = 1 / integral_value


#######################################################################################
# Simulation

# Input: 4D Array of walkers
# Output: 1D Array of intramolecular potential energies for each walker
# Calculates the potential energy of a walker based on the distance of bond lengths 
# from equilibrium
def intra_pe(x):
 	# Calculate the potential energy in each atom based on the bond length
    distance = np.linalg.norm(x[:,0,0]-x[:,0,1], axis=1)
	
    return .5 * kCO * (distance - eq_bond_length)**2
    
    
# Input: 4D array of walkers
# Output: 1D array of the sum of the intermolecular and intramolecular potential 
# energy of each walker
def total_pe(x):

    # Calculate the intramolecular potential energy of each walker
    intra_potential_energy = intra_pe(x)

    # Calculate the intermolecular potential energy of each walker
    # In this model, we assume that there is no intermolecular interaction
    
    return intra_potential_energy


#######################################################################################
# Simulation loop
# Iterates over the walkers array, propogating each walker. Deletes and replicates those 
# walkers based on their potential energies with respect to the calculated reference energy


# Input:
# walkers: 4D numpy array (n_walkers, n_molecules, n_atoms, coord_const)
# sim_length: int. number of iterations of the main simulation loop
# dt: float. time step for simulation
# dw_save: int. interval (in number of sim steps) after which to save a snapshot
#   If value == 0, no snapshots will be saved
# do_dw: bool. If true, keep track of ancestors, return a bincount at end of loop
# Output: dict of various outputs
#   'w': walkers. ndarray -- shape: (n_walkers,n_molecules,n_atoms,coord_const) 
#   'r': reference energy at each time step. 1d array
#   'n': num_walkers at each time step. 1d array
#   's': snapshots. python list of walker 4D arrays
#   'a': ancestor_weights of each walker at sim end. 1d array
def sim_loop(walkers,sim_length,dt,dw_save=0,do_dw=False):

    # Extract initial size constants from walkers array
    n_walkers, num_molecules, n_atoms, coord_const = walkers.shape 

    # DW snapshots
    # Views of the walker array at various times during the simulation
    # Used for Descendent Weighting calculations
    snapshots = []

    # DW indexing array: initially just a list from 0 up to num_walkers - 1
    dw_indices = np.arange(walkers.shape[0])


    # Create array to store the number of walkers at each time step
    num_walkers = np.zeros(sim_length)
    
    # Create array to store the reference energy at each time step
    reference_energy = np.zeros(sim_length)
    
    for i in range(sim_length):

        # DW saving
        if dw_save > 0 and i % dw_save == 0:
            snapshots.append(np.copy(walkers))

        # Calculate the Reference Energy
        # Energy is calculated based on the average of all potential energies of walkers.
        # Is adjusted by a statistical value to account for large or small walker populations.
        reference_energy[i] = np.mean( intra_pe(walkers) ) \
            + (1.0 - (walkers.shape[walker_axis] / n_walkers) ) / ( 2.0*dt )
                    
        # Current number of walkers
        num_walkers[i] = walkers.shape[walker_axis]
        #print('Num walkers: ', num_walkers[i])

            
        # Propagates each coordinate of each atom in each molecule of each walker within a normal
        # distribution given by the atomic mass of each atom.
        # Returns a 4D array in the shape of walkers with the standard deviation depending on the
        # atomic mass of each atom	
        propagations = np.random.normal(0, np.sqrt(dt/np.transpose(np.tile(atomic_masses, \
                (walkers.shape[walker_axis], num_molecules, coord_const, 1)), \
            (walker_axis, molecule_axis, coord_axis, atom_axis))))
                    
        # Adds the propagation lengths to the 4D walker array
        walkers = walkers + propagations
            
        
            
        # Calculates the potential energy of each walker in the system
        potential_energies = total_pe(walkers)

        
            
        # Gives a uniform distribution in the range [0,1) associated with each walker
        # in the system
        # Used to calculate the chance that a walker is deleted or replicated	
        thresholds = np.random.rand(walkers.shape[walker_axis])
        #thresholds = np.random.rand()
            
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
        walkers_to_replicate = (potential_energies < reference_energy[i]) * to_replicate
            
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
        walkers = np.append(walkers_after_delete, walkers_after_replication, axis = walker_axis)

        # Descendent Weighting Process
        if do_dw:
            # Descendant indices remaining after deletion
            # Analogous to the walker array, but with index id instead of cartesian position
            descendents_after_delete = dw_indices[walkers_to_remain]

            # Descendant indices that will be replicated
            descendents_after_replication = dw_indices[walkers_to_replicate]

            # New descendant indices corresponding to ancestors of current generation
            # in original generation
            dw_indices = np.append(descendents_after_delete, descendents_after_replication, axis=0)

    # number of walkers in final generation with each walker from 1st generation
    # as an ancestor
    # Returns empty if DW wasn't enabled
    ancestor_weights = np.bincount(dw_indices,minlength=n_walkers) if do_dw else []

    # All possible returns
    # To access a particular output: sim_loop(...)['w|r|n|s|a']
    return {'w':walkers, 'r':reference_energy, 'n':num_walkers, 's':snapshots, 'a':ancestor_weights}
