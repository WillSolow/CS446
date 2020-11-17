# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 11/04/20

# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. In this particular implementation, a 4D
# array is utilized as a main data structure for storing the coordinates of each atom 
# in each molecule in each walker. 

# To Run: Navigate to file in terminal directory and type 'python dmc_rs_4D_H2O.py'

# Output: Graphs for the reference energy, the n-step rolling average, and the number 
# of walkers at each time step, as well as a density histogram of the walker distance
# from equilibrium and the corresponding wave function

# Imports
import numpy as np
import itertools as it
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


# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e23


# Chemsitry constants for intermolecular energy
# Input as equation to avoid rounding errors
# Rounding should be at least 15 decimals otherwise error in the Lennard Jones 
# Energy will be incorrect by a magnitude of at least 100 depending on the 
# distance between atoms
sigma = 3.165492 / 0.529177
epsilon = 0.1554252 * (4.184 / 2625.5)

# Coulombic charges
q_oxygen = -.84
q_hydrogen = .42

# Coulomb's Constant
coulomb_const = 1.0 / (4.0*np.pi)


# Normalization constant
# Used in graphing the wave function. Can be found experimentally using the file
# dmc_rs_norm_constant.py. 
# Not calculated here given the time it takes each simulation
N = 4.0303907719347185


# Number of coordinates
# Always 3, used for clarity
coord_const = 3


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


# Length of the equilibration phase in time steps. The below data is for the water molecule
# If dt = 1.0, equilibration phase should be greater than 1500
# If dt = 0.5, equilibration phase should be greater than 2000
# If dt = 0.1, equilibration phase should be greater than 5000
equilibration_phase = 1500



# Number of time steps in a simulation
sim_length = 5000

# Number of initial walkers
n_walkers = 10000

# Number of time steps for rolling average calculation
rolling_avg = 1000


# Number of bins for histogram. More bins is more precise
n_bins = 50


# Set the dimensions of the 4D array of which the walkers, molecules, atoms, and positions 
# reside. Used for clarity in the simulation loop
walker_axis = 0
molecule_axis = 1
atom_axis = 2
coord_axis = 3



####################################################################################
# Molecule Model Constants


# Number of molecules in each walker
# Used to initialize the walker array
num_molecules = 3



# Atomic masses of atoms in system
# Used to calculate the atomic masses in Atomic Mass Units
oxygen_mass = 15.99491461957
hydrogen_mass = 1.007825
HOH_bond_angle = 112.0



# Equilibrium length of OH Bond
# Input as equation to avoid rounding errors
eq_bond_length = 1.0 / 0.529177

# Equilibrium angle of the HOH bond in radians
eq_bond_angle = HOH_bond_angle * np.pi/180

# Spring constant of the OH Bond
# Input as equation to avoid rounding errors 
kOH = 1059.162 * (1.0 / 0.529177)**2 * (4.184 / 2625.5)

# Spring constant of the HOH bond angle
kA = 75.90 * (4.184 / 2625.5)



# Returns an array of masses in Atomic Mass Units, Oxygen is first followed by both 
# Hydrogens
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)


# Returns an array of atomic charges based on the position of the atoms in the atomic_masses array
# This is used in the potential energy function and is broadcasted to an array of distances to
# calculate the energy using Coulomb's Law. 
atomic_charges = np.array([q_oxygen, q_hydrogen, q_hydrogen])



# Calculate the reduced mass of the system
# Note that as the wave function is being graphed for an OH vector, we only consider the
# reduced mass of the OH vector system
reduced_mass = (atomic_masses[0]*atomic_masses[1])/(atomic_masses[0]+atomic_masses[1])



# Initial 4D walker array
# Returns a uniform distribution cenetered at the given bond length
# Array axes are walkers, molecules, coordinates, and atoms
#walkers = (np.random.rand(n_walkers, num_molecules, atomic_masses.shape[0], \
#    coord_const) - .5) 

# Alternatively, load a 4D array of equilibrated walkers. Used in more complex
# systems where equilibration takes a large amount of time due to that amount of
# randomness introduced to the system on initialization
walkers = np.load('10000_water_trimer.npy')


# Stack another water molecule onto the walkers array to get a water trimer system

# Create a new water molecule from the old and propagate it a little bit
#new_water = walkers[:,0,np.newaxis,:,:] + np.array([-1.98702975, 5.1702229, 1.29956568])

#walkers = np.append(walkers, new_water, axis=1)
#print('Walkers shape: ', walkers.shape)
#print('walkers: \n', walkers)


#######################################################################################
# Simulation

# Create indexing arrays for the distinct pairs of water molecules in the potential 
# energy calculation. Based on the idea that there are num_molecules choose 2 distinct
# molecular pairs
molecule_index_a = np.array(sum([[i]*(num_molecules-(i+1)) \
                   for i in range(num_molecules-1)],[]))
molecule_index_b = np.array(sum([list(range(i,num_molecules)) \
                   for i in range(1,num_molecules)],[]))


# Create an array of the charges 
# Computes the product of the charges as the atom charges are multiplied together in accordance
# with Coulomb's Law.
coulombic_charges = (np.transpose(atomic_charges[np.newaxis]) \
                    @ atomic_charges[np.newaxis])  * coulomb_const


# Create arrays to store values for plotting at each time step
reference_energy = np.zeros(sim_length)
num_walkers = np.zeros(sim_length)


# Input: 4D Array of walkers
# Output: 1D Array of intramolecular potential energies for each walker
# Calculates the potential energy of a walker based on the distance of bond lengths and 
# bond angles from equilibrium
def intra_pe(x):
    # Return the two OH vectors
	# Used to calculate the bond lengths and angle in a molecule
    OH_vectors = x[:,:,np.newaxis,0]-x[:,:,1:]
	
    # Returns the lengths of each OH bond vector for each molecule 
	# in each walker. 
    lengths = np.linalg.norm(OH_vectors, axis=3)
	
	# Calculates the bond angle in the HOH bond
	# Computes the arccosine of the dot product between the two vectors, by normalizing the
	# vectors to magnitude of 1
    angle = np.arccos(np.sum(OH_vectors[:,:,0]*-OH_vectors[:,:,1], axis=2) \
	        / np.prod(lengths, axis=2))
			
	# Calculates the potential energies based on the magnitude vector and bond angle
    pe_bond_lengths = .5 * kOH * (lengths - eq_bond_length)**2
    pe_bond_angle = .5 * kA * (angle - eq_bond_angle)**2
	
	# Sums the potential energy of the bond lengths with the bond angle to get potential energy
	# of one molecule, then summing to get potential energy of each walker
    return np.sum(np.sum(pe_bond_lengths, axis = 2)+pe_bond_angle, axis=1)

    
    
# The lambda function below changes all instances of -inf or inf in a numpy array to 0
# under the assumption that the -inf or inf values result from divisions by 0
inf_to_zero = lambda dist: np.where(np.abs(dist) == np.inf, 0, dist)
    

# Input: 4D Array of walkers
# Output: Three 1D arrays for Intermolecular Potential Energy, Coulombic energy, and 
#         Leonard Jones energy
# Calculates the intermolecular potential energy of a walker based on the distances of the
# atoms in each walker from one another
def inter_pe(x):
    
    # Returns the atom positions between two distinct pairs of molecules 
    # in each walker. This broadcasts from a 4D array of walkers with axis dimesions 
    # (num_walkers, num_molecules, num_atoms, coord_const) to two arrays with 
    # dimesions (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const),
    # with the result being the dimensions:
    # (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const).
    # These arrays line up such that the corresponding pairs on the second dimension are 
    # the distinct pairs of molecules
    pairs_a = x[:,molecule_index_a]
    pairs_b = x[:,molecule_index_b]
    
    
    
    # Returns the distances between two atoms in each molecule pair. The distance array is 
    # now of dimension (num_walkers, num_distinct_pairs, num_atoms, num_atoms) as each
    # atom in the molecule has its distance computed with each atom in the other molecule in
    # the distinct pair.
    # This line works similar to numpy's matrix multiplication by broadcasting the 4D array
    # to a higher dimesion and then taking the elementwise difference before squarring and then
    # summing along the positions axis to collapse the array into distances.
    distances = np.sqrt( np.sum( (pairs_a[...,None] \
            - pairs_b[:,:,np.newaxis,...].transpose(0,1,2,4,3) )**2, axis=3) )
   
   
   
    # Calculate the Coulombic energy using Coulomb's Law of every walker. 
    # Distances is a 4D array and this division broadcasts to a 4D array of Coulombic energies
    # where each element is the Coulombic energy of an atom pair in a distinct pair of water 
    # molecules. 
    # Summing along the last three axis gives the Coulombic energy of each walker.
    # Note that we account for any instances of divide by zero by calling inf_to_zero on the
    # result of dividing coulombic charges by distance.
    coulombic_energy = np.sum( inf_to_zero(coulombic_charges / distances), axis=(1,2,3))
    
    
    

    # Calculate the quotient of sigma with the distances between pairs of oxygen molecules
    # Given that the Lennard Jones energy is only calculated for oxygen oxygen pairs.
    # By the initialization assumption, the Oxygen atom is always in the first index,
    # so the Oxygen pair is in the (0, 0) index in the last two dimensions of the 4D array with
    # dimension (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const).
    sigma_dist = inf_to_zero( sigma / distances[:,:,0,0] )
    
    # Calculate the Lennard Jones energy in accordance with the given equation
    # Sum along the first axis to get the total Lennard Jones energy in one walker.
    lennard_jones_energy = np.sum( 4*epsilon*(sigma_dist**12 - sigma_dist**6), axis = 1)
    
    
    
    # Gives the intermolecular potential energy for each walker as it is the sum of the 
    # Coulombic Energy and the Leonard Jones Energy.
    intermolecular_potential_energy = coulombic_energy + lennard_jones_energy
    
    
    
    # Return all three calculated energys which are 1D arrays of energy values for each walker
    return intermolecular_potential_energy, coulombic_energy, lennard_jones_energy

    
    
# Input: 4D array of walkers
# Output: 1D array of the sum of the intermolecular and intramolecular potential energy of each
# walker
def total_pe(x):

    # Calculate the intramolecular potential energy of each walker
    intra_potential_energy = intra_pe(x)
    
    # Calculate the intermolecular potential energy of each walker
    # only if there is more than one molecule in the system
    inter_potential_energy = 0
    if x.shape[molecule_axis] > 1:
        inter_potential_energy, coulombic, lennard_jones = inter_pe(x)
    
    
    # Return the total potential energy of the walker
    return intra_potential_energy + inter_potential_energy
    

	
	
	
# Simulation loop
# Iterates over the walkers array, propogating each walker. Deletes and replicates those 
# walkers based on their potential energies with respect to the calculated reference energy
for i in range(sim_length):

    # Calculate the Reference Energy
	# Energy is calculated based on the average of all potential energies of walkers.
	# Is adjusted by a statistical value to account for large or small walker populations.
    reference_energy[i] = np.mean( total_pe(walkers) ) \
        + (1.0 - (walkers.shape[walker_axis] / n_walkers) ) / ( 2.0*dt )
		
    # Current number of walkers
    num_walkers[i] = walkers.shape[walker_axis]
    #print(f'Reference energy: {reference_energy[i]:.8f}, Walker Shape: '+str(walkers.shape))

	
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
    
# Save the outputted walker array to a text file give that it takes a ridiculuous amount
# of time to equilibrate a water trimer system from random values
#np.save('equil_water_trimer.npy', walkers)

#####################################################################################
# Output


# Uncomment the below line to avoid graphing 
#sys.exit(0)


# Calculate the rolling average for rolling_avg time steps
ref_rolling_avg = np.zeros(sim_length)
for i in range(rolling_avg, sim_length):
    # Calculate the rolling average by looping over the past rolling_avg time steps 
    for j in range(rolling_avg):
        ref_rolling_avg[i] = ( ref_rolling_avg[i] - ( ref_rolling_avg[i] / (j+1) ) ) \
            + ( reference_energy[i-j] / (j+1) )
			
			
# Calculate the average reference convergence energy based on reference energy after the
# equilibration phase
ref_converge_num = np.mean(ref_rolling_avg[rolling_avg:])


# Create walker num array for plotting
init_walkers = (np.zeros(sim_length) + 1 )* n_walkers

# Create Zero Point Energy array for plotting
zp_energy = (np.zeros(sim_length) + 1) * ref_converge_num	

			

# Calculate the distance between one of the OH vectors
# Used in the histogram and wave function plot	
OH_positions = np.linalg.norm(walkers[:,0,0]-walkers[:,0,1], axis = molecule_axis)



# Get the range to graph the wave function in
# Step is .001, which is usually a good smooth value
x = np.arange(OH_positions.min(), OH_positions.max(), step = .001)



# Calculate the bond angle between every oxygen molecule in the water trimer.
# Get the distinct pairs of each oxygen atom
oxygen_pair_a = walkers[:,molecule_index_a]
oxygen_pair_b = walkers[:,molecule_index_b]

# Calculate the vector in between each distinct pair
oxygen_vectors = oxygen_pair_a - oxygen_pair_b

# Calculate the length of each oxygen vector
oxygen_lengths = np.linalg.norm(oxygen_vectors, axis=2)

# Find the three oxygen angles in the water trimer
oxygen_angle_1 = (180/np.pi)*np.arccos(np.sum(-oxygen_vectors[:,0]*-oxygen_vectors[:,1], \
                 axis=1) / (oxygen_lengths[:,0]*oxygen_lengths[:,1]))
                 
oxygen_angle_2 = (180/np.pi)*np.arccos(np.sum(-oxygen_vectors[:,0]*-oxygen_vectors[:,2], \
                 axis=1) / (oxygen_lengths[:,0]*oxygen_lengths[:,2]))
                 
oxygen_angle_3 = (180/np.pi)*np.arccos(np.sum(-oxygen_vectors[:,1]*-oxygen_vectors[:,2], \
                 axis=1) / (oxygen_lengths[:,1]*oxygen_lengths[:,2]))
                 

# Append all three angles into one matrix for graphing in the density histogram
oxygen_angles = np.concatenate((oxygen_angle_1, oxygen_angle_2, \
                oxygen_angle_3), axis=0)
print(oxygen_angles.shape)

	

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
plt.hist(oxygen_angle_1, bins=n_bins, density=True)
plt.xlabel('Oxygen Angle 1 in a Walker')
plt.ylabel('Density of Walkers')
plt.title('Density of Walker Oxygen Angles')
plt.legend

plt.figure(6)
plt.hist(oxygen_angle_2, bins=n_bins, density=True)
plt.xlabel('Oxygen Angle 2 in a Walker')
plt.ylabel('Density of Walkers')
plt.title('Density of Walker Oxygen Angles')
plt.legend

plt.figure(7)
plt.hist(oxygen_angle_3, bins=n_bins, density=True)
plt.xlabel('Oxygen Angle 3 in a Walker')
plt.ylabel('Density of Walkers')
plt.title('Density of Walker Oxygen Angles')
plt.legend
plt.show()

