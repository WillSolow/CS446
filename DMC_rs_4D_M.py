# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style

# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. In this particular implementation, a 4D 
# masked array is utilized as a main data structure for storing the coordinates of each 
# atom in each molecule in each walker. 

# To Run: Navigate to file in terminal directory and type 'python dmc_rs_4D_M.py'

# Output: Graphs for the reference energy, the n-step rolling average, and the number 
# of walkers at each time step, as well as a density histogram of the walker distance
# from equilibrium and the corresponding wave function

# Imports
import numpy as np
import numpy.ma as ma
import scipy.stats as st
import matplotlib.pyplot as plt

###################################################################################
# Scientific Constants


# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e+23

# Number of coordinates
# Always 3, used for clarity
coord_const = 3

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

# Number of time steps for rolling average calculation
rolling_avg = 1000

# Number of bins for histogram. More bins is more precise
n_bins = 50


####################################################################################
# Molecule Model Constants


# Number of atoms in each molecule of the system
# Used to initilize the walker array
num_atoms = 3

# Number of molecules in each walker
# Used to initialize the walker array
num_molecules = 1



# Atomic masses of atoms in system
# Used to calculate the atomic masses in Atomic Mass Units
oxygen_mass = 15.995
hydrogen_mass = 1.008
HOH_bond_angle = 112.0



# Equilibrium position of OH Bond
eq_bond_length = 1.8897

# Equilibrium angle of the HOH bond in radians
eq_bond_angle = HOH_bond_angle * np.pi/180

# Spring constant of the OH Bond
kOH = 6.0275

# Spring constant of the HOH bond angle
kA = 0.1209

# calculate the convergence reference energy based on the given equation.
ref_converge_num = .00494317



# Returns an array of masses in Atomic Mass Units, Oxygen is first followed by both 
# Hydrogens
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)


# Calculates the reduced mass of the system
# Used when graphing the wave fuction
reduced_mass = np.sum(atomic_masses) / (np.prod(atomic_masses))


# TODO - figure out how to generate water molecules in the right shape
# Initial 4D walker array
# Returns a uniform distribution cenetered at the given bond length
# Array axes are walkers, molecules, coordinates, and atoms
walkers = eq_bond_length + (np.random.rand(n_walkers, num_molecules, num_atoms, coord_const) - 0.25)


#######################################################################################
# Simulation


# Create arrays to store values for plotting at each time step
reference_energy = np.zeros(sim_length)
num_walkers = np.zeros(sim_length)


# Input: 4D Array of walkers
# Output: 1D Array of potential energies for each walker
# Calculates the potential energy of a walker based on the distance of bond lengths and 
# bond angles from equilibrium
# Currently assumes that there is no interaction between molecules in a walker
def potential_energy(x):
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

	
	# Propogates each coordinate of each atom in each molecule of each walker within a normal
	# distribution given by the atomic mass of each atom.
    # Returns a 4D array in the shape of walkers with the standard deviation depending on the
    # atomic mass of each atom	
    propogations = np.random.normal(0, np.sqrt(dt/np.transpose(np.tile(atomic_masses, \
	    (walkers.shape[0], num_molecules, coord_const, 1)), (0, 1, 3, 2))))
		
	# Adds the propogation lengths to the 4D walker array
    walkers = walkers + propogations
	
    
	
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
    walkers = np.append(walkers_after_delete, walkers_after_replication, axis=0)


#####################################################################################
# Output
	
# Create reference energy array for plotting
reference_converge = (np.zeros(sim_length) + 1) * ref_converge_num

# Create walker num array for plotting
init_walkers = (np.zeros(sim_length) + 1 )* n_walkers	
	
# Calculate the rolling average for n time steps
ref_rolling_avg = np.zeros(sim_length)
for i in range(sim_length):
    # If i less than rolling_avg, cannot calculate rolling average over the last 
	# rolling_avg time steps
	# Instead, calculate average over the first i time steps
    if i < rolling_avg:
        for j in range(i):
            ref_rolling_avg[i] = ( ref_rolling_avg[i] - ( ref_rolling_avg[i] / (j+1) ) ) \
                + (reference_energy[i-j] / (j+1) )
    else: 
        # Calculate the rolling average by looping over the past n time steps 
        for j in range(rolling_avg):
            ref_rolling_avg[i] = ( ref_rolling_avg[i] - ( ref_rolling_avg[i] / (j+1) ) ) \
                + ( reference_energy[i-j] / (j+1) )

			

# Calculate the distance between the atoms in the system
# Used in the histogram and wave function plot	
#distance = np.sqrt( (walkers[:,0,0,0]-walkers[:,0,0,1])**2 + (walkers[:,0,1,0]- \
        #walkers[:,0,1,1])**2 + (walkers[:,0,2,0]-walkers[:,0,2,1])**2)

	

# Plot the reference energy throughout the simulation
plt.figure(1)
plt.plot(reference_energy, label= 'Reference Energy')
plt.plot(reference_converge, label='Zero Point Energy')
plt.axis([0,sim_length,.04,.08])
plt.xlabel('Simulation Iteration')
plt.ylabel('System Energy')
plt.title('Convergence of Reference Energy')
plt.legend()


# Plot the rolling average of the reference energy throughout the simulation
for i in range(ref_rolling_avg.shape[0]):
    print('Timestep ' + str(i)+ ': ' + str(ref_rolling_avg[i]))

plt.figure(2)
plt.plot(ref_rolling_avg, label= 'Reference Energy')
plt.plot(reference_converge, label = 'Zero Point Energy')
plt.axis([0,sim_length,.05,.07])
plt.xlabel('Simulation Iteration')
plt.ylabel('System Energy')
plt.title(str(rolling_avg) + ' Step Rolling Average for Reference Energy')
plt.legend()


# Plot the number of walkers throughout the simulation
plt.figure(3)
plt.plot(num_walkers, label='Current Walkers')
plt.plot(init_walkers, label='Initial Walkers')
plt.xlabel('Simulation Iteration')
plt.ylabel('Number of Walkers')
plt.title('Number of Walkers Over Time')
plt.legend()

# Calculate the walker distance from the equilibrium bond length
# Negative is shorter than the bond length, positive is longer than bond length
#walker_pos = distance-eq_bond_length

# Plot a density histogram of the walkers at the final iteration of the simulation
# Line of Best Fit ought to approximate wave function
#plt.figure(4)
#_, bins, _ = plt.hist(walker_pos, bins=n_bins, density=True)
#mu, sigma = st.norm.fit(walker_pos)
#best_fit_line = st.norm.pdf(bins,mu,sigma)
#plt.plot(bins,best_fit_line)
#plt.xlabel('Walker Position')
#plt.ylabel('Density of Walkers')
#plt.title('Density of Walker Position')

plt.show()

