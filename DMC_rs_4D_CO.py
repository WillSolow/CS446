# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style

# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. In this particular implementation, a 4D
# array is utilized as a main data structure for storing the coordinates of each atom 
# in each molecule in each walker. 
# Here, we model the Carbon Monoxide (CO) bond, comparing it to our 2D array based 
# implementation

# To Run: Navigate to file in terminal directory and type 'python DMC_rs_4D_CO.py'

# Output: Graphs for the reference energy, the n-step rolling average, and the number 
# of walkers at each time step, as well as a density histogram of the walker distance
# from equilibrium and the corresponding wave function

# Imports
import numpy as np
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
sim_length = 10000

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
num_atoms = 2

# Number of molecules in each walker
# Used to initialize the walker array
num_molecules = 1



# Atomic masses of atoms in system
# Used to calculate the atomic masses in Atomic Mass Units
carbon_mass = 12.000
oxygen_mass = 15.995



# Equilibrium position of the system in atomic units
bond_length = 0.59707

# Spring constant of the atomic bond
k = 1.2216

# calculate the convergence reference energy based on the given equation.
ref_converge_num = .00494317



# Calculates the atomic masses in Atomic Mass Units
atomic_mass_carbon = carbon_mass / (avogadro * electron_mass)
atomic_mass_oxygen = oxygen_mass / (avogadro * electron_mass)

# Calculates the reduced mass of the system
# Used when graphing the wave fuction
reduced_mass = (atomic_mass_carbon+atomic_mass_oxygen) / (atomic_mass_carbon*atomic_mass_oxygen)


# Initial 4D walker array
# Returns a uniform distribution cenetered at the given bond length
# Array axes are walkers, molecules, coordinates, and atoms
walkers = bond_length + (np.random.rand(n_walkers, num_molecules, coord_const, num_atoms) - 0.25)


#######################################################################################
# Simulation


# Create arrays to store values for plotting at each time step
reference_energy = np.zeros(sim_length)
num_walkers = np.zeros(sim_length)


# Input: 4D Array of walkers
# Output: 1D Array of potential energies for each walker
# Calculates the potential energy of a walker based on the position of its atoms and 
# molecules with respect to the distances from each other
def potential_energy(x):
	# Calculate the distance between each atom
    distance = np.sqrt( (x[:,0,0,0]-x[:,0,0,1])**2 + (x[:,0,1,0]-x[:,0,1,1])**2 + \
	        (x[:,0,2,0]-x[:,0,2,1])**2)
	# Calculate the potential energy based on the distance
    return .5 * k * (distance - bond_length)**2
	
	
	
# Simulation loop
# Iterates over the walkers array, propogating each walker. Deletes and replicates those 
# walkers based on their potential energies with respect to the calculated reference energy
for i in range(sim_length):

    # calculate the Reference Energy
	# Energy is calculated based on the average of all potential energies of walkers.
	# Is adjusted by a statistical value to account for large or small walker populations.
    reference_energy[i] = np.mean( potential_energy(walkers) ) \
            + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )

	
    # Current number of walkers
    num_walkers[i] = walkers.shape[0]
    
	
	# Propogates each atom in a normal distribution about its current position
    propagate_oxygen = np.random.normal(0, np.sqrt(dt/atomic_mass_oxygen), \
            (walkers.shape[0], num_molecules, coord_const))
    propagate_carbon = np.random.normal(0, np.sqrt(dt/atomic_mass_carbon), \
            (walkers.shape[0], num_molecules, coord_const))
			
	# Adds the propogation lengths to the 4D walker array
    walkers = walkers + np.stack((propagate_oxygen, propagate_carbon), axis=-1)
	
    
	
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
distance = np.sqrt( (walkers[:,0,0,0]-walkers[:,0,0,1])**2 + (walkers[:,0,1,0]- \
        walkers[:,0,1,1])**2 + (walkers[:,0,2,0]-walkers[:,0,2,1])**2)

	

# Plot the reference energy throughout the simulation
plt.figure(1)
plt.plot(reference_energy, label= 'Reference Energy')
plt.plot(reference_converge, label='Zero Point Energy')
plt.axis([0,sim_length,.003,.007])
plt.xlabel('Simulation Iteration')
plt.ylabel('System Energy')
plt.title('Convergence of Reference Energy')
plt.legend()


# Plot the rolling average of the reference energy throughout the simulation
plt.figure(2)
plt.plot(ref_rolling_avg, label= 'Reference Energy')
plt.plot(reference_converge, label = 'Zero Point Energy')
plt.axis([0,sim_length,.0045,.0055])
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
walker_pos = distance-bond_length

# Plot a density histogram of the walkers at the final iteration of the simulation
# Line of Best Fit ought to approximate wave function
plt.figure(4)
_, bins, _ = plt.hist(walker_pos, bins=n_bins, density=True)
mu, sigma = st.norm.fit(walker_pos)
best_fit_line = st.norm.pdf(bins,mu,sigma)
plt.plot(bins,best_fit_line)
plt.xlabel('Walker Position')
plt.ylabel('Density of Walkers')
plt.title('Density of Walker Position')

plt.show()

