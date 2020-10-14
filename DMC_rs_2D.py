# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style


# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. In this particular implementation, the system 
# is simplified to a 6D system, where the distance between two arbitrary atoms is modeled
# with their positions in 3D space

# To Run: Navigate to file in terminal directory and type 'python dmc_rs_2D.py'

# Output: Graphs for the reference energy, the n-step rolling average, and the number 
# of walkers at each time step, as well as a density histogram of the walker distance
# from equilibrium and the corresponding wave function

# Imports
import numpy as np
import matplotlib.pyplot as plt

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



# Initial walker array
# Returns a uniform distriubtion centered at the equilibrium 
walkers = eq_length + (np.random.rand(n_walkers, system_dimensions) - 0.5)


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
	# calculate the distance in 3D space between the two atoms in each walker
    distance = np.sqrt( (x[:,0]-x[:,3])**2 + (x[:,1]-x[:,4])**2 + (x[:,2]-x[:,5])**2)
    return .5 * k * (distance - eq_length)**2
	
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
    propogation_lengths = np.random.normal(0, np.sqrt(dt/atomic_mass), \
        (walkers.shape[0], system_dimensions))
		
	# Adds the propagation lengths to the walker array
    walkers = walkers + propogation_lengths
	
	
    
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


# Calculate the rolling average for rolling_avg time steps
ref_rolling_avg = np.zeros(sim_length)
for i in range(sim_length):
	# if i less than rolling_avg, cannot calculate rolling average over the last 
    # rolling_avg time steps
    if i < rolling_avg:
        for j in range(i):
            ref_rolling_avg[i] = ( ref_rolling_avg[i] - ( ref_rolling_avg[i] / (j+1) ) ) \
                    + (reference_energy[i-j] / (j+1) )
    else: 
        # calculate the rolling average by looping over the past rolling_avg time steps 
        for j in range(rolling_avg):
            ref_rolling_avg[i] = ( ref_rolling_avg[i] - ( ref_rolling_avg[i] / (j+1) ) ) \
                    + ( reference_energy[i-j] / (j+1) )

					
# Calculate the distance between the two atoms to be used in the histogram	
distance = np.sqrt( (walkers[:,0]-walkers[:,3])**2 + (walkers[:,1]-walkers[:,4])**2 + (walkers[:,2]-walkers[:,5])**2)

# Calculate the walker distance from the equilibrium bond length
# Negative is shorter than the bond length, positive is longer than bond length
walker_pos = distance-eq_length	


# Plot the reference energy throughout the simulation
plt.figure(1)
plt.plot(reference_energy, label= 'Reference Energy')
plt.plot(reference_converge, label= 'Zero Point Energy')
plt.axis([0,sim_length,.003,.007])
plt.xlabel('Simulation Iteration')
plt.ylabel('System Energy')
plt.title('Convergence of Reference Energy')
plt.legend()

# Plot the rolling average of the reference energy throughout the simulation
plt.figure(2)
plt.plot(ref_rolling_avg, label= 'Reference Energy')
plt.plot(reference_converge, label = 'Zero Point Energy')
plt.axis([0,sim_length,.004,.006])
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

# Plot a density histogram of walkers at final iteration
plt.figure(4)
plt.hist(walker_pos, bins=n_bins, density=True)
plt.xlabel('Walker Position')
plt.ylabel('Number of Walkers')
plt.title('Walkers Final Position')


plt.show()

