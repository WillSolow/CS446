# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Goal: approximate Schrodinger Equation for 2 or more atoms
# This implementation uses a truly 1-dimensional simulation for the distance between two walkers, and is modeled after a harmonic oscillator

# Imports
import numpy as np

# Initial Constants
dt = 10

# simulation length
sim_length = 1000

# number of initial walkers
n_walkers = 1000

# spring constant
k = 1.0

# g/mol
mass_of_atom = 10

# Equilibrium position of the system in atomic units
equilibrium_position = 5

# Mass of an electron
electron_mass = 9.10938970000e-28
# avogadro's constant
avogadro = 6.02213670000e+23

# calculate the reduced mass of the system
reduced_mass = (mass_of_atom / (avogadro * electron_mass)) / 2

# get a uniform distribution about the equilibrium position
walkers = equilibrium_position + (np.random.rand(n_walkers) - 0.5)

print("#################### STARTING SIMULATION #########################")

# calculate the potential energy of a walker based on its distance from the equilibrium position of the system
def potential_energy(x):
    return .5 * k * (x - equilibrium_position)**2
	
# simulation loop
for i in range(sim_length):
    # calculate the reference energy
    # based on the average of all the potential energies of the system
    # adjusted by a statistical value to account for very large or very small populations of walkers
    reference_energy = np.mean( potential_energy(walkers) ) + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
    
	
	
    # gets a normal distribution about 0 in the range sqrt(dt/mass) of the atom
    # recall in the model of a harmonic oscillator, only the reduced mass matters
	# add these randomized propogation lengths to each walker
    propogation_lengths = np.random.normal(0, np.sqrt(dt/reduced_mass), walkers.shape[0])
    walkers = walkers + propogation_lengths
	
	
    
    # calculate the new potential energies of each walker in the system
    # returns an array of floats, one per walker
    potential_energies = potential_energy(walkers)

    
    # calculates a random in range [0,1) for each walker in the system
	# used to calculate the chance of a walker being deleted or replicated 
    thresholds = np.random.rand(walkers.shape[0])
	
	
	
    # calculates probability of a walker to be deleted
    # notice that this is calculated for every walker in the system
	# regardless of the potential energy of the walker
	# Notice that this is actually the probability that a walker surives
    prob_delete = np.exp(-(potential_energies-reference_energy)*dt)

	# Takes prob_delete and normalizes it to the probability of replication
    # Notice that in the model these differ by -1
    prob_replicate = prob_delete - 1

	
	
    # calculate which walkers actually have the necessary potential energies to merit deletion or 
	# Notice that to_delete and to_replicate are equal, but for clarity purposes we separate them
    to_delete = prob_delete > thresholds
    to_replicate = prob_replicate > thresholds
    
	
	
    # use pointwise multiplication with the walkers array with:
    # (if the potential energy is greater than the reference energy AND the walker has probability deleted)
    # then boolean array should be a 1. Invert this and multiply with walkers to get the non-zero positions of the walkers
    # that should remain_after_delete
    delete_walkers = walkers*np.invert( (potential_energies > reference_energy) * to_delete )
	
	
	# Truncates a shallow copy of the walkers array to store all the walkers that were not deleted
	# delete_walkers > 0 is an ndarray of booleans
    remain_after_delete = walkers[delete_walkers > 0]

	
	
    # (if the potential energy is less than the reference energy AND the walker should be replicated) 
    # then the value in the boolean array should be a 1. Multiplying this by walkers gives the non-zero positions of the walkers
    # that should be replicated
    replicate_walkers = (potential_energies < reference_energy)*to_replicate
    # print(f'repl:{replicate_walkers}')
    replications = walkers[replicate_walkers > 0]

	
	
    # concatenating the remaining after deletion and replications array gives exactly the walkers that weren't deleted (most of which were replicated)
    # note that if the walker was not replicated, it still appears in the remains after deletion array, effectively encompassing
    # the case where the threshold is equal to the probability of deletion or replication
	# However, due to the stochastic nature of the system, this is unlikely to happen
    walkers = np.append(remain_after_delete, replications)

	
	# print refernce energy and walkers at each time step
	# TODO Graph using Matlib
    print("Reference Energy: " + str(reference_energy))
    print("Num walkers: " + str(walkers.shape[0]))
    print("\n\n ######################### \n\n")

h=1
convergence_num = .5*h*np.sqrt(k/reduced_mass)
print("Convergence num" + str(convergence_num))


