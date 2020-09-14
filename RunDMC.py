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

sim_length = 10000

n_walkers = 2

k = 1.0

# g/mol
mass = 10

# Equilibrium position of the system. 
equilibrium_position = 5

# Mass of an electron
electron_mass = 9.10938970000e-28
# avogadro's constant
avogadro = 6.02213670000e+23

# calculate the reduced mass of the system
reduced_mass = (mass / (avogadro * electron_mass)) / 2

# get a uniform distribution about the equilibrium position
walkers = equilibrium_position + (np.random.rand(n_walkers) - 0.5)

# calculate the potential energy of a walker based on its distance from the equilibrium position of the system
def potentialEnergy(x):
    return .5 * k * (x - equilibrium_position)**2

# simulation loop
for i in range(sim_length):
    # calculate the reference energy
    # based on the average of all the potential energies of the system
    # adjusted by a statistical value to account for very large or very small populations of walkers
    reference_energy = np.mean( potential_energy(walkers) ) + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
    
    # gets a normal distribution about 0 in the range sqrt(dt/mass) of the atom
    # recall in the model of a harmonic oscillator, only one mass matters
    propogation_lengths = np.random.normal(0, np.sqrt(dt/mass), walkers.shape[0])
    # add the propogation length to the position of the current walkers
    walkers = walkers + propogation_lengths
    # calculate the new potential energies of each walker in the system
    # returns an array of floats, one per walker
    potential_energies = potential_energy(walkers)
    
    # calculates a random in range [0,1) for each walker in the system
    thresholds = np.random.rand(walkers.shape[0])
    # calculates probability of a walker to be deleted
    # notice that this is calculated for every walker in the system, regardless of the potential energy of the walker
    prob_delete = np.exp(-(potential_energies-reference_energy)*dt)
    # normalize the above probability to the probability of replication
    prob_replicate = prob_delete - 1
    
    # calculate which walkers actually have the necessary potential energies to merit deletion or replication
    to_delete = prob_delete > threshold
    to_replicate = prob_replicate > threshold

    # uses the argwhere() function to only collect the non-zero elements of the given array
    # use pointwise multiplication with the walkers array with:
    # (if the potential energy is greater than the reference energy AND the walker has probability deleted)
    # then boolean array should be a 1. Invert this and multiply with walkers to get the non-zero positions of the walkers
    # that should remain
    remain_after_delete = np.argwhere(walkers*np.invert((potential_energies > reference_energy)*to_delete))
    
    # uses the argwhere() function to collect the non-zero elements of the given array
    # (if the potential energy is less than the reference energy AND the walker should be replicated) 
    # then the value in the boolean array should be a 1. Multiplying this by walkers gives the non-zero positions of the walkers
    # that should be replicated
    replications = np.argwhere(walkers*(potential_energies<reference_energy)*to_replicate)
    
    # getting rid of this line. Notice that the no change ones already appear in the remain_after_delete array, so having this array actually replicates them
    # noChange = np.argwhere(walkers*(potential_energies==referenceEnergy))
    #walkers = np.concatenate(remain_after_delete,replications,noChange)
    
    # concatenating the remaining after deletion and replications array gives exactly the walkers that weren't deleted (most of which were replicated)
    # note that if the walker was not replicated, it still appears in the remains after deletion array, effectively encompassing
    # the else statement in the given psuedocode 
    walkers = np.concatenate(remain_after_delete, replications)


print(reducedMass)
