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
sim_length = 10

# number of initial walkers
n_walkers = 2

# spring constant
k = 1.0

# g/mol
mass = 10

# Equilibrium position of the system in atomic units
equilibrium_position = 5

# Mass of an electron
electron_mass = 9.10938970000e-28
# avogadro's constant
avogadro = 6.02213670000e+23

# calculate the reduced mass of the system
reduced_mass = (mass / (avogadro * electron_mass)) / 2

# get a uniform distribution about the equilibrium position
walkers = equilibrium_position + (np.random.rand(n_walkers) - 0.5)

print("#################### STARTING SIMULATION #########################")

# calculate the potential energy of a walker based on its distance from the equilibrium position of the system
def potential_energy(x):
    return .5 * k * (x - equilibrium_position)**2
	
# simulation loop
for i in range(sim_length):
    # print(f'wlk:{walkers}')
    # calculate the reference energy
    # based on the average of all the potential energies of the system
    # adjusted by a statistical value to account for very large or very small populations of walkers
    reference_energy = np.mean( potential_energy(walkers) ) + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
    
    # gets a normal distribution about 0 in the range sqrt(dt/mass) of the atom
    # recall in the model of a harmonic oscillator, only one mass matters
    print("prop lengths")
    propogation_lengths = np.random.normal(0, np.sqrt(dt/mass), walkers.shape[0])
    print(propogation_lengths)
    # print(f'pln:{propogation_lengths}')
    # add the propogation length to the position of the current walkers
    print("\n current walkers")
    print(walkers)
    walkers = walkers + propogation_lengths
    # print(f'wlk:{walkers}')
    # calculate the new potential energies of each walker in the system
    # returns an array of floats, one per walker
    print("\n Prop length change")
    print(walkers)
    potential_energies = potential_energy(walkers)
    print("\n PotentialEnergies")
    print(potential_energies)
    # print(f'pte:{potential_energies}')
    
    # calculates a random in range [0,1) for each walker in the system
    thresholds = np.random.rand(walkers.shape[0])
    # print(f'thr:{thresholds}')
    # calculates probability of a walker to be deleted
    # notice that this is calculated for every walker in the system, regardless of the potential energy of the walker
    prob_delete = np.exp(-(potential_energies-reference_energy)*dt)
    # print(f'p_d:{prob_delete}')
    # normalize the above probability to the probability of replication
    prob_replicate = prob_delete - 1
    # print(f'p_r:{prob_replicate}')
    
    # calculate which walkers actually have the necessary potential energies to merit deletion or replication
    to_delete = prob_delete > thresholds
    # print(f't_d:{to_delete}')
    to_replicate = prob_replicate > thresholds
    # print(f't_r:{to_replicate}')
    
    # uses the argwhere() function to only collect the non-zero elements of the given array
    # use pointwise multiplication with the walkers array with:
    # (if the potential energy is greater than the reference energy AND the walker has probability deleted)
    # then boolean array should be a 1. Invert this and multiply with walkers to get the non-zero positions of the walkers
    # that should remain_after_delete
    
    delete_walkers = walkers*np.invert( (potential_energies > reference_energy) * to_delete )
    # print(f'del:{delete_walkers}')
    remain_after_delete = walkers[delete_walkers > 0]
    print("\nWalkers not deleted: ")
    print(remain_after_delete)
    # print(f'rem:{remain_after_delete}')
    # print(f'{remain_after_delete.shape}')
     
    # Will thinks try the where() function. Check out the documentation and maybe figure out how this condition works?
    
    # uses the argwhere() function to collect the non-zero elements of the given array
    # (if the potential energy is less than the reference energy AND the walker should be replicated) 
    # then the value in the boolean array should be a 1. Multiplying this by walkers gives the non-zero positions of the walkers
    # that should be replicated
    replicate_walkers = (potential_energies < reference_energy)*to_replicate
    # print(f'repl:{replicate_walkers}')
    replications = walkers[replicate_walkers > 0]
    print("\n Replicated walkers")
    print(replications)
    # print(f'repld:{replications}')
    # print(f'{replications.shape}')
    
    # getting rid of this line. Notice that the no change ones already appear in the remain_after_delete array, so having this array actually replicates them
    # noChange = np.argwhere(walkers*(potential_energies==referenceEnergy))
    #walkers = np.concatenate(remain_after_delete,replications,noChange)
    
    # concatenating the remaining after deletion and replications array gives exactly the walkers that weren't deleted (most of which were replicated)
    # note that if the walker was not replicated, it still appears in the remains after deletion array, effectively encompassing
    # the else statement in the given psuedocode 
    walkers = np.append(remain_after_delete, replications)
    #print(walkers
    print("Reference Eneryg: " + str(reference_energy))
    print("Num walkers: " + str(walkers.shape[0]))
    print("\n\n ######################### \n\n")
    # print(f'walkers:{walkers}')


