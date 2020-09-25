# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Goal: approximate Schrodinger Equation for 2 or more atoms
# This implementation builds off the 6D system for the CO bond with the goal of being 
# generalizable to multi-molecule systems. The main change is that the walkers array is now
# 3 dimensional to easily keep track of multiple atoms

# Imports
import numpy as np
import matplotlib.pyplot as plt

####################################################################################
# Initial Constants

# Time step
dt = 10.0

# simulation length
sim_length = 1000

# number of time steps for rolling average calculation
n = 1000


# number of initial walkers
n_walkers = 1000

# Equilibrium position of the system in atomic units
bond_length = 0.59707

# To be used in ndarray of atom masses for each walker
# mass of carbon
carbon_mass = 12.000
# mass of oxygen
oxygen_mass = 15.995

# spring constant
k = 1.2216

# Mass of an electron
electron_mass = 9.10938970000e-28
# avogadro's constant
avogadro = 6.02213670000e+23

# calculate the atomic mass of the system
atomic_mass_carbon = carbon_mass / (avogadro * electron_mass)
atomic_mass_oxygen = oxygen_mass / (avogadro * electron_mass)

# number of atoms in each molecule system
# used in generating the walkers in the initial walkers array
num_atoms = 2

# the number of molecules in each walker. Is used to generate the correct 4D array size
# can be generalized to systems where not all molecules are identical
# is currently implemented under the assumption that all molecules are the same (in this case water molecules)
num_molecules = 1

# required user input. Records the masses of atoms in the order in which they appear in a walker
# Used to generate the propogation lengths in each walker
atom_walker_masses = np.array([atomic_mass_oxygen, atomic_mass_carbon])



# cartesian coodinates constant. Always 3, used for clarity purposes
coord_const = 3


# get a uniform distribution about the equilibrium position for random walkers
# Creates a 4D array with dimesions of walkers by molecules by coord constant by num atoms
walkers = bond_length + (np.random.rand(n_walkers, num_molecules, coord_const, num_atoms) - 0.5)

# calculate the convergence reference energy based on the given equation.
ref_converge_num = .00494317


# create reference energy array for plotting
reference_energy = np.zeros(sim_length)
reference_converge = (np.zeros(sim_length) + 1) * ref_converge_num

# create walker num array for plotting
num_walkers = np.zeros(sim_length)
init_walkers = (np.zeros(sim_length) + 1 )* n_walkers

#######################################################################################
# Simulation

# calculate the potential energy of a walker based on the position of its atoms or molecules inside of it and their distances from each other
def potential_energy(x):
	# calculate the distance in 3D space between the two atoms in each walker
    distance = np.sqrt( (x[:,0,0,0]-x[:,0,0,1])**2 + (x[:,0,1,0]-x[:,0,1,1])**2 + (x[:,0,2,0]-x[:,0,2,1])**2)
    return .5 * k * (distance - bond_length)**2
	
# simulation loop
for i in range(sim_length):
    # calculate the reference energy
    # based on the average of all the potential energies of the system
    # adjusted by a statistical value to account for very large 
    # or very small populations of walkers
    reference_energy[i] = np.mean( potential_energy(walkers) ) \
            + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )

	
    # collect the current number of walkers for plotting purposes
    num_walkers[i] = walkers.shape[0]
    
	
	
    propogate_oxygen = np.random.normal(0, np.sqrt(dt/atom_walker_masses[0]), \
            (walkers.shape[0], num_molecules, coord_const))
    propogate_carbon = np.random.normal(0, np.sqrt(dt/atom_walker_masses[1]), \
            (walkers.shape[0], num_molecules, coord_const))
    walkers = walkers + np.stack((propogate_oxygen, propogate_carbon), axis=-1)
	
    
    # calculate the new potential energies of each walker in the system
    # returns an array of floats, one per walker
    potential_energies = potential_energy(walkers)

    
    # picks from a uniform distribution in range [0,1) for each walker in the system
    # used to calculate the chance of a walker being deleted or replicated 
    thresholds = np.random.rand(walkers.shape[0])
	
	
	
    # calculates probability of a walker to be deleted
    # notice that this is calculated for every walker in the system
    # regardless of the potential energy of the walker
    # Notice that this is actually the probability that a walker surives
    prob_delete = np.exp(-(potential_energies-reference_energy[i])*dt)

    # Takes prob_delete and normalizes it to the probability of replication
    # Notice that in the model these differ by -1
    prob_replicate = prob_delete - 1

	
	
    # calculate which walkers actually have the necessary potential energies 
    # to merit deletion or replication
    # these two arrays are not mutally exclusive, but below they are pointwise AND 
    # with mutually exclusive energy statements to ensure that no walker will get
    # both replicated and deleted at the same time
    to_delete = prob_delete < thresholds
    to_replicate = prob_replicate > thresholds
    
	
	
    # use pointwise multiplication with the walkers array with:
    # (if the potential energy is greater than the reference energy 
    # AND the walker has probability deleted)
    # then boolean array should be a 1. Invert this and multiply with walkers 
    # to get the non-zero positions of the walkers
    # that should remain_after_delete
    delete_walkers = np.invert( (potential_energies > reference_energy[i]) * to_delete )
	
    # Truncates a shallow copy of the walkers array to store 
    # all the walkers that were not deleted
    # delete_walkers > 0 is an ndarray of booleans
    remain_after_delete = walkers[delete_walkers > 0]

	
    # (if the potential energy is less than the reference energy 
    # AND the walker should be replicated) 
    # then the value in the boolean array should be a 1. 
    # Multiplying this by walkers gives the non-zero positions of the walkers
    # that should be replicated
    replicate_walkers = (potential_energies < reference_energy[i])*to_replicate
	
	
    # Truncates a shallow copy of the walkres array to store only 
    # the walkers to be replicated
    # repiclate_walkres > 0 is an ndarry of booleans
    replications = walkers[replicate_walkers > 0]
	
	
    # concatenating the remaining after deletion and replications array 
    # gives exactly the walkers that weren't deleted (most of which were replicated)
    # note that if the walker was not replicated, it still appears in the 
    # remains after deletion array, effectively encompassing
    # the case where the threshold is equal to the probability of deletion or replication
    # However, due to the stochastic nature of the system, this is unlikely to happen
    walkers = np.append(remain_after_delete, replications, axis=0)


#####################################################################################
# Output
	
# calculate the rolling average for n time steps
ref_rolling_avg = np.zeros(sim_length)
for i in range(sim_length):
    # if i less than n, cannot calculate rolling average over the last n time steps
    if i < n:
        for j in range(i):
            ref_rolling_avg[i] = ( ref_rolling_avg[i] - ( ref_rolling_avg[i] / (j+1) ) ) \
                + (reference_energy[i-j] / (j+1) )
    else: 
        # calculate the rolling average by looping over the past n time steps 
        for j in range(n):
            ref_rolling_avg[i] = ( ref_rolling_avg[i] - ( ref_rolling_avg[i] / (j+1) ) ) \
                + ( reference_energy[i-j] / (j+1) )

			

# calculate the distance between the two atoms to be used in the histogram	
distance = np.sqrt( (walkers[:,0,0,0]-walkers[:,0,0,1])**2 + (walkers[:,0,1,0]- \
        walkers[:,0,1,1])**2 + (walkers[:,0,2,0]-walkers[:,0,2,1])**2)



# plotting reference energy converging to zero-point energy
plt.figure(1)
plt.plot(reference_energy, label= 'Reference Energy')
plt.plot(reference_converge, label= 'Zero Point Energy')
plt.axis([0,sim_length,0,.01])
plt.xlabel('Simulation Iteration')
plt.ylabel('System Energy')
plt.title('Convergence of Reference Energy')
plt.legend()

# plotting the rolling average of the reference energy converging to zero-point energy
plt.figure(2)
plt.plot(ref_rolling_avg, label= 'Reference Energy')
plt.plot(reference_converge, label = 'Zero Point Energy')
plt.axis([0,sim_length,0,.01])
plt.xlabel('Simulation Iteration')
plt.ylabel('System Energy')
plt.title(str(n) + ' Step Rolling Average for Reference Energy')
plt.legend()

# plotting number of walkers over time
plt.figure(3)
plt.plot(num_walkers, label='Current Walkers')
plt.plot(init_walkers, label='Initial Walkers')
plt.xlabel('Simulation Iteration')
plt.ylabel('Number of Walkers')
plt.title('Number of Walkers Over Time')
plt.legend()

# plot histogram of walkers at final iteration
plt.figure(4)
plt.hist(distance)
plt.xlabel('Walker Position')
plt.ylabel('Number of Walkers')
plt.title('Walkers Final Position')
plt.show()

