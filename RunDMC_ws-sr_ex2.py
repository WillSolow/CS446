# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Goal: approximate Schrodinger Equation for 2 or more atoms
# This implementation uses a modified 1 dimensional system, now in 6 dimensions,
#to simulate 2 atoms in 3D space. This has been updated from the first #simulation to account for the facct that each walker is now 6 

# Imports
import numpy as np
import matplotlib.pyplot as plt

#######################################################################################
# Initial Constants

# Time step
dt = 10

# simulation length
sim_length = 10000

# number of time steps for rolling average calculation
n = 1000


# number of initial walkers
n_walkers = 1000

# Equilibrium position of the system in atomic units
equilibrium_position = 5

# Dimensions in system. Two walkers is 6 dimensional as it takes 6 coordinates to
# simulate the walker in 3 space
system_dimensions = 6



# spring constant
k = 1.0

# Mass of an electron
electron_mass = 9.10938970000e-28
# avogadro's constant
avogadro = 6.02213670000e+23

# g/mol
mass_of_atom = 10


# calculate the reduced mass of the system
atomic_mass = (mass_of_atom / (avogadro * electron_mass) )
reduced_mass = (atomic_mass * atomic_mass) / (atomic_mass + atomic_mass)

# get a uniform distribution about the equilibrium position
walkers = equilibrium_position + (np.random.rand(n_walkers, system_dimensions) - 0.5)

# constant for covergence of reference energy
h = 1

# calculate the convergence reference energy based on the given equation.
ref_converge_num = .5*h*np.sqrt(k/reduced_mass)


# create reference energy array for plotting
reference_energy = np.zeros(sim_length)
reference_converge = (np.zeros(sim_length) + 1) * ref_converge_num

# create walker num array for plotting
num_walkers = np.zeros(sim_length)
init_walkers = (np.zeros(sim_length) + 1 )* n_walkers

#######################################################################################
# Simulation

# calculate the potential energy of a walker based on its distance from the equilibrium position of the system
def potential_energy(x):
    # calculate the distance in 3D space between the two atoms in each walker
    distance = np.sqrt( (x[:,0]-x[:,3])**2 + (x[:,1]-x[:,4])**2 + (x[:,2]-x[:,5])**2)
    return .5 * k * (distance - equilibrium_position)**2
	
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
    
	
	
    # picks from a normal distribution about 0 in the range sqrt(dt/mass) of the atom
    # recall in the model of a harmonic oscillator, only the reduced mass matters
    # add these randomized propogation lengths to each walker
    propogation_lengths = np.random.normal(0, np.sqrt(dt/atomic_mass), \
        (walkers.shape[0], system_dimensions))
    walkers = walkers + propogation_lengths
	
	
    
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

	
	
    # calculate which walkers actually have the necessary potential energies to merit 
    # deletion or replication
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
	
	
	
    # Truncates a shallow copy of the walkers array to store all the walkers 
    # that were not deleted
    # delete_walkers > 0 is an ndarray of booleans
    remain_after_delete = walkers[delete_walkers > 0]
	
	
    # (if the potential energy is less than the reference energy 
    # AND the walker should be replicated) 
    # then the value in the boolean array should be a 1. Multiplying this by walkers 
    # gives the non-zero positions of the walkers
    # that should be replicated
    replicate_walkers = (potential_energies < reference_energy[i]) * to_replicate
	
	
    # Truncates a shallow copy of the walkres array to store only 
    # the walkers to be replicated
    # repiclate_walkres > 0 is an ndarray of booleans
    replications = walkers[replicate_walkers > 0]
	
	
    # concatenating the remaining after deletion and replications array gives exactly 
    # the walkers that weren't deleted (most of which were replicated)
    # note that if the walker was not replicated, it still appears in the 
    # remains after deletion array, effectively encompassing
    # the case where the threshold is equal to the probability of deletion or replication
    # However, due to the stochastic nature of the system, this is unlikely to happen
    walkers = np.append(remain_after_delete, replications, axis=0)

#########################################################################################
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
distance = np.sqrt( (walkers[:,1]-walkers[:,3])**2 + (walkers[:,1]-walkers[:,4])**2 + (walkers[:,2]-walkers[:,5])**2)



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

