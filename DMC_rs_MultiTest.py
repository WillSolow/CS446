# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 10/27/20

# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. This file supports the main simulation loop 
# as a function so that the user can run multiple tests and compute useful data.

# To Run: Navigate to file in terminal directory and type 'python dmc_rs_MultiTest.py'

# Output: Graphs for the reference energy, the n-step rolling average, and the number 
# of walkers at each time step, as well as a density histogram of the walker distance
# from equilibrium and the corresponding wave function

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import time
import sys


###################################################################################
# Scientific Constants


# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e23


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
seed = np.random.randint(100000)
# Set the seed manually for replicability purposes over multiple simulations
# seed = 81716

# Set the seed for the pseudo-random number generator. 
np.random.seed(seed)
print('Seed used: ' + str(seed))

####################################################################################
# Simulation Loop Constants


# Time step 
# Used to calculate the distance an atom moves in a time step
# Smaller time step means less movement in a given time step
dt = .5

# Length of the equilibration phase in time steps. The below data is for the water molecule
# If dt = 1.0, equilibration phase should be greater than 1500
# If dt = 0.5, equilibration phase should be greater than 2000
# If dt = 0.1, equilibration phase should be greater than 5000
equilibration_phase = 1500

# Number of time steps in a simulation.
# Simulation length should be at least five times the length of the equilibration phase
sim_length = 10000

# Number of initial walkers
n_walkers = 1000

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
num_molecules = 1



# Atomic masses of atoms in system
# Used to calculate the atomic masses in Atomic Mass Units
oxygen_mass = 15.994915
hydrogen_mass = 1.007825
HOH_bond_angle = 112.0



# Equilibrium length of OH Bond
eq_bond_length = 1.889727

# Equilibrium angle of the HOH bond in radians
eq_bond_angle = HOH_bond_angle * np.pi/180

# Spring constant of the OH Bond
kOH = 6.027540

# Spring constant of the HOH bond angle
kA = 0.120954



# Returns an array of masses in Atomic Mass Units, Oxygen is first followed by both 
# Hydrogens
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)


#######################################################################################
# Potential Energy Functions

# Input: 4D Array of walkers
# Output: 1D Array of potential energies for each walker
# Calculates the potential energy of a walker based on the distance of bond lengths and 
# bond angles from equilibrium
# Currently assumes that there is no interaction between molecules in a walker
def pe_SR(x):
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


# Professor Madison's potential energy function
def pe_M(positions):
    #This first finds the sum of the intramolecular energy (intRA) (energy due to atomic positions of atoms IN a molecule) and 
    #sums them together.
    #Second, it finds the intermolecular energy (intER) due to waters interacting with each other.

    #positions is the postitions of all of the atoms in all of the walkers.
    #positions is structured [nWalkers, nWaters, nAtoms, 3]
    #nWalkers is how many walkers your are calculation the PE for
    #nWater is how many water molecules are in each walker (a constant)
    #nAtoms is how many atoms are in water (always 3)
    #3 is for the number of cartesian coordinates

    (nWalkers,nWaters,nAtoms,nCartesian)=positions.shape

    intRAmolecularEnergy=np.zeros(nWalkers)
    for iWat in range(nWaters):
        #print("For ",iWat," the energy is ",PotentialEnergySingleWater(positions[:,iWat]))
        intRAmolecularEnergy=intRAmolecularEnergy+PotentialEnergySingleWater(positions[:,iWat])
        #print("current sum: ",intRAmolecularEnergy)
    intERmolecularEnergy=np.zeros(nWalkers)
    for iWat in range(nWaters):
        for jWat in range(iWat,nWaters): 
            intERmolecularEnergy=intERmolecularEnergy+PotentialEnergyTwoWaters(positions[:,iWat],positions[:,jWat])

    #print("Sum IntRAmolecular Energy: ",intRAmolecularEnergy)
    potentialEnergy=intRAmolecularEnergy+intERmolecularEnergy
    return potentialEnergy  
    
def PotentialEnergyTwoWaters(water1pos, water2pos):
    #NOT YET IMPLEMENTED!! Until implemented this will return zeros which correspond to the waters not interacting 
    #with each other
    (nWalkers,nAtoms,nCartesian)=water1pos.shape
    return np.zeros(nWalkers)
    
def PotentialEnergySingleWater(OHHpositions):
    #This calculates the potential energy of a single water molecule.  A walker might be made up of many
    #water molecules.  You'd calculate the energy of each discrete water molecule and sum those energies together.
    #This is done in the function PotentialEnergyManyWaters, above.

    #The structure of OHHpositions is [nWalkers,nAtoms,3]
    #where nWalkers is how many walkers you are calculating the PE for
    #nAtoms is the number of atoms in water (always 3)
    #and the last is 3 for the number of cartesian coordinates

    #The first atom is assumed to be Oxygen
    #The second and third atoms are assumed to be the two Hydrogens

    #The potential energy of water is the sum of the PE from the two OH 
    #bond lengths and the H-O-H bond angle

    #Bondlength #1
    rOH1=np.linalg.norm(OHHpositions[:,0,:]-OHHpositions[:,1,:],axis=1)
    #Energy due to Bond length #1
    rOHeq=1.0 /0.529177 #equilibrium bond length in atomic units of distance
    kb= 1059.162 *(1.0/0.529177)**2 * (4.184/2625.5)# spring constant in atomic units of energy per (atomic units of distance)^2

    potROH1=kb/2.0 *(rOH1-rOHeq)**2
    
    #print('equilibrium distance: ',rOHeq)
    #print("rOH1: ",rOH1, " atomic units of distance")
    #print("potROH1: ", potROH1)

    #Bondlength #2
    rOH2=np.linalg.norm(OHHpositions[:,0]-OHHpositions[:,2],axis=1)
    #Energy due to Bond length #2
    #we reuse rOHeq and kb for potROH2 because they are the same type of bond (an OH bond)
    potROH2=kb/2.0 *(rOH2-rOHeq)**2

    #print("rOH2: ",rOH2, " atomic units of distance")
    #print("potROH2: ", potROH2)
    #angle of H-O-H bond angle (O is the vertex) determined using cos^-1 which is (in TeX):
    #\theta = \arccos \left( \frac{\vec{OH_1}\cdot \vec{OH_2}}{ \|\vec{OH_1}\| \, \|\vec{OH_2}\|}\right)
    #as far as I know, np.arccos cannot handle my
    aHOH=[]
    for walkerPos in OHHpositions:
        vecOH_1=walkerPos[0]-walkerPos[1]
        vecOH_2=walkerPos[2]-walkerPos[0]
        cosAngle=np.dot(vecOH_1,vecOH_2)/(np.linalg.norm(vecOH_1)*np.linalg.norm(vecOH_2))
        aHOH.append(np.arccos(cosAngle))

    aHOH=np.array(aHOH)
    ka=75.90*(4.184/2625.5) #spring constant in atomic units of energy per (rad)^2
    aHOHeq= 112.0 * np.pi/180.0 #equilibrium HOH bond angle in radians
    potAHOH=ka/2.0*(aHOH-aHOHeq)**2

    #print('equilibrium bond angle: ',aHOHeq)   
    #print("aHOH: ",aHOH, " radians")
    #print("aHOH: ",aHOH*180.0/np.pi, " degrees")   

    #print("pot: ", potAHOH)

    potentialEnergy=potROH1+potROH2+potAHOH
    #print("intra molecular potential ",potentialEnergy)
    return potentialEnergy

    
    
#######################################################################################
# Simulation Functions


# Output: an equilibrated array of walkers based on random initial positions
# Runs the simulation as fast as possible to equilibrate the walkers based on the length of 
# the equilibration phase
def equilibrate_walkers():

    # Initial 4D walker array
    # Returns a uniform distribution cenetered at the given bond length
    # Array axes are walkers, molecules, coordinates, and atoms
    walkers = (np.random.rand(n_walkers, num_molecules, atomic_masses.shape[0], \
        coord_const) - .5) 
    
    
    # Simulation loop
    # Iterates over the walkers array, propogating each walker. Deletes and replicates those 
    # walkers based on their potential energies with respect to the calculated reference energy
    for i in range(equilibration_phase):
    
        # Calculate the Reference Energy
        # Energy is calculated based on the average of all potential energies of walkers.
        # Is adjusted by a statistical value to account for large or small walker populations.
        reference_energy = np.mean( pe_SR(walkers) ) \
            + (1.0 - (walkers.shape[walker_axis] / n_walkers) ) / ( 2.0*dt )
       

    
        # Propagates each coordinate of each atom in each molecule of each walker within a 
        # normal distribution given by the atomic mass of each atom.
        # Returns a 4D array in the shape of walkers with the standard deviation depending 
        # on the atomic mass of each atom   
        propagations = np.random.normal(0, np.sqrt(dt/np.transpose(np.tile(atomic_masses, \
           (walkers.shape[walker_axis], num_molecules, coord_const, 1)), (walker_axis, \
            molecule_axis, coord_axis, atom_axis))))
        
        # Adds the propagation lengths to the 4D walker array
        walkers = walkers + propagations
    
    
    
        # Calculates the potential energy of each walker in the system
        potential_energies = pe_SR(walkers)

    
    
        # Gives a uniform distribution in the range [0,1) associated with each walker
        # in the system
        # Used to calculate the chance that a walker is deleted or replicated   
        thresholds = np.random.rand(walkers.shape[walker_axis])
    
    
        # Calculates a probability for each walker that it is deleted
        # This is actually the probability that a walker is not deleted
        prob_delete = np.exp(-(potential_energies-reference_energy)*dt)

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
        walkers_to_remain = np.invert( (potential_energies > reference_energy) * to_delete )
    
        # Returns the walkers that remain after deletion
        walkers_after_delete = walkers[walkers_to_remain]

    
    
        # Gives a boolean array of indices of the walkers that are replicated
        # Calculates if a walker is replicated by if its potential energy is less than
        # the reference energy and if its threshold is below the prob_replicate threshold.
        walkers_to_replicate = (potential_energies < reference_energy) * to_replicate
    
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
        walkers = np.append(walkers_after_delete, walkers_after_replication, axis=walker_axis)
        
    return walkers
    

# Input: Boolean - if true used SR vectorized PE, if false use Madison PE function
# Output: Time taken to complete computation
# Runs a simulation with vectorized techniques
def sim_loop(vec_PE, init_walkers):
    
    # Create array to store the number of walkers at each time step
    num_walkers = np.zeros(sim_length)
    
    # Create array to store the reference energy at each time step
    reference_energy = np.zeros(sim_length)
    
    # Set the equilibrated walkers array to 
    walkers = init_walkers
    
    # Calculate the start time of the program to be used when calculating efficiency
    start_time = time.time()
    
    
    # Simulation loop
    # Iterates over the walkers array, propogating each walker. Deletes and replicates those 
    # walkers based on their potential energies with respect to the calculated reference energy
    for i in range(sim_length):
    
        # Calculate the Reference Energy
        # Energy is calculated based on the average of all potential energies of walkers.
        # Is adjusted by a statistical value to account for large or small walker populations.
        if vec_PE:
            reference_energy[i] = np.mean( pe_SR(walkers) ) \
                + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
        else:
            reference_energy[i] = np.mean( pe_M(walkers) ) \
                + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
        
        # Current number of walkers
        num_walkers[i] = walkers.shape[0]

    
        # Propagates each coordinate of each atom in each molecule of each walker within a normal
        # distribution given by the atomic mass of each atom.
        # Returns a 4D array in the shape of walkers with the standard deviation depending on the
        # atomic mass of each atom  
        propagations = np.random.normal(0, np.sqrt(dt/np.transpose(np.tile(atomic_masses, \
           (walkers.shape[walker_axis], num_molecules, coord_const, 1)), (walker_axis, \
           molecule_axis, coord_axis, atom_axis))))
        
        
        # Adds the propagation lengths to the 4D walker array
        walkers = walkers + propagations
    
    
    
        # Calculates the potential energy of each walker in the system
        if vec_PE:
            potential_energies = pe_SR(walkers)
        else:
            potential_energies = pe_M(walkers)

    
    
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

    
    
        # Gives a boolean array of indices of the walkers that are replicated
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
        walkers = np.append(walkers_after_delete, walkers_after_replication, axis=walker_axis)
        
    # Calculate the time at the end of the simulation
    end_time = time.time()
    
    
    # Calculate the rolling average for rolling_avg time steps
    ref_rolling_avg = np.zeros(sim_length-rolling_avg)
    for i in range(sim_length-rolling_avg):
        # Calculate the rolling average by looping over the past rolling_avg time steps 
        for j in range(rolling_avg):
            ref_rolling_avg[i] = ( ref_rolling_avg[i] - ( ref_rolling_avg[i] / (j+1) ) ) \
                + ( reference_energy[(i+rolling_avg)-j] / (j+1) ) 
            
    # Calculate the distance between one of the OH vectors
    # Used in the histogram and wave function plot  
    OH_positions = np.linalg.norm(walkers[:,0,0]-walkers[:,0,1], axis = molecule_axis)
    
    # Returns the elapsed time, number of walkers, and a bond length
    return (end_time-start_time), num_walkers, ref_rolling_avg, OH_positions

    
#######################################################################################
# Main Testing Loop

    
# Number of simulations ran 
num_sims = 2

# Get an initial position for walkers based on the equilibration phase
init_walkers = equilibrate_walkers()


# Create arrays to record output
# Elapsed time of the simulation loop - used for comparing PE functions
elapsed_time = []

# Number  of walkers in the simulation - used for showing convergence and a valid time step
num_walkers = []

# Rolling average of the reference energy
ref_avg = []

# Length of one OH bond
OH_positions = []

# Run the simulation num_sims times and record all the outputs in an array
for i in range(num_sims):
    time_taken, walkers, ref, OH_p = sim_loop(True, init_walkers)
    
    elapsed_time.append(time_taken)
    num_walkers.append(walkers)
    ref_avg.append(ref)
    OH_positions.append(OH_p)

elapsed_time = np.stack(elapsed_time, axis = -1)
#num_walkers = np.stack(num_walkers, axis = -1)
#ref_avg = np.stack(ref_avg, axis = -1)
#OH_positions = np.stack(OH_positions, axis = -1)

    

# Calculate the average time
avg_time = np.mean(elapsed_time)

# Calculate an array of length sim_length of the average number of walkers at a time step
avg_walkers = np.mean(num_walkers, axis = 1)

# Calculate an array of length sim_length of the average ref energy rolling average 
avg_ref_avg = np.mean(ref_avg, axis = 1)

# Calculate the zero point energy based on the average of the ref rolling energy
ref_converge_num = np.mean(avg_ref_avg)
    

    
#####################################################################################
# Output


# Create an array to graph the Zero point energy
zp_energy = np.ones(sim_length) * ref_converge_num

# Create an array to graph the initial number of walkers
initial_walkers = np.ones(sim_length) * n_walkers


# Range for rolling average graphing
ref_x = np.arange(rolling_avg,sim_length)
    
# Plot the rolling average of the reference energy throughout the simulation
plt.figure(1)
# Plot every reference energy
for i in range(num_sims):
    plt.plot(ref_x, ref_avg[i], label= 'Reference Energy ' + str(i))
# Plot the average reference rolling average
#plt.plot(ref_x, avg_ref_avg, label='Average Reference Energy')
# Plot the Zero-Point energy
plt.plot(zp_energy, label='ZP Energy (' + str.format('{0:.6f}', ref_converge_num) + ')')

plt.axis([0,sim_length,.06,.065])
plt.xlabel('Simulation Iteration')
plt.ylabel('Reference Energy')
plt.title(str(rolling_avg) + ' Step Rolling Average')
plt.legend()
    
# Plot the number of walkers throughout the simulation
plt.figure(2)
# Plot every set of walkers
for i in range(num_sims):
    plt.plot(num_walkers[i], label='Current Walkers ' + str(i))
# Plot the average number of walkers
#plt.plot(avg_walkers, label='Average Current Walkers')
# Plot the initial number of walkers
plt.plot(initial_walkers, label='Initial Walkers')

plt.xlabel('Simulation Iteration')
plt.ylabel('Number of Walkers')
plt.title('Number of Walkers Over Time')
plt.legend()

plt.show()


