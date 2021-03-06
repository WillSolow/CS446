# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 11/23/20

# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. This file supports the main simulation 
# loop as a function so that the user can run multiple tests and compute useful data.

# To Run: Navigate to file in terminal directory and type 'python dmc_rs_MultiTest.py'

# Output: Graphs for the reference energy, the n-step rolling average, and the number 
# of walkers at each time step, as well as a density histogram of the walker distance
# from equilibrium and the corresponding wave function

# Imports
import numpy as np
#import matplotlib.pyplot as plt
#import scipy.integrate as integrate
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
# seed = 81716

# Set the seed for the pseudo-random number generator. 
np.random.seed(seed)
print('Seed used: ' + str(seed))



####################################################################################
# Simulation Loop Constants


# Time step 
# Used to calculate the distance an atom moves in a time step
# Smaller time step means less movement in a given time step
dt = .1



# Number of simulations ran 
num_sims = 10



# Length of the equilibration phase in time steps. The below data is for the 
# water molecule
# If dt = 1.0, equilibration phase should be greater than 1500
# If dt = 0.5, equilibration phase should be greater than 2500
# If dt = 0.1, equilibration phase should be greater than 5000
# If dt = 0.01, equilibration phase should be greater than 100000
# For smaller dt, can calculate (1 / sqrt(dt)) * 1500 and round up generously given that
# the time taken to reach equilibrium is dependent on the square root of the time step. 
# In theory this works, but experimentally it has been found that smaller time steps 
# require drastically more time to reach equilibrium
#equilibration_phase = 5000




# Number of time steps in a simulation.
# Simulation length should be at least five times the length of the equilibration phase
sim_length = 5000

# Number of initial walkers
n_walkers = 5000

# Number of time steps for rolling average calculation
rolling_avg = 1000


# Number of bins for histogram. More bins is more precise
n_bins = 50




# Set the dimensions of the 4D array of which the walkers, molecules, atoms, 
# and positions reside. Used for clarity in the simulation loop
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
eq_bond_length = 1.0 / 0.529177

# Equilibrium angle of the HOH bond in radians
eq_bond_angle = HOH_bond_angle * np.pi/180

# Spring constant of the OH Bond
kOH = 1059.162 * (1.0 / 0.529177)**2 * (4.184 / 2625.5)

# Spring constant of the HOH bond angle
kA = 75.90 * (4.184 / 2625.5)



# Returns an array of masses in Atomic Mass Units, Oxygen is first followed by both 
# Hydrogens
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)


# Returns an array of atomic charges based on the position of the atoms in the 
# atomic_masses array. This is used in the potential energy function and is 
# broadcasted to an array of distances to calculate the energy using Coulomb's Law. 
atomic_charges = np.array([q_oxygen, q_hydrogen, q_hydrogen])



# Calculate the reduced mass of the system
# Note that as the wave function is being graphed for an OH vector, we only consider the
# reduced mass of the OH vector system
reduced_mass = (atomic_masses[0]*atomic_masses[1])/(atomic_masses[0]+atomic_masses[1])


#######################################################################################
# Potential Energy Functions


# The lambda function below changes all instances of -inf or inf in a numpy array to 0
# under the assumption that the -inf or inf values result from divisions by 0
inf_to_zero = lambda dist: np.where(np.abs(dist) == np.inf, 0, dist)


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
coulombic_charges = np.matmul(np.transpose(atomic_charges[np.newaxis]), \
                    atomic_charges[np.newaxis])  * coulomb_const


# Input: 4D Array of walkers
# Output: 1D Array of potential energies for each walker
# Calculates the potential energy of a walker based on the distance of bond lengths and 
# bond angles from equilibrium
# Currently assumes that there is no interaction between molecules in a walker
def intra_pe(x):
    # Return the two OH vectors
    # Used to calculate the bond lengths and angle in a molecule
    OH_vectors = x[:,:,np.newaxis,0]-x[:,:,1:]
    
    # Returns the lengths of each OH bond vector for each molecule 
    # in each walker. 
    lengths = np.linalg.norm(OH_vectors, axis=3)
    
    
    # Calculates the bond angle in the HOH bond
    # Computes the arccosine of the dot product between the two vectors, by 
    # normalizing the vectors to magnitude of 1
    angle = np.arccos(np.sum(OH_vectors[:,:,0]*-OH_vectors[:,:,1], axis=2) \
            / np.prod(lengths, axis=2))
            
    # Calculates the potential energies based on the magnitude vector and bond angle
    pe_bond_lengths = .5 * kOH * (lengths - eq_bond_length)**2
    pe_bond_angle = .5 * kA * (angle - eq_bond_angle)**2
    
    # Sums the potential energy of the bond lengths with the bond angle to get potential energy
    # of one molecule, then summing to get potential energy of each walker
    return np.sum(np.sum(pe_bond_lengths, axis = 2)+pe_bond_angle, axis=1)
   
    

# Input: 4D Array of walkers
# Output: Three 1D arrays for Intermolecular Potential Energy, Coulombic energy, and 
#         Leonard Jones energy
# Calculates the intermolecular potential energy of a walker based on the distances of 
# the atoms in each walker from one another
def inter_pe(x):
    
    # Returns the atom positions between two distinct pairs of molecules 
    # in each walker. This broadcasts from a 4D array of walkers with axis dimesions 
    # (num_walkers, num_molecules, num_atoms, coord_const) to two arrays with 
    # dimesions (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const),
    # with the result being the dimensions:
    # (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const).
    # These arrays line up such that the corresponding pairs on the second dimension 
    # are the distinct pairs of molecules
    pairs_a = x[:,molecule_index_a]
    pairs_b = x[:,molecule_index_b]
    
    
    
    # Returns the distances between two atoms in each molecule pair. The distance 
    # array is now of dimension (num_walkers, num_distinct_pairs, num_atoms, num_atoms)
    # as each atom in the molecule has its distance computed with each atom in the 
    # other molecule in the distinct pair.
    # This line works similar to numpy's matrix multiplication by broadcasting the 4D 
    # array to a higher dimesion and then taking the elementwise difference before
    # squarring and then summing along the positions axis to collapse the array into
    # distances.
    distances = np.sqrt( np.sum( (pairs_a[...,None] \
                - pairs_b[:,:,np.newaxis,...].transpose(0,1,2,4,3) )**2, axis=3) )
   
   
   
    # Calculate the Coulombic energy using Coulomb's Law of every walker. 
    # Distances is a 4D array and this division broadcasts to a 4D array of 
    # Coulombic energies where each element is the Coulombic energy of an atom pair 
    # in a distinct pair of water molecules. 
    # Summing along the last three axis gives the Coulombic energy of each walker.
    # Note that we account for any instances of divide by zero by calling inf_to_zero
    # on the result of dividing coulombic charges by distance.
    coulombic_energy = np.sum( inf_to_zero(coulombic_charges / distances), axis=(1,2,3))
    
    
    

    # Calculate the quotient of sigma with the distances between pairs of 
    # oxygen molecules
    # Given that the Lennard Jones energy is only calculated for oxygen oxygen pairs.
    # By the initialization assumption, the Oxygen atom is always in the first index,
    # so the Oxygen pair is in the (0, 0) index in the last two dimensions of the 4D
    # array with dimension
    # (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const).
    sigma_dist = inf_to_zero( sigma / distances[:,:,0,0] )
    
    # Calculate the Lennard Jones energy in accordance with the given equation
    # Sum along the first axis to get the total Lennard Jones energy in one walker.
    lennard_jones_energy = np.sum( 4*epsilon*(sigma_dist**12 - sigma_dist**6), \
                           axis = 1)
    
    
    
    # Gives the intermolecular potential energy for each walker as it is the sum of the 
    # Coulombic Energy and the Leonard Jones Energy.
    intermolecular_potential_energy = coulombic_energy + lennard_jones_energy
    
    
    
    # Return all three calculated energys which are 1D arrays of energy values 
    # for each walker
    return intermolecular_potential_energy, coulombic_energy, lennard_jones_energy

    
    
# Input: 4D array of walkers
# Output: 1D array of the sum of the intermolecular and intramolecular 
#         potential energy of each walker
# Calculates the total potential energy of the molecular system in each walker
def total_pe(x):

    # Calculate the intramolecular potential energy of each walker
    intra = intra_pe(x)
    
    # Calculate the intermolecular potential energy of each walker
    inter, coulombic, lennard_jones = inter_pe(x)
    
    
    # Return the total potential energy of the walker
    return intra + inter

 
    
#######################################################################################
# Simulation Functions


# Output: an equilibrated array of walkers based on random initial positions
# Runs the simulation as fast as possible to equilibrate the walkers based on the length of 
# the equilibration phase
def equilibrate_walkers(init_walkers, equil_phase):

    # Initial 4D walker array
    # Returns a uniform distribution cenetered at the given bond length
    # Array axes are walkers, molecules, coordinates, and atoms
    walkers = init_walkers
    
    
    # Simulation loop
    # Iterates over the walkers array, propogating each walker. Deletes and replicates those 
    # walkers based on their potential energies with respect to the calculated reference energy
    for i in range(equil_phase):
    
        # Calculate the Reference Energy
        # Energy is calculated based on the average of all potential energies of walkers.
        # Is adjusted by a statistical value to account for large or small walker populations.
        reference_energy = np.mean( intra_pe(walkers) ) \
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
        potential_energies = intra_pe(walkers)

    
    
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
def sim_loop(init_walkers):

    # Create array to store the number of walkers at each time step
    num_walkers = np.zeros(sim_length)
    
    # Create array to store the reference energy at each time step
    reference_energy = np.zeros(sim_length)
    
    # Set the equilibrated walkers array to 
    walkers = init_walkers
    
    
    # Simulation loop
    # Iterates over the walkers array, propogating each walker. Deletes and replicates those 
    # walkers based on their potential energies with respect to the calculated reference energy
    for i in range(sim_length):
        
        # Calculate the Reference Energy
        # Energy is calculated based on the average of all potential energies of walkers.
        # Is adjusted by a statistical value to account for large or small walker populations.
        reference_energy[i] = np.mean( total_pe(walkers) ) \
            + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
               
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
        
    
    
    # Calculate the rolling average for rolling_avg time steps
    ref_rolling_avg = np.zeros(sim_length-rolling_avg)
    for i in range(sim_length-rolling_avg):
        # Calculate the rolling average by looping over the past rolling_avg time steps 
        for j in range(rolling_avg):
            ref_rolling_avg[i] = ( ref_rolling_avg[i] - ( ref_rolling_avg[i] / (j+1) ) ) \
                + ( reference_energy[(i+rolling_avg)-j] / (j+1) ) 
         
    
    # Returns the elapsed time, number of walkers, and a bond length
    return num_walkers, ref_rolling_avg

    
#######################################################################################
# Main Testing Loop

# Equilibrated walkers array
num_walkers = ['1000_trimer.npy', '5000_trimer.npy','10000_trimer.npy']

# dt values to test
dt_values = [5,1,.5,.1]

# number of times the average is taken over
sim_times = [10,20]

# equilbration standard
equilibrate = 1500



# For every time step 
for time_step in dt_values:
    dt = time_step

    # create a data file for all sims
    fl = open('convergence_data_'+str(dt)+'.txt', 'a') 
    fl.write('Data for time step: '+str(dt)+'\n\n')
    
    # For every walker value
    for walkers in num_walkers:
    
    # Load the initial walkers array
        init_walkers = np.load(walkers)
        num_walk, ext = walkers.split('_')

        # Set the initial number of walkers
        n_walkers = int(num_walk)

        # For every set of times we run the simulation
        for sims in sim_times:
            print('Walkers: '+str(n_walkers)+'. dt: '+str(dt)+'. Sim times: '+str(sims))
            fl.write('Walkers: '+str(n_walkers)+'. Sim times: '+str(sims)+'\n')
            
            tot_avg_ref = []
            tot_avg_walkers = []
            
            # Find the length of the equilibration phase
            equil_phase = int(np.ceil(1/np.sqrt(dt)*equilibrate))
            sim_length = equil_phase*5
            

            for i in range(num_sims):
                print('Test num: ', i)
                # Number  of walkers in the simulation - used for showing convergence and a valid time step
                num_walkers_arr = []

                # Rolling average of the reference energy
                ref_avg = []

                # Propagate the equilibrated walkers for that length based on 
                # the time step
                equil_walkers = equilibrate_walkers(init_walkers, equil_phase)
            
       
                for j in range(sims):
                    walkers_n, ref = sim_loop(equil_walkers)
                    num_walkers_arr.append(walkers_n)
                    ref_avg.append(ref)
                    
                # Collect all data into a numpy array for graphing
                avg_walkers_arr = np.stack(num_walkers_arr, axis = -1)
                ref_avg_arr = np.stack(ref_avg, axis = -1)    
                
                # Calculate an array of length sim_length of the average number of walkers at a time step
                avg_walkers = np.mean(avg_walkers_arr, axis = 1)

                # Calculate an array of length sim_length of the average ref energy rolling average 
                avg_ref_avg = np.mean(ref_avg_arr, axis = 1)
                # Calculate the zero point energy based on the average of the ref rolling energy
                ref_converge_num = np.mean(avg_ref_avg)
                
                tot_avg_walkers.append(np.mean(avg_walkers))
                tot_avg_ref.append(ref_converge_num)
                print('Ref', ref_converge_num)
                print('Walk', np.mean(avg_walkers))
            
            print('\nReference Converge Nums: \n',tot_avg_ref)
            fl.write('\n\n')
            for k in tot_avg_ref:
                fl.write(str.format('{0:.8f}',k)+ ' ') 
            print('\nAverage Walkers: \n',tot_avg_walkers)
            fl.write('\n\n')
            for k in tot_avg_walkers:
                fl.write(str.format('{0:.2f}',k)+' ') 
        pop_std = np.std(tot_avg_walkers)
        ref_std = np.std(tot_avg_ref)
        fl.write('\n\nRef Energy Standard Deviation '+str.format('{0:.8f}',ref_std)+'\n')
        fl.write('Walker Pop Standard Deviation '+str.format('{0:.6f}',pop_std))
        print('\n\n###########################################\n\n')
        fl.write('\n\n########################################\n\n')
    fl.close()
        

