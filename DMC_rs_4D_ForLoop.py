# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 10/27/20

# This program runs a Diffusion Monte Carlo simulation to find an approximation for the
# ground state energy of a system of molecules. We provide a for-loop implementation to
# demonstrate that the vectorized approach gives the same results as an easier to validate
# for loop

# To Run: Navigate to file in terminal directory and type 'python dmc_rs_4D_ForLoop.py'

# Output: Graphs for the reference energy, the n-step rolling average, and the number 
# of walkers at each time step, as well as a density histogram of the walker distance
# from equilibrium and the corresponding wave function

# Imports
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.integrate as integrate


###################################################################################
# Scientific Constants


# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e23

# Number of coordinates
# Always 3, used for clarity
coord_const = 3

# create a random seed for the number generator, can be changed to a constant value
# for the purpose of replicability
seed = np.random.randint(100000)
# Set the seed manually for replicability purposes over multiple simulations
seed = 81716

# Set the seed for the pseudo-random number generator. 
np.random.seed(seed)
print('Seed used: ' + str(seed))

####################################################################################
# Simulation Loop Constants


# Time step 
# Used to calculate the distance an atom moves in a time step
# Smaller time step means less movement in a given time step
dt = 10

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


# Number of molecules in each walker
# Used to initialize the walker array
num_molecules = 1



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

# Calculate the convergence reference energy based on the given equation.
ref_converge_num = .00494317



# Returns an array of masses in Atomic Mass Units, Oxygen is first followed by both 
# Hydrogens
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)


# Calculate the reduced mass of the system
# Note that as the wave function is being graphed for an OH vector, we only consider the
# reduced mass of the OH vector system
reduced_mass = (atomic_masses[0]*atomic_masses[1])/(atomic_masses[0]+atomic_masses[1])
# Legacy code
# reduced_mass = np.prod(atomic_masses) / np.sum(atomic_masses)



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
def potential_energy_SR(x):
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
def potential_energy_M(positions):
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
	

# Input: Boolean - if true used SR vectorized PE, if false use Madison PE function
# Output: Time taken to complete computation
# Runs a simulation with vectorized techniques
def vectorized_loop(vec_PE):
    # Reset the seed each loop to ensure that simulations are the same
    np.random.seed(seed)
	
	
    # Initial 4D walker array
    # Returns a uniform distribution cenetered at the given bond length
    # Array axes are walkers, molecules, coordinates, and atoms
    walkers = (np.random.rand(n_walkers, num_molecules, atomic_masses.shape[0], \
        coord_const) - .5) 
	
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
            reference_energy[i] = np.mean( potential_energy_SR(walkers) ) \
                + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
        else:
            reference_energy[i] = np.mean( potential_energy_M(walkers) ) \
                + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
		
        # Current number of walkers
        num_walkers[i] = walkers.shape[0]

	
	    # Propagates each coordinate of each atom in each molecule of each walker within a normal
	    # distribution given by the atomic mass of each atom.
        # Returns a 4D array in the shape of walkers with the standard deviation depending on the
        # atomic mass of each atom	
        propagations = np.random.normal(0, np.sqrt(dt/np.transpose(np.tile(atomic_masses, \
	       (walkers.shape[0], num_molecules, coord_const, 1)), (0, 1, 3, 2))))
		
	    # Propagates each coordinate of each atom in each molecule of each walker within a normal
	     #distribution given by the atomic mass of each atom.
	    # This method  does the same as above but it is a little more straightforward to see the 
	    # correctness. Both return the same results, and the above is faster
        #propagate_atoms = [np.random.normal(0, np.sqrt(dt/atomic_masses[i]), (walkers.shape[0],\
	    #    num_molecules, coord_const)) for i in range(atomic_masses.shape[0])]
	    #propagations = np.stack(propagate_atoms, axis = 2)
		
	    # Adds the propagation lengths to the 4D walker array
        walkers = walkers + propagations
	
    
	
	    # Calculates the potential energy of each walker in the system
        if vec_PE:
            potential_energies = potential_energy_SR(walkers)
        else:
            potential_energies = potential_energy_M(walkers)

    
	
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
        walkers = np.append(walkers_after_delete, walkers_after_replication, axis=0)
	
	# return the difference between the start time and current time to get time elapsed
	# in seconds
    return (time.time()-start_time)
	
	
	
# Input: Boolean - if true used SR vectorized PE, if false use Madison PE function
# Output: Time taken to complete computation
# Runs a DMC simulation without vectorized techniques
def unvectorized_loop(vec_PE):
    # Reset the seed each loop to ensure that simulations are the same
    np.random.seed(seed)
	
	
    # Initial 4D walker array
    # Returns a uniform distribution cenetered at the given bond length
    # Array axes are walkers, molecules, coordinates, and atoms
    walkers = (np.random.rand(n_walkers, num_molecules, atomic_masses.shape[0], \
        coord_const) - .5) 
	
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
            reference_energy[i] = np.mean( potential_energy_SR(walkers) ) \
                + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
        else:
            reference_energy[i] = np.mean( potential_energy_M(walkers) ) \
                + (1.0 - (walkers.shape[0] / n_walkers) ) / ( 2.0*dt )
		
        # Current number of walkers
        num_walkers[i] = walkers.shape[0]

	
	    # Propagates each coordinate of each atom in each molecule of each walker within a normal
	    # distribution given by the atomic mass of each atom.
        # Returns a 4D array in the shape of walkers with the standard deviation depending on the
        # atomic mass of each atom	
        propagations = np.random.normal(0, np.sqrt(dt/np.transpose(np.tile(atomic_masses, \
	       (walkers.shape[0], num_molecules, coord_const, 1)), (0, 1, 3, 2))))
		
	    # Propagates each coordinate of each atom in each molecule of each walker within a normal
	     #distribution given by the atomic mass of each atom.
	    # This method  does the same as above but it is a little more straightforward to see the 
	    # correctness. Both return the same results, and the above is faster
        #propagate_atoms = [np.random.normal(0, np.sqrt(dt/atomic_masses[i]), (walkers.shape[0],\
	    #    num_molecules, coord_const)) for i in range(atomic_masses.shape[0])]
	    #propagations = np.stack(propagate_atoms, axis = 2)
		
	    # Adds the propagation lengths to the 4D walker array
        walkers = walkers + propagations
	
    
	
	    # Calculates the potential energy of each walker in the system
        if vec_PE:
            potential_energies = potential_energy_SR(walkers)
        else:
            potential_energies = potential_energy_M(walkers)

    
	
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
        walkers = np.append(walkers_after_delete, walkers_after_replication, axis=0)
	
	# return the difference between the start time and current time to get time elapsed
	# in seconds
    return (time.time()-start_time)
	
# Number of simulations ran 
num_sims = 10
# Get output:
vec_time = 0
unvec_time = 0
for i in range(num_sims):
    print('\nStarting loop: ', i)
    vec_time = vec_time + vectorized_loop(True)
    unvec_time = unvec_time + vectorized_loop(False)


print('Vectorized PE function average time: ', vec_time/num_sims)
print('Unvectorized PE function average time: ', unvec_time/num_sims)



