# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 11/04/20

# This code demonstrates the correctness of our potential energy functions


import numpy as np

# Initial constants

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


num_molecules = 3

HOH_bond_angle = 112.0


# Equilibrium length of OH Bond
# Input as equation to avoid rounding errors
eq_bond_length = 1.0 / 0.529177

# Equilibrium angle of the HOH bond in radians
eq_bond_angle = HOH_bond_angle * np.pi/180

# Spring constant of the OH Bond
# Input as equation to avoid rounding errors 
kOH = 1059.162 * (1.0 / 0.529177)**2 * (4.184 / 2625.5)

# Spring constant of the HOH bond angle
kA = 75.90 * (4.184 / 2625.5)

# Returns an array of atomic charges based on the position of the atoms in the atomic_masses array
# This is used in the potential energy function and is broadcasted to an array of distances to
# calculate the energy using Coulomb's Law. 
atomic_charges = np.array([q_oxygen, q_hydrogen, q_hydrogen])


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
coulombic_charges = (np.transpose(atomic_charges[np.newaxis]) @ atomic_charges[np.newaxis])  * coulomb_const


# Input: 4D Array of walkers
# Output: 1D Array of potential energies for each walker
# Calculates the potential energy of a walker based on the distance of bond lengths and 
# bond angles from equilibrium
# Currently assumes that there is no interaction between molecules in a walker
def intra_pe_SR(x):
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
def inter_pe_SR(x):
    
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
def total_pe_SR(x):

    # Calculate the intramolecular potential energy of each walker
    intra_pe = intra_pe_SR(x)
    
    # Calculate the intermolecular potential energy of each walker
    inter_pe, coulombic, lennard_jones = inter_pe_SR(x)
    
    
    # Return the total potential energy of the walker
    return intra_pe + inter_pe


    
# Functions provided by Prof Madison
# Extra comments for clarity added by Will Solow

# Input: 4D array of walkers
# Output: 1D array of total potential energies
# Calculates the intermolecular potential energy and intramolecular potential 
# energy of each walker
def PotentialEnergyManyWaters(positions):

    # Gets the shape of the walker array 
    (nWalkers,nWaters,nAtoms,nCartesian)=positions.shape
    
    # Intialize intramolecular energy array
    intRAmolecularEnergy=np.zeros(nWalkers)
    
    # For each water, calculate the intramolecular potential energy
    # This passes a 1D array of shape (walkers,) to PotentialEnergySingleWater
    # and calculates them in a vectorized manner
    for iWat in range(nWaters):
        intRAmolecularEnergy=intRAmolecularEnergy+PotentialEnergySingleWater(positions[:,iWat])

    # Initialize the intermolecular energy array
    intERmolecularEnergy=np.zeros(nWalkers)
    
    # For every distinct pair of water molecules in each walker
    for iWat in range(nWaters):
        for jWat in range(iWat,nWaters):
            # Calculate the intermolecular potential energy by passing each pair 
            # to PotentialEnergyTwoWaters
            intERmolecularEnergy=intERmolecularEnergy+PotentialEnergyTwoWaters(positions[:,iWat],positions[:,jWat])
    
    # Calculate the sum of the potential energys
    potentialEnergy=intRAmolecularEnergy+intERmolecularEnergy
    
    return potentialEnergy

    
    
# Input: Two ingters in range [0, 1, 2] to represent an atom, as well as a distance
#        between the two atoms
# Output: The Coulombic energy according to Coulomb's Law
def coloumbic(atom1, atom2, distance):
    # Initialize Charges to 0
    q1 = 0
    q2 = 0
    
    # If atom 1 is oxygen (index 0), give it charge -.84
    # Otherwise (hydrogen), give it charge .42
    if atom1 == 0:
        q1 = -0.84
    else:
        q1 = 0.42

    # If atom 2 is oxygen (index 0), give it charge -.84
    # Otherwise (hydrogen), give it charge .42
    if atom2 == 0:
        q2 = -0.84
    else:
        q2 = 0.42
        
    # Calculate the Coulombic energy
    # Distance is already assumed to be non-zero
    coloumbic1 = q1*q2/distance*(1.0/(4.0*np.pi))
    
    return coloumbic1

    
    
# Input: Two arrays of size (3,) representing the xyz coordinates of each atom
# Output: The distance between the two atoms
def atomdistance(atom1, atom2):
    
    # Create a list to store each x, y and z distance
    distancelist = np.zeros(atom1.size)

    # Go through every (x, y, z) coord in atom and calculate the difference
    for i in range(atom1.size):
        axesdistance = atom1[i]-atom2[i]
        distanceSquared = axesdistance ** 2
        distancelist[i] = distanceSquared

    # Calculate the distance by taking the square root of the sum of the 
    # xyz differences
    distance = np.sum(distancelist)
    distance = np.sqrt(distance)

    return distance
  

  
# Input: Two arrays of size (3,3) representing the xyz coordinates of each atom
#        in the water molecule
# Output: The intermolecular potential energy between the two water moolecules
def PotentialEnergyTwoWaters(water1pos, water2pos):

    # Get the shape of both water molecules. Note that they will always be 
    # 3 by 3 in the case of the water molecule.
    (walkers, nAtoms1,nCartesian1)=water1pos.shape
    (walkers, nAtoms2,nCartesian2)=water2pos.shape

    # Initial Epsilon and Sigma constant
    epsilon = 0.1554252
    sigma = 3.165492

    # Converted Epsilon and Sigma constant
    epsilon = epsilon*(4.184/2625.5)
    sigma = sigma/0.529177

    # Initialize energy lists
    potentialEnergyList = []
    coloumbicEnergyList = []
    lennardJonesList = []

    # For every atom in water 1
    for atomNum1 in range(nAtoms1):
        # For every atom in water 2
        for atomNum2 in range(nAtoms2):
        
            # Get the position index [0, 1, 2] of atom 1 and 2
            atom1 = water1pos[atomNum1]
            atom2 = water2pos[atomNum2]
            
            # Calculate the distance between atoms
            distance = atomdistance(atom1, atom2)
            
            # If the distance is not 0, calculate QiQj
            if distance != 0.0:
                # Calculate Coulombic energy
                coloumbicV = coloumbic(atomNum1, atomNum2, distance)
                
                # If both atoms are oxygen, calculate the lennard jones energy
                if atomNum1 == 0 and atomNum2 == 0:
                    lennardJones = 4*epsilon*((sigma/distance)**12 \
                                   - (sigma/distance)**6)
                    
                    # Edit made by Will Solow (11/4/20) 
                    # Moved this line of code
                    lennardJonesList.append(lennardJones)
                                   
                    # Calculate the intermolecular potential energy              
                    potential = lennardJones + coloumbicV
                    
                # If not both oxygen, then potential energy is just Coulombic energy
                else:
                    potential = coloumbicV
                
                potentialEnergyList.append(potential)
                coloumbicEnergyList.append(coloumbicV)
                
                # This is commented out and moved into the if statement as it is 
                # believed to be a bug in the code (11/4/2)
                # lennardJonesList.append(lennardJones)
    
    # Cast Python lists to Numpy arrays for quick summing
    potentialEnergyList = np.array(potentialEnergyList)
    coloumbicEnergyList = np.array(coloumbicEnergyList)
    lennardJonesList = np.array(lennardJonesList)
    
    # Sum up all energies
    VinterSum = np.sum(potentialEnergyList)
    coloumbicEnergySum = np.sum(coloumbicEnergyList)
    lennardJonesSum = np.sum(lennardJonesList)


    return VinterSum, coloumbicEnergySum, lennardJonesSum


# Input: a walkersx3x3 array of xyz coordinates representing the position of a water
# Output: The intramolecular potential energy  
def PotentialEnergySingleWater(OHHpositions):

    #The first atom is assumed to be Oxygen
    #The second and third atoms are assumed to be the two Hydrogens

    #The potential energy of water is the sum of the PE from the two OH 
    #bond lengths and the H-O-H bond angle

    # Calculate the length of the first OH bond
    rOH1=np.linalg.norm(OHHpositions[:,0,:]-OHHpositions[:,1,:],axis=1) 
    
    
    # equilibrium bond length in atomic units of distance
    rOHeq=1.0 /0.529177 
    # spring constant in atomic units of energy per (atomic units of distance)^2
    kb= 1059.162 *(1.0/0.529177)**2 * (4.184/2625.5)

    # Calculate the potential energy of OH bond 1
    potROH1=kb/2.0 *(rOH1-rOHeq)**2
    

    # Calculate the length of the second OH bond 
    rOH2=np.linalg.norm(OHHpositions[:,0]-OHHpositions[:,2],axis=1)

    # Calculate the potential energy of OH bond 2 
    potROH2=kb/2.0 *(rOH2-rOHeq)**2

    aHOH=[]
    # For each walker in the inputted list, calculate the angle of the 
    # H - O - H atom
    for walkerPos in OHHpositions:
        vecOH_1=walkerPos[0]-walkerPos[1]
        vecOH_2=walkerPos[2]-walkerPos[0]
        cosAngle=np.dot(vecOH_1,vecOH_2)/(np.linalg.norm(vecOH_1)*np.linalg.norm(vecOH_2))
        aHOH.append(np.arccos(cosAngle))

    # Convert the Python list to a Numpy arra
    aHOH=np.array(aHOH)
    
    #spring constant in atomic units of energy per (rad)^2
    ka=75.90*(4.184/2625.5) 
    #equilibrium HOH bond angle in radians
    aHOHeq= 112.0 * np.pi/180.0 
    
    # Calculate the potential energy of the HOH angle
    potAHOH=ka/2.0*(aHOH-aHOHeq)**2

    # Sum all potential energies
    potentialEnergy=potROH1+potROH2+potAHOH

    return potentialEnergy



test_walkers = np.random.rand(10, num_molecules, 3, 3)  

print('SR total PE:      ', total_pe_SR(test_walkers))
print('Madison total PE: ', PotentialEnergyManyWaters(test_walkers))  
    
