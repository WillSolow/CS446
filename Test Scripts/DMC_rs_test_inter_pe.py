# Will Solow, Skye Rhomberg (With code from Professor Lindsey Madison, Colby College)
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 11/2/20

# The purpose of this file is to demonstrate that the intermolecular potential energy created
# by Will and Skye produces the correct output


# Imports
import numpy as np 
import itertools as it
import matplotlib.pyplot as plt



# Code from Professor Lindsey Madison
def PotentialEnergyTwoWaters(water1pos, water2pos):
    #not yet implemented!! Until implemented this will return zeros which correspond to the waters not interacting 
    #with each other

    # (nWalkers1,nAtoms1,nCartesian1)=water1pos.shape
    # (nWalkers2,nAtoms2,nCartesian2)=water2pos.shape

    (nAtoms1,nCartesian1)=water1pos.shape
    (nAtoms2,nCartesian2)=water2pos.shape

    # intTERmolecularEnergy=np.zeros(nWalkers1)

    # for iWat in range(nWaters1):
    #   for jWat in range(nAtoms2)
    epsilon = 0.1554252
    sigma = 3.165492

    # # conversion factors
    # rOHeq=1.0 /0.529177 #equilibrium bond length in atomic units of distance
    # kb= 1059.162 *(1.0/0.529177)**2 * (4.184/2625.5)# spring constant in atomic units of energy per (atomic units of distance)^2

    # converted epsilon and sigma
    epsilon = epsilon*(4.184/2625.5)
    sigma = sigma/0.529177

    # initialize potential energy list
    potentialEnergyList = []
    coloumbicEnergyList = []
    lennardJonesList = []

    # sort through atoms in water 1
    for atomNum1 in range(nAtoms1):
        # sort through atoms in water 2
        for atomNum2 in range(nAtoms2):
            # obtain atom in water1
            atom1 = water1pos[atomNum1]
            #print('ind atom1: ', atom1, "atom: ", atomNum1)
            # obtain atom in water2
            atom2 = water2pos[atomNum2]
            #print('ind atom2: ', atom2, "atom: ", atomNum2)
            # calculate atom atom distance
            distance = atomdistance(atom1, atom2)
            # print('Distance ', distance)
            # if distance is not 0, calculate qiqj
            if distance != 0.0:
                coloumbicV = coloumbic(atomNum1, atomNum2, distance)
                #print('Colombic energy between two atoms: ', coloumbicV)
                # if oxygen and oxygen
                if atomNum1 == 0 and atomNum2 == 0:
                    #print('Madison Oxygen distance: ', distance)
                    lennardJones = 4*epsilon*((sigma/distance)**12 - (sigma/distance)**6)
                    #print('Madison lennard Jones', lennardJones)
                    potential = lennardJones + coloumbicV
                # for every other combination
                else:
                    potential = coloumbicV
                
                #print('Potential energy between two atoms: ', potential)
                potentialEnergyList.append(potential)
                coloumbicEnergyList.append(coloumbicV)
                lennardJonesList.append(lennardJones)
          
    #print('Lennard Jones list: ', lennardJonesList)
    potentialEnergyList = np.array(potentialEnergyList)
    coloumbicEnergyList = np.array(coloumbicEnergyList)
    lennardJonesList = np.array(lennardJonesList)
    # sum up all potential energies
    VinterSum = np.sum(potentialEnergyList)
    coloumbicEnergySum = np.sum(coloumbicEnergyList)
    lennardJonesSum = np.sum(lennardJonesList)
    #print('Madison lennardJonesSum 2: ', lennardJonesSum)


    return VinterSum, coloumbicEnergySum, lennardJonesSum
    
def coloumbic(atom1, atom2, distance):
    """
    Return q1q2/R

    Parameters
    ------------
    atom1: int.
        index number of atom 1 (0: oxygen, 1: hydrogen, 1: hydrogen)
    atom2: int.
        index number of atom 2 (0: oxygen, 1: hydrogen, 1: hydrogen)
    """
    q1 = 0
    q2 = 0

    if atom1 == 0:
        q1 = -0.84
    else:
        q1 = 0.42


    if atom2 == 0:
        q2 = -0.84
    else:
        q2 = 0.42
    coloumbic1 = q1*q2/distance*(1.0/(4.0*np.pi))
    
    return coloumbic1


def atomdistance(atom1, atom2):
    """
    Return the atom-atom distance 

    Parameters
    ------------
    atom1: numpy 1D array 
    atom2: numpy 1D array
    """
    distancelist = np.zeros(atom1.size)

    # go through every (x, y, z) coord in atom and calculate distance
    for i in range(atom1.size):
        axesdistance = atom1[i]-atom2[i]
        distanceSquared = axesdistance ** 2
        distancelist[i] = distanceSquared

    distance = np.sum(distancelist)
    distance = np.sqrt(distance)

    return distance
  
# SR constants

# number of molecules 
num_molecules = 2


# Chemsitry constants for intermolecular energy
SRsigma = 3.165492 / 0.529177
SRepsilon = 0.1554252 * (4.184/2625.5)

# Coulombic charges
q_oxygen = -0.84
q_hydrogen = 0.42

# Coulomb's Constant
coulomb_const = 1.0 / (4.0*np.pi)    
 
# Returns an array of atomic charges based on the position of the atoms in the atomic_masses array
# This is used in the potential energy function and is broadcasted to an array of distances to
# calculate the energy using Coulomb's Law. 
atomic_charges = np.array([q_oxygen, q_hydrogen, q_hydrogen])

# Computes the product of the charges as the atom charges are multiplied together in accordance
# with Coulomb's Law
coulombic_charges = (np.transpose(atomic_charges[np.newaxis]) @ atomic_charges[np.newaxis])  * coulomb_const

# Create indexing arrays for the distinct pairs of water molecules in the potential 
# energy calculation. Based on the idea that there are num_molecules choose 2 distinct
# molecular pairs
molecule_index_1 = np.array(sum([[i]*(num_molecules-(i+1)) for i in range(num_molecules-1)],[]))
molecule_index_2 = np.array(sum([list(range(i,num_molecules)) for i in range(1,num_molecules)],[]))
#molecule_idx = lambda n: list(zip(*it.combinations(n, 2))) 
#molecule_index_1, molecule_index_2 = [np.array(m) for m in molecule_idx(num_molecules)]
 
# Input: 4D Array of walkers
# Output: Three 1D arrays for Intermolecular Potential Energy, Coulombic energy, and 
#         Leonard Jones energy
# Calculates the intermolecular potential energy of a walker based on the distances of the
# atoms in each walker from one another
def inter_potential_energy(x):

    #print('walker data type: ', x.dtype)
    
    # Returns the difference of the atom positions between two distinct pairs of molecules 
    # in each walker. This broadcasts from a 4D array of walkers with axis dimesions 
    # (num_walkers, num_molecules, num_atoms, coord_const) to two arrays with 
    # dimesions (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const),
    # with the result being the dimensions:
    # (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const)
    #molecule_difference = x[:,molecule_index_1] - x[:,molecule_index_2]
    #print('Molecule difference: \n', molecule_difference)
    #print('Molecule diff transpose: \n', np.transpose(molecule_difference, (0, 1, 3, 2)))
    #print('Molecule matrix mult: \n', molecule_difference @ np.transpose(molecule_difference, (0, 1, 3, 2)))
    
    # Returns the distances between two atoms in each molecule pair. The distance array is 
    # now of dimension (num_walkers, num_distinct_pairs, num_atoms, num_atoms) as each
    # atom in the molecule has its distance computed with each atom in the other molecule in
    # the distinct pair.
    # distances = np.sqrt(molecule_difference @ np.transpose(molecule_difference, (0, 1, 3, 2)))
    mol_a, mol_b = x[:,molecule_index_1,...], x[:,molecule_index_2,...]
    
    distances = np.sqrt( np.sum( (mol_a[...,None] \
            - mol_b[:,:,np.newaxis,...].transpose(0,1,2,4,3) )**2, axis=3, dtype='float64') , dtype='float64' )
    #print('distances \n', distances)
   
   
    # Calculate the Coulombic energy using Coulomb's Law of every walker. 
    # Distances is a 4D array and this division broadcasts to a 4D array of Coulombic energies
    # where each element is the Coulombic energy of an atom pair in a distinct pair of water 
    # molecules. 
    # Summing along the last three axis gives the Coulombic energy of each walker
    coulombic_energy = np.sum(coulombic_charges / distances, axis=(1,2,3), dtype='float64')
    
    
    
    oxygen = distances[0,0,0,0]
    #print('SR oxygen distance ', oxygen)
    o_LJ = 4*SRepsilon*((SRsigma/oxygen)**12 - (SRsigma/oxygen)**6)
    #print('SR Lennard jones test', o_LJ)
    # Calculate the Leonard Jones Energy given that it is only calculated when both atoms
    # are Oxygen. By the initialization assumption, the Oxygen atom is always in the first index,
    # so the Oxygen pair is in the (0, 0) index in the last two dimensions of the 4D array with
    # dimension (num_walkers, num_distinct_molecule_pairs, num_atoms, coord_const)
    lennard_jones_energy = np.sum( 4*SRepsilon*((SRsigma/distances[:,:,0,0])**12 - (SRsigma/distances[:,:,0,0])**6), axis = 1, dtype='float64')
    #print('SR oxygen distance', float(distances[:,:,0,0]))
    #print('SR lennard jones: ', float(lennard_jones_energy))
    
    
    # Returns the intermolecular potential energy for each walker as it is the sum of the 
    # Coulombic Energy and the Leonard Jones Energy
    intermolecular_potential_energy = coulombic_energy + lennard_jones_energy
    
    return intermolecular_potential_energy, coulombic_energy, lennard_jones_energy

    
    
# Test code provided by Prof Madison
for i in range(100):
    atom1 = [[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,3.0]]
    atom1 = np.array(atom1)
    atom2 = [[1.0*(i+2),0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,1.0]]
    atom2 = np.array(atom2)
    VinterSum, coloumbicEnergySum, lennardJonesSum  = PotentialEnergyTwoWaters(atom1, atom2)
    
    
    water_walker = np.stack((atom1[np.newaxis,:,:], atom2[np.newaxis,:,:]), axis=1)
   
    
    SR_inter_sum, SR_coulombic_energy, SR_lennardJones = inter_potential_energy(water_walker)
    
    print('\n\nMadison Coulombic energy sum: ', coloumbicEnergySum)
    print('SR Coulombic energy sum: ', float(SR_coulombic_energy))
    print('\nMadison PE: ', VinterSum)
    print('SR PE: ', float(SR_inter_sum))
    #plt.scatter(i,  coloumbicEnergySum)
    #plt.xlim(4, 100)
    # plt.ylim(-0.002, 0.0)

# plt.show()
    
# print("Result: ",PotentialEnergyManyWaters(sample2WaterWalkers))
print("\n\nEnd test 1. \n \n")

# Create 2 water molecules randomly for testing purposes with non zero values
atom1 = np.random.rand(3, 3)
#print(atom1)
print('\n\n')
atom2 = np.random.rand(3, 3)
#print(atom2)

water_walker = np.stack((atom1[np.newaxis,:,:], atom2[np.newaxis,:,:]), axis=1)

VinterSum, coloumbicEnergySum, lennardJonesSum  = PotentialEnergyTwoWaters(atom1, atom2)

SR_inter_sum, SR_coulombic_energy, SR_lennardJones = inter_potential_energy(water_walker)

print('\n\nMadison Coulombic energy sum: ', coloumbicEnergySum)
print('SR Coulombic energy sum: ', float(SR_coulombic_energy))
print('\nMadison PE: ', VinterSum)
print('SR PE: ', float(SR_inter_sum))
print('\n\n Madison LJ: ', lennardJonesSum)
print('SR LJ: ', float(SR_lennardJones))

#print('Atomic charges: \n', atomic_charges)
#print('Atoimc charges shape:', atomic_charges[np.newaxis].shape)
#print('Columibc charges arr\n', coulombic_charges)
#print('Atomic charges tranpose: \n', np.transpose(atomic_charges[np.newaxis]))
#print('Atoimc charges tranpose shape: ', np.transpose(atomic_charges[np.newaxis]).shape)
#print('Coul Charges: \n', (np.transpose(atomic_charges[np.newaxis]) @ atomic_charges[np.newaxis]) )
