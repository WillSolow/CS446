# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 10/30/20

# This file contains a library of potential energy functions to be used in the main simulation
# loop. They are stored here for the purpose of keeping the main body of code clean

# Imports
import numpy as np

# Initial Constants
# Equilibrium length of OH Bond
eq_bond_length = 1.889727

# Equilibrium angle of the HOH bond in radians
eq_bond_angle = HOH_bond_angle * np.pi/180

# Spring constant of the OH Bond
kOH = 6.027540

# Spring constant of the HOH bond angle
kA = 0.120954

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