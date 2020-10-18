# Author: Lindsey Madison, Colby College (Chemistry)
# (very lightly edited / a few comments added by Eric Aaron, Colby College (CS))
# Oct. 6, 2020

#potential energy function for 1 flexible water. The bond lenghts and bond angle of water is not fixed
#Based on Wu, Tepper, and Voth's paper: https://aip.scitation.org/doi/10.1063/1.2136877
#and then re-parameterized by Paesani, et. al: https://aip.scitation.org/doi/10.1063/1.2386157
#nicely summarized here (q-SPC/Fw):  http://www.sklogwiki.org/SklogWiki/index.php/SPC/Fw_model_of_water

#A note about mass:
#In your diffusion function the diffusion depends on the mass of the atom.
#A mass array of the atoms in a water molecule might look like:
#[massO, massH, massH]=np.array([15.99491461957,1.007825, 1.007825])/(6.02213670000e23*9.10938970000e-28)

import numpy as np

# NOTE: not fully implemented for many waters interacting, 
# as PotentialEnergyTwoWaters is not fully implemented (see below)

def PotentialEnergyManyWaters(positions):
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

	print("Sum IntRAmolecular Energy: ",intRAmolecularEnergy)
	potentialEnergy=intRAmolecularEnergy+intERmolecularEnergy
	return potentialEnergy


# SEE NOTE BELOW
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
	#This could definitely be streamlined/sped up as some of the functions are being computed repeatedly 
	#(the norm of the OH vectors, for instance)
# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e+23

	
# Atomic masses of atoms in system
# Used to calculate the atomic masses in Atomic Mass Units
oxygen_mass = 15.995
hydrogen_mass = 1.008
HOH_bond_angle = 112.0



# Equilibrium length of OH Bond
eq_bond_length = 1.0 /0.529177

# Equilibrium angle of the HOH bond in radians
eq_bond_angle = HOH_bond_angle * np.pi/180

# Spring constant of the OH Bond
kOH = 1059.162 *(1.0/0.529177)**2 * (4.184/2625.5)

# Spring constant of the HOH bond angle
kA = 75.90*(4.184/2625.5)

# Returns an array of masses in Atomic Mass Units, Oxygen is first followed by both 
# Hydrogens
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)


def potential_energy(x):
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
	
	
#This is 3 walkers with three different configurations of the atoms
print("Testing Potential for walkers with single water")
sample1WaterWalkers=[[[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,3.0]],
			[[0.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,1.0]],
			[[0.0,0.0,0.0],[1.8897,0.0,0.0],[0.0,1.8897,0.0]]]
sample1WaterWalkers=np.array(sample1WaterWalkers)
print("Result: ", PotentialEnergySingleWater(sample1WaterWalkers))
print("End test. \n \n")



print("Testing Potential for walkers with two waters")
sample2WaterWalkers=[ 
					[ [[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,3.0]], #Walker 1, water 1's positions
					 [[0.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,1.0]] ], #Walker 1, water 2's positions
					
					[ [[0.0,0.0,0.0],[1.8897,0.0,0.0],[0.0,1.8897,0.0]], #Walker 2, water 1's positions
					  [[0.0,0.0,1.0],[1.8897,0.0,1.0],[0.0,1.8897,1.0]] ], #Walker 2, water 2's positions

					[ [[1.0,0.0,0.0],[1.8897,0.0,0.0],[1.0,1.8897,0.0]], #Walker 3, water 1's positions
					  [[0.0,0.0,1.0],[-1.8897,0.0,1.0],[-1.0,1.8897,1.0]] ], #Walker 3, water 2's positions
					]
sample2WaterWalkers=np.array(sample2WaterWalkers)

print("Testing Potential for walkers with two waters")
print("Result: ",PotentialEnergyManyWaters(sample2WaterWalkers))
print("End test. \n \n")
print("SR Result: ", potential_energy(sample2WaterWalkers))
