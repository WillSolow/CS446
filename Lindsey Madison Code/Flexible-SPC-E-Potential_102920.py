# Author: Lindsey Madison, Colby College (Chemistry)
# (very lightly edited / a few comments added by Eric Aaron, Colby College (CS))
# Oct. 29, 2020

#potential energy function for 1 flexible water. The bond lenghts and bond angle of water is not fixed
#Based on Wu, Tepper, and Voth's paper: https://aip.scitation.org/doi/10.1063/1.2136877
#and then re-parameterized by Paesani, et. al: https://aip.scitation.org/doi/10.1063/1.2386157
#nicely summarized here (q-SPC/Fw):  http://www.sklogwiki.org/SklogWiki/index.php/SPC/Fw_model_of_water

#A note about mass:
#in your diffusion function the diffusion depends on the mass of the atom.
#that mass array might look like:
#[massO, massH, massH]=np.array([15.99491461957,1.007825, 1.007825])/(6.02213670000e23*9.10938970000e-28)


import numpy as np
import matplotlib.pyplot as plt


# NOTE: NEWLY ADDED FOR 102920 VERSION
# feel free to direct comments / question to Eric Aaron (not the author,
# but still the designated recipient for questions as of now)
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
  


def PotentialEnergyTwoWaters(water1pos, water2pos):
	#not yet implemented!! Until implemented this will return zeros which correspond to the waters not interacting 
	#with each other

	# (nWalkers1,nAtoms1,nCartesian1)=water1pos.shape
	# (nWalkers2,nAtoms2,nCartesian2)=water2pos.shape

	(nAtoms1,nCartesian1)=water1pos.shape
	(nAtoms2,nCartesian2)=water2pos.shape

	# intTERmolecularEnergy=np.zeros(nWalkers1)

	# for iWat in range(nWaters1):
	# 	for jWat in range(nAtoms2)
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
			print('ind atom1: ', atom1, "atom: ", atomNum1)
			# obtain atom in water2
			atom2 = water2pos[atomNum2]
			print('ind atom2: ', atom2, "atom: ", atomNum2)
			# calculate atom atom distance
			distance = atomdistance(atom1, atom2)
			print(distance)
			# if distance is not 0, calculate qiqj
			if distance != 0.0:
				coloumbicV = coloumbic(atomNum1, atomNum2, distance)
				print(coloumbicV)
				# if oxygen and oxygen
				if atomNum1 == 0 and atomNum2 == 0:
					lennardJones = 4*epsilon*((sigma/distance)**12 - (sigma/distance)**6)
					potential = lennardJones + coloumbicV
				# for every other combination
				else:
					potential = coloumbicV
				
				print(potential)
				potentialEnergyList.append(potential)
				coloumbicEnergyList.append(coloumbicV)
				lennardJonesList.append(lennardJones)

	potentialEnergyList = np.array(potentialEnergyList)
	coloumbicEnergyList = np.array(coloumbicEnergyList)
	lennardJonesList = np.array(lennardJonesList)
	# sum up all potential energies
	VinterSum = np.sum(potentialEnergyList)
	coloumbicEnergySum = np.sum(coloumbicEnergyList)
	lennardJonesSum = np.sum(lennardJonesList)


	return VinterSum, coloumbicEnergySum, lennardJonesSum


	# return np.zeros(nWalkers1)
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
	# calculate bond length between O and H for [all walkers, oxygen (0), 3 cartesian coordinates] [all walkers, H (0), :]
	# find the length of all the vectors along the cartesian coordinate dimension of the array
	
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

#This is 3 walkers with three different configurations of the atoms
# print("Testing Potential for walkers with single water")
# sample1WaterWalkers=[[[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,3.0]],
# 			[[0.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,1.0]],
# 			[[0.0,0.0,0.0],[1.8897,0.0,0.0],[0.0,1.8897,0.0]]]
# # OHH
# sample1WaterWalkers=np.array(sample1WaterWalkers)
# print("Result: ", PotentialEnergySingleWater(sample1WaterWalkers))
# print("End test. \n \n")



# print("Testing Potential for walkers with two waters")
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

for i in range(100):
	atom1 = [[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,3.0]]
	atom1 = np.array(atom1)
	atom2 = [[1.0*(i+2),0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,1.0]]
	atom2 = np.array(atom2)
	VinterSum, coloumbicEnergySum, lennardJonesSum  = PotentialEnergyTwoWaters(atom1, atom2)
	print(coloumbicEnergySum)
	plt.scatter(i,  coloumbicEnergySum)
	plt.xlim(4, 100)
	# plt.ylim(-0.002, 0.0)

plt.show()
	
# print("Result: ",PotentialEnergyManyWaters(sample2WaterWalkers))
print("End test. \n \n")
