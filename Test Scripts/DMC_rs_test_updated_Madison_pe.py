# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 11/13/20

# This code demonstrates the correctness of Madison's updated PE
# function to add support for multiple walkers


import numpy as np

np.set_printoptions(suppress=True)
    
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
    # Only calculate intermolecular PE if there is more than 1 water molecule
    if nWaters > 1:
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
    (nAtoms1,nCartesian1)=water1pos.shape
    (nAtoms2,nCartesian2)=water2pos.shape

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
    

# Input: Two arrays of size (3,3) representing the xyz coordinates of each atom
#        in the water molecule
# Output: The intermolecular potential energy between the two water moolecules
def SR_PotentialEnergyTwoWaters(water1, water2):

    # Get the shape of both water molecules. Note that they will always be 
    # 3 by 3 in the case of the water molecule.
    (nWalkers1, nAtoms1, nCartesian1)=water1.shape
    (nWalkers2, nAtoms2, nCartesian2)=water2.shape
    
    # Initial Epsilon and Sigma constant
    epsilon = 0.1554252
    sigma = 3.165492

    # Converted Epsilon and Sigma constant
    epsilon = epsilon*(4.184/2625.5)
    sigma = sigma/0.529177

    
    # Initialize energy lists
    potentialEnergyList = np.zeros(nWalkers1)
    coloumbicEnergyList = np.zeros(nWalkers1)
    lennardJonesList = np.zeros(nWalkers1)

    # For every atom in water 1
    for atomNum1 in range(nAtoms1):
        # For every atom in water 2
        for atomNum2 in range(nAtoms2):
        
            # Get the position index [0, 1, 2] of atom 1 and 2
            atom1 = water1[:,atomNum1]
            atom2 = water2[:,atomNum2]
            
            # Calculate the distance between atoms
            distance = np.linalg.norm(water1[:,atomNum1]-water2[:,atomNum2], axis=1)
            
            # If the distance is not 0, calculate QiQj
            # Calculate Coulombic energy
            coloumbicV = coloumbic(atomNum1, atomNum2, distance)
            
            # If the distance is 0, we will get a np.inf value in the array, convert this to 0
            # as 0 distance means 0 PE
            coloumbicV = np.where(np.abs(coloumbicV) == np.inf, 0, coloumbicV)
                
            # If both atoms are oxygen, calculate the lennard jones energy
            if atomNum1 == 0 and atomNum2 == 0:
                lennardJones = 4*epsilon*((sigma/distance)**12 \
                                   - (sigma/distance)**6)
                # If the distance is 0, we will get a np.inf value in the array, convert this to 0
                # as 0 distance means 0 PE
                lennardJones = np.where(np.abs(lennardJones) == np.inf, 0, lennardJones)
                
                lennardJonesList = lennardJonesList+lennardJones
                                   
                # Calculate the intermolecular potential energy              
                potential = lennardJones + coloumbicV
                    
            # If not both oxygen, then potential energy is just Coulombic energy
            else:
                potential = coloumbicV
                
            potentialEnergyList = potentialEnergyList + potential
            coloumbicEnergyList = coloumbicEnergyList + coloumbicV
                
    
    # Sum up all energies
    # Note that we do not need to sum up all the energies anymore as they are getting summed
    # in each addition step
    VinterSum = potentialEnergyList
    coloumbicEnergySum = coloumbicEnergyList 
    lennardJonesSum = lennardJonesList


    return VinterSum, coloumbicEnergySum, lennardJonesSum
    
    



test_walkers = np.random.rand(10, 1, 3, 3)  

water1 = np.random.rand(5,3,3)
water2 = np.random.rand(5,3,3)

for i in range(water1.shape[0]):
    pe, col, lj = PotentialEnergyTwoWaters(water1[i], water2[i])
    print('Madison Inter PE : ', pe)
    print('Madison Coloumbic: ', col)
    print('Madison Lennard  : ', lj)
    
sr_pe, sr_col, sr_lj = SR_PotentialEnergyTwoWaters(water1, water2)
print('SR M Inter PE   : ', sr_pe)
print('SR M coloumibc  : ', sr_col)
print('SR M lennard    : ', sr_lj)


    
