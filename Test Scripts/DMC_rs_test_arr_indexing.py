# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Updated 10/27/20

# The purpose of this code is to demonstrate that our array indexing for the 
# distinct molecule pairings of the water molecules in a walker works correctly

import numpy as np

# This first test validates that the boolean indexing we are doing works correctly


# Number of molecules in the system, typically 3 for the water trimer
n = 5

# When combinging the lists below pointwise, we get the equivalent of n choose 2
# or the list of distinct pairs of molecules, as is needed to calculate the 
# intermolecular potential energy

print('Num Molecules: ', n)
print('There should be ' + str(int(n*(n-1)/2)) + ' distinct pairs\n\n')

# Returns a list of the first molecule in the distinct pairs
pair_1 = np.array(sum([[i]*(n-(i+1)) for i in range(n-1)],[]))

# Returns a list of the second molecule in the distinct pairs
pair_2 = np.array(sum([list(range(i,n)) for i in range(1,n)],[]))



print('Pair 1: ', pair_1)
print('\nPair 2: ', pair_2)
