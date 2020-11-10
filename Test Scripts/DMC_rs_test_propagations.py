# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 10/27/20

# This function tests the tiling method of the walker propagations 

import numpy as np

# Create a random array of walkers by molecules by atoms by coord const 
walkers = (np.random.rand(4, 1, 3, 3)-.5)
print('Walkers: \n', walkers)


# Create an array of "atomic masses" to demonstrate the tiling feature
offset = np.array([.0001, 10, 100000])

# Create an array of propogation values using the same method in the main code file
# with size of the walkers array
propogations = np.random.normal(0, np.transpose(np.tile(offset, (4, 1, 3, 1)) \
    , (0,1,3,2)))
print('\n\n\nPropagations\n', propogations)

# For each value in offset, get a normal distribution
atom_props = [np.random.normal(0, offset[i], (4, 1, 3)) for i in range(offset.shape[0])]

# Stack the propagates values together to create the entire 4d array
propagations2 = np.stack( atom_props, axis = 2)
#print('\n\npropagations2\n', propagations2)

# Output is the new walkers array based on the propagation lengths
output = walkers + propogations
print('\n\nOutput\n', output)

# Output 2
output2 = walkers + propagations2
#print('\n\nOutput2\n', output2)