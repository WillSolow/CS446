# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 11/05/20

# This code tests how to calculate the Oxygen bond angles in the water trimer system

import numpy as np
import matplotlib as plt

# Initialize test array with 1 walker, 3 water molecules with oxygens in positions so that
# they will form a 45-45-90 triangle
walkers = np.array([ [ [ [0,0,0] , [0,0,0] ], [ [1,0,0], [0,0,0] ], \
                        [ [0,1,0], [0,0,0] ] ], [ [ [1,0,0] , [0,0,0] ], \
                        [ [0,1,0], [0,0,0] ], \
                        [ [0,0,1], [0,0,0] ] ] ])
print('walkers:\n', walkers)


# Get only the oxygen atoms
oxygen_atoms = walkers[:,:,0]
print('\noxygen atoms\n', oxygen_atoms)
#print('oxygen atoms shape: ', oxygen_atoms.shape)

# Get the pairs of the vectors for subtraction to find the norm
vec_pair_1 = oxygen_atoms[:,[0,0,1]]
vec_pair_2 = oxygen_atoms[:,[1,2,2]]

#print('Vec pair 1: \n', vec_pair_1)
#print('Vec pair 2: \n', vec_pair_2)

OO_vectors = vec_pair_1-vec_pair_2
#print('Oxygen vectors: \n', OO_vectors)
#print('Oxygen vectors shape: ', OO_vectors.shape)

# calculate the magnitude of each vector pair
OO_lengths = np.linalg.norm(OO_vectors, axis=2)
print('\noxygen vector lengths:\n ', OO_lengths)

angle_1 = np.arccos(np.sum(-OO_vectors[:,0]*-OO_vectors[:,1], axis=1) / \
                (OO_lengths[:,0]*OO_lengths[:,1]))

angle_2 = np.arccos(np.sum(OO_vectors[:,0]*-OO_vectors[:,2] , axis=1) / \
                (OO_lengths[:,0]*OO_lengths[:,2]))

angle_3 = np.arccos(np.sum(OO_vectors[:,2]*OO_vectors[:,1]  , axis=1) / \
                (OO_lengths[:,2]*OO_lengths[:,1]))
                
print('\nAngle 1: ', angle_1)
print('Angle 2: ', angle_2)
print('Angle 3: ', angle_3)