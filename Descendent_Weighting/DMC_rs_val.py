# Will Solow, Skye Rhomberg
# CS446 Spring 2021
# Test PE functions 
# Script Style
# Last Updated 02/28/2021

# This is a library of the scientific constants and functions used in our DMC simulations
# Everything in here should be constant across all simulations which import this file

# Imports
import numpy as np
import itertools as it
import DMC_rs_lib as lib
import sys


# Start of validation here, everything above is solely from copied files to keep the same PE functions
# INPUT is filename, number of molecules for trimer/dimer/monomer
filename = sys.argv[1]

num_mol = int(sys.argv[2])

# load the walker array
walk_array = np.load(filename+'.npy')

print(walk_array)

np.save(filename+'_20',walk_array[:20])

# reshape the walker array
walk_reshape = np.reshape(walk_array, (walk_array.shape[0], num_mol, 3, 3))

print(walk_reshape)

intra_PE = lib.intra_pe(walk_reshape)
if num_mol > 1:
    inter_PE , _, _, = lib.inter_pe(walk_reshape)

total_PE = lib.total_pe(walk_reshape)

np.savetxt(filename+'_intra', intra_PE)

if num_mol > 1:
    np.savetxt(filename+'_inter', inter_PE)

np.savetxt(filename+'_total',total_PE)