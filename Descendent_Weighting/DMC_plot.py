# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 03/01/20

# This library of functions lets us graph various data files

import numpy as np
import sys
import matplotlib.pyplot as plt

for i in range(15):
    ref = np.load(f'monomer/monomer_ln_{i}/sim{i}_refenergy.npy')

    for j in range(20):
    # Save the ref energy and avg ref energy from 50k, 100k, ... , 1000k
    np.savetxt(f'{output_filename}_ref_{(j+1)*50000}', [ref_energy[(j+1)*50000], np.mean(ref_energy[j*50000:(j+1)*50000]), 
                np.mean(ref_energy[int(((j+1)*50000)/2):(j+1)*50000])])