# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 03/01/20

# This library of functions lets us graph various data files

import numpy as np
import sys
import matplotlib.pyplot as plt

destination = sys.argv[1]
output_filename = sys.argv[2]
sim_length = 1000000

ref_energy = np.load(f'{destination}{output_filename}_ref_array.npy')

ref_total = []

for i in range(10):
    np.savetxt(f'{destination}{output_filename}_ref_{(i+1)*1000}', [ref_energy[(i+1)*1000], np.mean(ref_energy[i*1000:(i+1)*1000]), np.mean(ref_energy[int(((i+1)*1000)/2):(i+1)*1000])])
    ref_total.append(np.mean(ref_energy[(i+1)*1000]))
    ref_total.append(np.mean(ref_energy[int(((i+1)*1000)/2):(i+1)*1000]))

for i in range(20):
    if (i+1)*50000 == sim_length:
        np.savetxt(f'{destination}{output_filename}_ref_{(i+1)*50000}', [ref_energy[(i+1)*50000-1], np.mean(ref_energy[i*50000:(i+1)*50000]), np.mean(ref_energy[int(((i+1)*50000)/2):(i+1)*50000])])
        ref_total.append(np.mean(ref_energy[(i+1)*50000-1]))
        ref_total.append(np.mean(ref_energy[int(((i+1)*50000)/2):(i+1)*50000]))
    else:   
        np.savetxt(f'{destination}{output_filename}_ref_{(i+1)*50000}', [ref_energy[(i+1)*50000], np.mean(ref_energy[i*50000:(i+1)*50000]), np.mean(ref_energy[int(((i+1)*50000)/2):(i+1)*50000])])
        ref_total.append(np.mean(ref_energy[(i+1)*50000]))
        ref_total.append(np.mean(ref_energy[int(((i+1)*50000)/2):(i+1)*50000]))


np.savetxt(f'{destination}{output_filename}_ref_total_new', ref_total)