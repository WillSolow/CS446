# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Updated 10/27/20

# The purpose of this code is to validate that the replication and deletion phases
# are working correctly in the simulation loop

import numpy as np

# This first test validates that the boolean indexing we are doing works correctly

print('Test 1\n')
# Intial test array of walkers
walkers = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print('Walkers: ', walkers)

# Create a set of thresholds for replication and deletion
thresholds = np.random.rand(walkers.shape[0])
thresholds[5] = .5
print('\n\nThresholds: ', thresholds)


walkers_to_remain = np.invert(thresholds > .5)
print('\n\nBoolean Remain: ', walkers_to_remain)


walkers_after_delete = walkers[walkers_to_remain]
print('\n\nWalkers after Delete: ', walkers_after_delete)

print('\n\nWalkers Array after Deletion: ', walkers)


walkers_to_replicate = (thresholds > .5)
print('\n\nBoolean Replicate: ', walkers_to_replicate)


walkers_after_replication = walkers[walkers_to_replicate]
print('\n\nWalkers after Replication: ', walkers_after_replication)


walkers = np.append(walkers_after_delete, walkers_after_replication)
print('\n\nFinal Walker Array: ', walkers)
