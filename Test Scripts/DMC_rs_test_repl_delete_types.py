# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Updated 10/29/2020

# This code prints to a text file the result of many deletions and replication of 
# arbitrary walkers. The purpose of this code is to validate that the vectorized 
# replication and deletion method used in the DMC algorithm is correct in choosing the right
# walkers to replicate or delete. 

# NOTE: Print to a file, not to the command line or you will ruin your day

import numpy as np

# Initial Constants
electron_mass = 9.10938970000e-28

avogadro = 6.02213670000e23


oxygen_mass = 15.99491461957
hydrogen_mass = 1.007825

# Calculate the atomic masses of the atoms in a molecule
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)

# Testing Sizes
# Input: Number of walkers, flag for fprint statement
# Output: A string that states whether the replications are correct or not
def test_del_repl(wlk_size=4,verbose='True'):

    # Initialize the dimension of the 4D walker array
    n_walkers, num_molecules, coord_const = wlk_size,3,3
    print(f'Testing Replications & Deletions: {n_walkers} walkers, {num_molecules} waters')


	
    # Initialize the walker array with random values in the range [-.5, 5)
    walkers = (np.random.rand(n_walkers, num_molecules, atomic_masses.shape[0], \
        coord_const) - .5) 
    print(f'Initial Walker Array (Random):\n{walkers}\n' if verbose else '')
    print(f'Walker Array Specs:\n\tshape: {walkers.shape}\n\ttype: {walkers.dtype}\n')

	
	
    # Create sample thresholds and probabilities to mimic that in the DMC algorithm
	# The purpose is to show that the replication and deletion conditions are mutually
	# exclusive
    thresholds = np.random.rand(walkers.shape[0])
    prob_delete = np.random.uniform(0.25,1.5,walkers.shape[0])
    prob_replicate = prob_delete - 1
	
	
	
    # Sample Potential Energies: True --> Potential Energy > Ref Energy
    pot_vs_ref = np.random.choice([True,False],walkers.shape[0])
    print( 
        (   f'Sample Probabilities:\n'
            f'Thresholds (uniform 0,1): {thresholds}, type: {thresholds.dtype}\n'
            f'Delete Prob (uniform 0.15,1.5): {prob_delete}, type: {prob_delete.dtype}\n'
            f'Repl Prob (above -1): {prob_replicate}, type: {prob_replicate.dtype}\n'
            f'Pot vs Ref Energy (boolean -- true >): {pot_vs_ref}, '
            f'type: {pot_vs_ref.dtype}\n'
        ) )

		
		
    # Candidates for Deletion/Replication should be mutually exclusive
    to_delete = prob_delete < thresholds
    to_replicate = prob_replicate > thresholds
    print(
        (   f'Candidates for Deletion and Replication must be mutually exclusive\n'
            f'Deletion (p_d < thres): {to_delete}, type: {to_delete.dtype}\n'
            f'Replication (p_r > thres): {to_replicate}, type: {to_replicate.dtype}\n'
        ) ) 

		
		
    # Walkers Deleted: Pot > Ref AND del < thres 
    walkers_to_remain = np.invert( (pot_vs_ref) * to_delete )
	
    # Should be all the rest
    walkers_after_delete = walkers[walkers_to_remain]
    print(
        (   f'Deletion Step: Pot-v-Ref AND To_Del for deletion\n'
            f'Mask showing Remaining: {walkers_to_remain}, '
            f'type: {walkers_to_remain.dtype}\n'
            f'Indices Deleted: {np.invert(walkers_to_remain)}\n'
            f'After:\n{walkers_after_delete if verbose else ""}'
            f'\nshape: {walkers_after_delete.shape}, '
            f'type: {walkers_after_delete.dtype}\n'
        ) )

		
		
    # Walkers Replicated: Pot < Ref AND repl > thres
    walkers_to_replicate = ( np.invert(pot_vs_ref) ) * to_replicate
	
    # Should be ONLY replications
    walkers_replicated = walkers[walkers_to_replicate]
    print(
        (   f'Replication Step: NOT Pot-v-Ref AND To_Repl for replication\n'
            f'Mask showing Repl: {walkers_to_replicate}, '
            f'type: {walkers_to_replicate.dtype}\n'
            f'After:\n{walkers_replicated if verbose else ""}\n'
            f'shape: {walkers_replicated.shape}, '
            f'type: {walkers_replicated.dtype}\n'
        ) )

		

    # New Walkers
    new_walkers = np.append(walkers_after_delete, walkers_replicated, axis=0)
    print(f'New Walker Array:\n{new_walkers}' if verbose else '')
    print(f'New Array Specs:\n\tshape: {new_walkers.shape}\n\ttype: {new_walkers.dtype}\n')

	
	
    # Test Conditions:
    repl_ex_del = all(b==False for b in np.invert(walkers_to_remain)*walkers_to_replicate)
    num_correct = np.sum(walkers_to_remain)+np.sum(walkers_to_replicate)==new_walkers.shape[0]
    print(
        (   f'Replication and Deletion Mutually Exclusive: '
            f'{repl_ex_del}\n'
            f'Correct Number of Walkers in New Array: '
            f'{num_correct}\n'
        ) )
    return repl_ex_del, num_correct

batch_test = [test_del_repl(1000,False) for i in range(1000)]
print(f'Batch Test Passed: {all( a for a in list( sum( batch_test, () ) ) )}')
