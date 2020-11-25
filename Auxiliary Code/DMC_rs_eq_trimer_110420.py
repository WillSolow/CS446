# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 11/24/20

# This code takes a 4D array of equilibrated 2 water molecule systems and outputs
# coordinates to for the offset of the 3rd water molecule. This code is mostly
# auxiliary and allows us to equilibrate a water trimer system

import numpy as np

def third_water(walkers):
    '''In: walkers array for dimer (nWalkers,2,3,3)
    Out: Coordinate offsets to get third water in trimer in proper position (3,)
    This will be the position of the third oxygen, we assume hydrogens will follow
    Add these coords to the FIRST molecule in all the walkers and stack
    '''
    # Average Oxygen Positions,s u,v each (3,)
    avg_walker = np.mean(walkers,axis=0)
    #print(avg_walker)
    u,v = avg_walker[:,0,:]
    #print(u)
    #print(v)
    # Midpoint between u,v (3,)
    m = (u + v)/2 
    # Orthogonal vector to midpoint
    m0 = v - m
    p = np.array([-m0[1],m0[0],0])
    # Position of third water
    t = p * np.linalg.norm(m0) * np.sqrt(3) / np.linalg.norm(p) + m
    # Offset for u
    return t - u
  


# Test on Two Water Equilibriated Array
def main():
    walkers = np.load('two_water_eq_4.npy')
    print(third_water(walkers))
    alt_third_water(walkers)

if __name__ == '__main__':
    main()
