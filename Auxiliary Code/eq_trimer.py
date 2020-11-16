import numpy as np

def third_water(walkers):
    '''In: walkers array for dimer (nWalkers,2,3,3)
    Out: Coordinate offsets to get third water in trimer in proper position (3,)
    This will be the position of the third oxygen, we assume hydrogens will follow
    Add these coords to the FIRST molecule in all the walkers and stack
    '''
    # Average Oxygen Positions,s u,v each (3,)
    avg_walker = np.mean(walkers,axis=0)
    u,v = avg_walker[:,0,:]
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

if __name__ == '__main__':
    main()
