import numpy as np

def third_water(walkers):
    '''In: walkers array for dimer (nWalkers,2,3,3)
    Out: Coordinate offsets to get third water in trimer in proper position (3,)
    This will be the position of the third oxygen, we assume hydrogens will follow
    Add these coords to the FIRST molecule in all the walkers and stack
    '''
    # Average Oxygen Positions,s u,v each (3,)
    avg_walker = np.mean(walkers,axis=0)
    print(avg_walker)
    u,v = avg_walker[:,0,:]
    print(u)
    print(v)
    # Midpoint between u,v (3,)
    m = (u + v)/2 
    # Orthogonal vector to midpoint
    m0 = v - m
    p = np.array([-m0[1],m0[0],0])
    # Position of third water
    t = p * np.linalg.norm(m0) * np.sqrt(3) / np.linalg.norm(p) + m
    # Offset for u
    return t - u
   
# Input (nWalkers, 2, 3, 3) array
# Output (nWalkers, 3, 3, 3) array for the water trimer
def alt_third_water(walkers):
    # Find positions of each oxygen in the walkers
    oxygen_pos = walkers[:,:,0,:]
    midpoints = (oxygen_pos[:,0]+oxygen_pos[:,1])/2
    print(midpoints.shape)
    
    m0 = oxygen_pos[:,1]-midpoints
    print(m0.shape)
    p = np.stack((-m0[:,1],m0[0],np.zeros(m0.shape[0])),axis=-1)
    print(p.shape)
    t = p*np.linalg.norm(m0,axis=1) * sqrt(3) / np.linalg.norm(p,axis=1) + m
    # Find the midpoint coordinate between the two oxygen atoms
    #midpoint = (walkers[:,0,0]+walkers[:,1,0])/2
    #distance = np.linalg.norm(walkers[:,0,0]-walkers[:,1,0])
    offset = t - oxygen_pos[:0]
    print(offset)
    
    
    new_water = walkers[:,0] + offset
    return np.stack((walkers, new_water), axis=1)


# Test on Two Water Equilibriated Array
def main():
    walkers = np.load('two_water_eq_4.npy')
    print(third_water(walkers))
    alt_third_water(walkers)

if __name__ == '__main__':
    main()
