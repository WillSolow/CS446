# Function for printing XYZ output of a given walker array
# This only works for walkers with homogeneous molecules
# Also had to hard-code the sig digs (it's 8 right now)
# To change that, change the 8 in the ".8f" on line 27

# THIS IS THE GROSSEST THING I'VE EVER WRITTEN

def print_xyz(walkers, atoms = ['C','H','H','H','H'], \
        comment = '### Comment ###', coord_cnst = 3):
    '''
    Input: walkers: ndarray shape=(nWalkers,nMolecules,nAtoms,coord_cnst)
           atoms: list of strings --> order of atoms in a particular molecule
           comment: string --> comment for that walker
           coord_cnst: int --> number of coordinate dimensions (usually 3 for XYZ)
    Output: string --> for each walker:
    nAtoms
    ### Comment ###
    Atom0    X0   Y0   Z0
    Atom1    X1   Y1   Z1
    ...
    AtomN    XN   YN   ZN
    [Blank Line]
    '''
    tb = '\t'
    nl = '\n'
    return '\n\n'.join( [ (
        f'{walkers.shape[2]}\n{comment}\n'
        f'''{nl.join( [atoms[c % walkers.shape[2]] + tb 
            + tb.join( [f"{el:.8f}" for el in row] ) 
            for c,row in enumerate( walkers[i,...].reshape((-1,coord_cnst)) )] )}'''
        ) for i in range(walkers.shape[0]) ] )
