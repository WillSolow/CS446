# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 03/01/20

# Function for printing XYZ output of a given walker array
# This only works for walkers with homogeneous molecules
# Also had to hard-code the sig digs (it's 8 right now)
# To change that, change the 8 in the ".8f" on line 27

# Added descendent weighting output
# Comment line now shows ancestor


import numpy as np

def print_xyz(walkers, ancestors = None, atoms = ['O','H','H'], \
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
        f'{walkers.shape[2]}\n{ancestors[i] if ancestors is not None else comment}\n'
        f'''{nl.join( [atoms[c % walkers.shape[2]] + tb 
            + tb.join( [f"{el:.8f}" for el in row] ) 
            for c,row in enumerate( walkers[i,...].reshape((-1,coord_cnst)) )] )}'''
        ) for i in range(walkers.shape[0]) ] )

def print_csv(walkers):
    '''
    Could be implemented if desired
    '''
    pass

def print_arr(walkers,ext,**kwargs):
    '''
    Input: walkers: ndarray shape=(nWalkers,nMolecules,nAtoms,coord_const)
           ext: string --> file extension "xyz" or "csv"
    Output: string --> xyz or csv formatted output for walkers
    '''
    if ext == 'xyz':
        return print_xyz(walkers,**kwargs)
    elif ext == 'csv':
        return print_csv(walkers)
    else:
        print('Error: Invalid Extension for Printing')
        exit()

def write_array(filename,ext='xyz',**kwargs):
    '''
    Input: filename: str --> name of input file to load (no extension needed)
           ext: str --> output file type 'xyz' or 'csv'
    Output: None
           filename.ext written with proper formatted output of walker array loaded from input
    '''
    # Format filename -- remove ext
    filename = filename.strip('./').split('.')[0]
    with open(filename+'.'+ext,'w') as fl:
        wlk = np.load(filename+'.npy')
        fl.write(print_arr(wlk,ext,**kwargs))

