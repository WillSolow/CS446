# New Structured Data Type
# and Conversion Functions for xyz <-> npy

import numpy as np

def make_dtype(n_atoms):
    return np.dtype([('n_atoms','u1'), ('comment', 'U10'), \
            ('atoms', [('id','u1'),('pos', '3f8')], (n_atoms,))])

#def atom_xyz(atom):
#    return f'''{np.squeeze(atom['id'])} {' '.join(np.squeeze(atom['pos']))}'''

#def wlk_xyz(walker):
#    nl = '\n'
#    return f'''{walker['n_atoms']}\n{walker['comment']}\n{nl.join([atom_xyz(a) for a in walker['atoms']])}'''

def print_xyz(walkers):
    strn = lambda l : [str(i) for i in l]
    nl = '\n'
    atom_xyz = lambda a : f"{np.squeeze(a['id'])} {' '.join(strn(np.squeeze(a['pos'])))}"
    wlk_xyz = lambda w : \
        f"{w['n_atoms']}\n{w['comment']}\n{nl.join([atom_xyz(a) for a in w['atoms']])}"  
    return '\n\n'.join([wlk_xyz(w) for w in walkers])

def read_xyz(filename,dt):
    make_atom = lambda l : (int(l[0]),[float(f) for f in l[1:]])
    with open(filename) as f:
        walkers_raw = f.read().strip().split('\n\n')
        walkers = []
        for w in walkers_raw:
            lines = [l.strip() for l in w.split('\n')]
            walkers.append((int(lines[0]),lines[1],\
                    [make_atom(l.split()) for l in lines[2:]]))
        return np.array(walkers,dtype=dt)
