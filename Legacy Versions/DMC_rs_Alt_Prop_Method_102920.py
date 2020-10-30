# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style

# This file gives another alternative to the tiling method we present in the main body of code
# Note that both are equivalent and while the tiling method is less immediately intuitive,
# they both produce the same results and the tiling one is faster and cleaner


# Propagates each coordinate of each atom in each molecule of each walker within a normal
# distribution given by the atomic mass of each atom.
# This method  does the same as above but it is a little more straightforward to see the 
# correctness. Both return the same results, and the above is faster
propagate_atoms = [np.random.normal(0, np.sqrt(dt/atomic_masses[i]), (walkers.shape[0],\
    num_molecules, coord_const)) for i in range(atomic_masses.shape[0])]
propagations = np.stack(propagate_atoms, axis = 2)

