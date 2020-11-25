# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 10/30/20

# The purpose of this file is to calculate the Normalization Constant for the wave function
# in the histogram of walker positions

# Output: Normalization constant N to the terminal

# Imports
import numpy as np
import scipy.integrate as integrate


# Initial Constants

# Mass of an electron
electron_mass = 9.10938970000e-28
# Avogadro's constant
avogadro = 6.02213670000e23

# Atomic masses of atoms in system
# Used to calculate the atomic masses in Atomic Mass Units
oxygen_mass = 15.994915
hydrogen_mass = 1.007825


# Spring constant of the bond to be graphed
k = 6.027540

# Atoms in the system. 
# This should only be the atoms that are present in the bond interaction that is
# being graphed in the histogram
# For example in a H2O molecule when graphing the OH bond, the only atoms present are 
# one Oxygen and one Hydrogen atom
atomic_masses = np.array([oxygen_mass, hydrogen_mass, hydrogen_mass]) / (avogadro * electron_mass)

# Reduced mass of the system
# Used in calculating the normalization constant in the wave function
reduced_mass = ((atomic_masses[0]+atomic_masses[1])*atomic_masses[2])/np.sum(atomic_masses)


# Part of the wave function. Used in integration to solve for the normalization constant
# under the assumption that the integral should be 1.
wave_func = lambda x: np.exp(-(x**2)*np.sqrt(k*reduced_mass)/2)

# Get the integral of the wave function and the error
integral_value, error = integrate.quad(wave_func, -np.inf, np.inf)

# Calculate the Normalization constant
N = 1 / integral_value

print('Normalization constant:')
print(N)
