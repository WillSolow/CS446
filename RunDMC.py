# Will Solow and Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Goal is to approximate Schrodinger Equation for 2 or more atoms

# import packages
import numpy as np

# Initialize Constants

# time in atomic units
 dt = 10 
 
# duration of the simulation
simLength = 10000

# initial number of walkers in simulation
 nWalkers = 2
 
# spring constant for harmonic oscilator
k = 1.0 
 

# Start simulation for given duration
for i in range(simLength): 
	
# Input: Variable x, the distance between two atoms
# Output: The potential energy in the system
# Uses the equation for potential energy with position and spring constant
def potentialEnergy(x):
	return .5*k*(x-EquilibriumPosition)**2
	
