# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Goal: approximate Schrodinger Equation for 2 or more atoms
# This is a change

# Imports
import numpy as np

# Initial Constants
dt = 10

simLength = 10000

nWalkers = 2

k = 1.0

# g/mol
mass = 10

equilibriumPosition = 5

electronMass = 9.10938970000e-28
avogadro = 6.02213670000e+23

reducedMass = (mass / (avogadro * electronMass)) / 2

walkers = 5 + (np.random.rand(nWalkers) - 0.5)
print(walkers)

for i in range(simLength/dt):
    referenceEnergy = np.mean(potentialEnergy(walkers)) + (1.0 - (walkers.shape[0]/nWalkers))/(2.0*dt)
    propogationLengths = np.random.normal(0,np.sqrt(dt/mass),walkers.shape[0])
    walkers = walkers + propogationLengths
    potentialEnergies = potentialEnergy(walkers)
    thresholds = np.random.rand(walkers.shape[0])
    probDelete = np.exp(-(potentialEnergies-referenceEnergy)*dt)
    probReplicate = probDelete - 1
    toReplicate = probReplicate > threshold
    toDelete = probDelete > threshold

    remainAfterDelete = np.argwhere(walker*np.invert((potentialEnergies > referenceEnergy)*toDelete))
    replications = np.argwhere(walkers*(potentialEnergies<referenceEnergy)*toReplicate)
    noChange = np.argwhere(walkers*(potentialEnergies==referenceEnergy))
    walkers = np.concatenate(remainAfterDelete,replications,noChange)

def potentialEnergy(x):
    return .5 * k * (x - equilibriumPosition)**2

print(reducedMass)
