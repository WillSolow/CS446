# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 03/01/20

# This library of functions lets us graph various data files

import numpy as np
import sys
import matplotlib.pyplot as plt

n_bins = 50

def plot_oxy_ang(filename):

    wave_func_out = np.load(filename,allow_pickle=True)

    o_ang_1 = []
    o_ang_2 = []
    o_ang_3 = []
    # Loop over all wave function snapshots and calculate the oxygen angles
    for i in range(len(wave_func_out)):
        oxy = wave_func_out[i][:,:,0]
        oxy_vec_10 = oxy[:,1]-oxy[:,0]
        oxy_vec_20 = oxy[:,2]-oxy[:,0]
        oxy_vec_21 = oxy[:,2]-oxy[:,1]
        oxy_ln_10 = np.linalg.norm(oxy_vec_10, axis=1)
        oxy_ln_20 = np.linalg.norm(oxy_vec_20, axis=1)
        oxy_ln_21 = np.linalg.norm(oxy_vec_21, axis=1)

        o1 = (180/np.pi)*np.arccos(np.sum(oxy_vec_10*oxy_vec_20, axis=1) / \
            (oxy_ln_10*oxy_ln_20))
        o2 = (180/np.pi)*np.arccos(np.sum(-oxy_vec_10*oxy_vec_21, axis=1) / \
            (oxy_ln_10*oxy_ln_21))
        o3 = (180/np.pi)*np.arccos(np.sum(-oxy_vec_20*-oxy_vec_21, axis=1) / \
            (oxy_ln_20*oxy_ln_21))
    
        o_ang_1 = np.concatenate((o_ang_1,o1),axis=0)
        o_ang_2 = np.concatenate((o_ang_2,o2),axis=0)
        o_ang_3 = np.concatenate((o_ang_3,o3),axis=0)

    plt.figure(1)
    plt.hist(o_ang_1,bins=n_bins,density=True)
    plt.xlabel('Walker Oxygen Bond Angle')
    plt.ylabel('Density')
    plt.title(f'Density of Oxygen Angle 1')

    plt.figure(2)
    plt.hist(o_ang_2,bins=n_bins,density=True)
    plt.xlabel('Walker Oxygen Bond Angle')
    plt.ylabel('Density')
    plt.title(f'Density of Oxygen Angle 2')

    plt.figure(3)
    plt.hist(o_ang_3,bins=n_bins,density=True)
    plt.xlabel('Walker Oxygen Bond Angle')
    plt.ylabel('Density')
    plt.title(f'Density of Oxygen Angle 3')

    plt.figure(4)
    plt.hist(np.concatenate((o_ang_1,o_ang_2,o_ang_3),axis=0),bins=n_bins,density=True)
    plt.xlabel('Walker Oxygen Bond Angle')
    plt.ylabel('Density')
    plt.title('Density of all Oxygen Angles')

    

def plot_h_dist(filename):
    wave_func_out = np.load(filename,allow_pickle=True)

    h_dist = []
    for i in range(len(wave_func_out)):
        walk = wave_func_out[i]

        # calculate two vectors on the plane
        v1 = walk[:,0,0] - walk[:,1,0]
        v2 = walk[:,0,0] - walk[:,2,0]

        # compute the normal vector to the oxygen plane
        n = np.cross(v1,v2)

        # get every hydrogen atom
        hydrogens = walk[:,:,1:]

        # dot product of the normal vector with each hydrogen atom
        dp = np.sum(n[:,None,None,:]*hydrogens,axis=3) + np.sum(n*walk[:,0,0],axis=1)[:,None,None] \
            / np.linalg.norm(n,axis=1)[:,None,None]
        
        h_dist = np.concatenate((h_dist,dp.flatten()),axis=0)

    plt.figure(5)
    plt.hist(h_dist,bins=n_bins,density=True)
    plt.xlabel('Hydrogen Distance from Oxygen Plane')
    plt.ylabel('Density')
    plt.title('Density of All Hydrogen Distances')
    plt.show()






if __name__ == '__main__':
    plot_oxy_ang(sys.argv[1])
    plot_h_dist(sys.argv[1])