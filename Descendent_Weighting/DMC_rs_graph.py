# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 03/01/20

# This library of functions lets us graph various data files

import numpy as np
import sys
import matplotlib.pyplot as plt
import DMC_rs_lib as lib

n_bins = 50

def avg_hist_2(filename,num_files):
    for i in range(num_files):
        walk = np.load(filename+f'_{i}.npy',allow_pickle=True)
        plt.figure(i)
        plt.xlabel('Walker Oxygen Bond Angle')
        plt.ylabel('Density')
        plt.title(f'Wave Function of Oxygen Angles at {i+1} million steps')
        plt.bar(align='edge', width=1.5, linewidth=0, **lib.avg_hist(walk),alpha=.6)

    #plt.show()

def avg_hist(filename,num_files):
    walkers = []
    for i in range(num_files):
        walk = np.load(filename+f'_{i}.npy',allow_pickle=True)
        for j in range(len(walk)):
            walkers.append(walk[j])
        plt.figure(i)
        plt.xlabel('Walker Oxygen Bond Angle')
        plt.ylabel('Density')
        plt.title(f'Wave Function of Oxygen Angles after {i+1} million steps')
        plt.bar(align='edge', width=1.5, **lib.avg_hist(walkers))
    plt.show()

def plot_wave_functions_2(filename,num_files):
    for i in range(num_files):
        o1, o2, o3 = oxy_ang(filename+f'_{i}.npy')

        total_o = np.concatenate((o1,o2,o3),axis=0)
        plt.figure(i)
        plt.hist(total_o,bins=n_bins,density=True,alpha=.6)
        plt.xlabel('Walker Oxygen Bond Angle')
        plt.ylabel('Density')
        plt.title(f'Wave Function of Oxygen Angles at {i+1} million steps')

    #plt.show()

def plot_wave_functions(filename,num_files):
    total_o = []
    for i in range(num_files):
        o1, o2, o3 = oxy_ang(filename+f'_{i}.npy')
        '''
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
        '''
        plt.figure(i)
        total_o = np.concatenate((total_o,o1,o2,o3),axis=0)
        plt.hist(total_o,bins=n_bins,density=True)
        plt.xlabel('Walker Oxygen Bond Angle')
        plt.ylabel('Density')
        plt.title(f'Wave Function of Oxygen Angles after {i+1} million steps')

    plt.show()

# removes the smallest k and largest k oxygen angles from each angle
# we could also remove by walker, this might be slightly different but 
# should equate to almost the same thing
def rm_outliers(o1,o2,o3,k):
    o1_ind = np.argsort(o1)
    o1 = o1[o1_ind[k:-k]]

    o2_ind = np.argsort(o2)
    o2 = o2[o2_ind[k:-k]]

    o3_ind = np.argsort(o3)
    o3 = o3[o3_ind[k:-k]]

    return o1, o2, o3


def oxy_ang(filename):

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

        #o1, o2, o3 = rm_outliers(o1, o2, o3,1)

        o_ang_1 = np.concatenate((o_ang_1,o1),axis=0)
        o_ang_2 = np.concatenate((o_ang_2,o2),axis=0)
        o_ang_3 = np.concatenate((o_ang_3,o3),axis=0)
    

    '''
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
    plt.show()
    '''
    
    return o_ang_1, o_ang_2, o_ang_3
    

def plot_h_dist_2(filename,num_files):

    for i in range(num_files):
        h_dist = calc_h_dist(filename+f'_{i}.npy')

        plt.figure(i)
        plt.hist(h_dist,bins=n_bins,density=True)
        plt.xlabel('Hydrogen Distance from Oxygen Plane')
        plt.ylabel('Density')
        plt.title(f'H Dist Wave Function at {i+1} million time steps')

    plt.show()

def plot_h_dist(filename,num_files):
    total_h = []
    for i in range(num_files):
        h_dist = calc_h_dist(filename+f'_{i}.npy')

        total_h = np.concatenate((total_h,h_dist),axis=0)

        plt.figure(i)
        plt.hist(total_h,bins=n_bins,density=True)
        plt.xlabel('Hydrogen Distance from Oxygen Plane')
        plt.ylabel('Density')
        plt.title(f'H Dist Wave Function after {i+1} million time steps')
    plt.show()

def calc_h_dist(filename):
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

    return h_dist

def graph_dw(filename,num_files):
    for i in range(num_files):
        o1, o2, o3 = oxy_ang(filename+f'_{i}.npy')
        total_o = np.concatenate((o1,o2,o3),axis=0)
        dw_weights = np.load(filename+f'_{i}_dw.npy',allow_pickle=True)

        weights = []
        for j in range(len(dw_weights)):
            weights = np.concatenate((weights,dw_weights[j]),axis=0)

        plt.figure(i)
        plt.hist(total_o,bins=n_bins,density=True,weights=np.tile(weights,3),alpha=.6)
        plt.xlabel('Walker Oxygen Bond Angle')
        plt.ylabel('Density')
        plt.title(f'DW Wave Function of Oxygen Angles at {i+1} million steps')
    
    plt.show()

def graph_dw_avg(filename,num_files):
    for i in range(num_files):
        walk = np.load(filename+f'_{i}.npy',allow_pickle=True)
        dw_weights = np.load(filename+f'_{i}_dw.npy',allow_pickle=True)
        plt.figure(i)
        plt.xlabel('Walker Oxygen Bond Angle')
        plt.ylabel('Density')
        plt.title(f'DW Wave Function of Oxygen Angles at {i+1} million steps')
        plt.bar(align='edge', width=1.5, linewidth=0,**lib.avg_hist(walk,dw_list=dw_weights),alpha=.6)
    plt.show()



if __name__ == '__main__':
    #avg_hist_2(sys.argv[1],int(sys.argv[2]))
    avg_hist(sys.argv[1],int(sys.argv[2]))
    #plot_wave_functions(sys.argv[1],int(sys.argv[2]))
    #plot_wave_functions_2(sys.argv[1],int(sys.argv[2]))

    #plot_h_dist_2(sys.argv[1],int(sys.argv[2]))
    #plot_h_dist(sys.argv[1],int(sys.argv[2]))

    #graph_dw(sys.argv[1],int(sys.argv[2]))
    #graph_dw_avg(sys.argv[1],int(sys.argv[2]))
    '''
    if len(sys.argv) < 3:
        print('Usage: dmc_rs_graph.py filename num_files')
        sys.exit(0)
    plot_wave_functions(sys.argv[1],int(sys.argv[2]))
    plot_h_dist(sys.argv[1])
    '''