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


def intra_bond_length2(filename,num_files):
    for i in range(num_files):
        walk = np.load(f'{filename}/{filename}_{i}.npy',allow_pickle=True)

        OH_positions1 = np.linalg.norm(walk[:,0,0]-walk[:,0,1], axis = 2)
        OH_positions2 = np.linalg.norm(walk[:,1,0]-walk[:,1,1], axis = 2)
        OH_positions3 = np.linalg.norm(walk[:,2,0]-walk[:,2,1], axis = 2)

        total_oh = np.concatenate((OH_positions1,OH_positions2,OH_positions3),axis=0)

        plt.figure(i)
        plt.xlabel('Walker OH Bond Length')
        plt.ylabel('Density')
        plt.title(f'Wave Function of Oxygen ANgles at {i+1} million steps')
        plt.hist(total_oh,bins=n_bins,density=True,alpha=.6)

def intra_bond_angle2(filename,num_files):
    for i in range(num_files):
        total_hoh = hoh_ang(f'{filename}/{filename}_{i}.npy')

        plt.figure(i)
        plt.xlabel('Walker Intra HOH Bond Angle')
        plt.ylabel('Density')
        plt.title(f'Wave Function of HOH angle at {i+1} million steps')
        plt.hist(total_hoh,bins=n_bins,density=True,alpha=.6)
    plt.show()

def intra_bond_angle(filename,num_files):
    total_hoh = []
    for i in range(num_files):
        hoh = hoh_ang(f'{filename}/{filename}_{i}.npy')
        total_hoh = np.concatenate((total_hoh,hoh),axis=0)

        plt.figure(i)
        plt.hist(total_hoh,bins=n_bins,density=True)
        plt.xlabel('Walker Intra HOH Bond Angle')
        plt.ylabel('Density')
        plt.title(f'Wave Function of HOH Angles after {i+1} million steps')

    plt.show()

def hoh_ang(filename):
    walk = np.load(filename,allow_pickle=True)
    total_hoh =[]
    for i in range(len(walk)):
        walkers = walk[i]
        for j in range(3):
            wlks = walkers[:,j]

            oh1_vec = wlks[:,1] - wlks[:,0]
            oh2_vec = wlks[:,2] - wlks[:,0]

            oh1_dist = np.linalg.norm(oh1_vec,axis=1)
            oh2_dist = np.linalg.norm(oh2_vec,axis=1)

            hoh = (180/np.pi)*np.arccos(np.sum(oh1_vec*-oh2_vec,axis=1) / \
                (oh1_dist*oh2_dist))

            total_hoh = np.concatenate((total_hoh,hoh),axis=0)
    
    return total_hoh



def avg_hist_2(filename,num_files):
    for i in range(num_files):
        walk = np.load(f'{filename}/{filename}_{i}.npy',allow_pickle=True)
        plt.figure(i)
        plt.xlabel('Walker Oxygen Bond Angle')
        plt.ylabel('Density')
        plt.title(f'Wave Function of Oxygen Angles at {i+1} million steps')
        plt.bar(align='edge', width=1.5, linewidth=0, **lib.avg_hist(walk),alpha=.6)

    plt.show()

def avg_hist(filename,num_files):
    walkers = []
    for i in range(num_files):
        walk = np.load(f'{filename}/{filename}_{i}.npy',allow_pickle=True)
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
        o1, o2, o3 = oxy_ang(f'{filename}/{filename}_{i}.npy')

        total_o = np.concatenate((o1,o2,o3),axis=0)
        plt.figure(i)
        plt.hist(total_o,bins=n_bins,density=True,alpha=.6)
        plt.xlabel('Walker Oxygen Bond Angle')
        plt.ylabel('Density')
        plt.title(f'Wave Function of Oxygen Angles at {i+1} million steps')

    plt.show()

def plot_wave_functions(filename,num_files):
    total_o = []
    for i in range(num_files):
        o1, o2, o3 = oxy_ang(f'{filename}/{filename}_{i}.npy')
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
        h_dist = calc_h_dist(filename)

        plt.figure(i)
        plt.hist(h_dist,bins=n_bins,density=True)
        plt.xlabel('Hydrogen Distance from Oxygen Plane')
        plt.ylabel('Density')
        plt.title(f'H Dist Wave Function at {i+1} million time steps')

    plt.show()

def plot_h_dist(filename,num_files):
    total_h = []
    for i in range(num_files):
        h_dist = calc_h_dist(filename)

        total_h = np.concatenate((total_h,h_dist),axis=0)

        plt.figure(i)
        plt.hist(total_h,bins=n_bins,density=True)
        plt.xlabel('Hydrogen Distance from Oxygen Plane')
        plt.ylabel('Density')
        plt.title(f'H Dist Wave Function after {i+1} million time steps')
    plt.show()

def calc_h_dist(filename):
    wave_func_out = np.load(f'{filename}/{filename}_{i}.npy',allow_pickle=True)

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
        o1, o2, o3 = oxy_ang(f'{filename}/{filename}_{i}.npy')
        total_o = np.concatenate((o1,o2,o3),axis=0)
        dw_weights = np.load(f'{filename}/{filename}_{i}_dw.npy',allow_pickle=True)

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
        walk = np.load(f'{filename}/{filename}_{i}.npy',allow_pickle=True)
        dw_weights = np.load(f'{filename}/{filename}_{i}_dw.npy',allow_pickle=True)
        plt.figure(i)
        plt.xlabel('Walker Oxygen Bond Angle')
        plt.ylabel('Density')
        plt.title(f'DW Wave Function of Oxygen Angles at {i+1} million steps')
        plt.bar(align='edge', width=1.5, linewidth=0,**lib.avg_hist(walk,dw_list=dw_weights),alpha=.6)
    plt.show()

def plot_ref_energy(filename):
    filepath = f'{filename}/{filename}_refenergy.npy'
    ref_energy = np.load(filepath,allow_pickle=True)
    for i in range(10):
        plt.figure(i)
        avg_ref = np.mean(ref_energy[100*i:100*(i+1)])
        plt.plot(np.arange(1000000*i,1000000*(i+1),10000),ref_energy[100*i:100*(i+1)],label='Ref Energy')
        plt.plot(np.arange(1000000*i,1000000*(i+1),10000),np.tile(avg_ref,100),label=f'Avg Ref Energy: {avg_ref:06f}')
        plt.xlabel('Simulation Time Step')
        plt.ylabel('Reference Energy')
        plt.legend()
        plt.title(f'Reference energy between {i} and {i+1} million time steps')

    plt.figure(10)
    plt.plot(np.arange(0,10000000,10000),ref_energy,label='Ref Energy')
    plt.plot(np.arange(0,10000000,10000),np.tile(np.mean(ref_energy),1000),label=f'Avg Ref Energy: {np.mean(ref_energy):.06f}')
    plt.xlabel('Simulation Time Step')
    plt.ylabel('Reference Energy')
    plt.title(f'Reference energy across all 10 million time steps')
    plt.legend()
    plt.show()

def check_dw(filename, num):
    dw_weights = np.load(f'{filename}/{filename}_{num}_dw.npy',allow_pickle=True)
    walkers = np.load(f'{filename}/{filename}_{num}.npy',allow_pickle=True)
    print(dw_weights.shape)
    for i in range(20):

        oxy = walkers[i][:,:,0]
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


        weight = dw_weights[i]
        print(weight.shape)
        print('Median: ',np.median(weight))
        print('Mean: ',np.mean(weight))
        print('Num of 0: ',np.sum(weight==0))
        print('\n\n')
        plt.figure(i)
        plt.hist(o1,bins=n_bins,density=True,alpha=.6)
        plt.hist(o1,bins=n_bins,density=True,weights=weight,alpha=.6)
        #plt.hist(np.concatenate((o1,o2,o3),axis=0),bins=n_bins,density=True,alpha=.6)
        #plt.hist(np.concatenate((o1,o2,o3),axis=0),bins=n_bins,density=True,weights=np.tile(weight,3),alpha=.6)

    plt.show()

def dw_weights_analysis(filename, num):
    dw_weights = np.load(f'{filename}/{filename}_{num}_dw.npy',allow_pickle=True)

    tot_weight = []
    for i in range(1):
        weight = dw_weights[i]
        tot_weight = np.concatenate((tot_weight,weight),axis=0)
        print(weight.shape)
        print('Median: ',np.median(weight))
        print('Mean: ',np.mean(weight))
        print('Num of 0: ',np.sum(weight==0))
        print('Num in (0,1]: ',np.sum( (weight <= 1) * (weight > 0)))
        print('Num in (1,5]: ',np.sum( (weight <= 5) * (weight > 1)))
        print('Num in (5,10]: ',np.sum( (weight <= 10) * (weight > 5)))
        print('Num in (10,20]: ',np.sum( (weight <= 20) * (weight > 10)))
        print('Num in (20,50]:',np.sum( (weight <= 50) * (weight > 20)))
        print('Num greater than 50:',np.sum(weight > 50))
        print('\n\n')

    plt.hist(tot_weight,bins=n_bins)
    plt.show()



if __name__ == '__main__':
    #avg_hist_2(sys.argv[1],int(sys.argv[2]))
    #avg_hist(sys.argv[1],int(sys.argv[2]))
    #plot_wave_functions(sys.argv[1],int(sys.argv[2]))
    #plot_wave_functions_2(sys.argv[1],int(sys.argv[2]))

    #plot_h_dist_2(sys.argv[1],int(sys.argv[2]))
    #plot_h_dist(sys.argv[1],int(sys.argv[2]))

    #graph_dw(sys.argv[1],int(sys.argv[2]))
    #graph_dw_avg(sys.argv[1],int(sys.argv[2]))

    #plot_ref_energy(sys.argv[1])

    #check_dw(sys.argv[1],int(sys.argv[2]))
    dw_weights_analysis(sys.argv[1],int(sys.argv[2]))

    #intra_bond_angle2(sys.argv[1],int(sys.argv[2]))
    #intra_bond_angle(sys.argv[1],int(sys.argv[2]))
    '''
    if len(sys.argv) < 3:
        print('Usage: dmc_rs_graph.py filename num_files')
        sys.exit(0)
    plot_wave_functions(sys.argv[1],int(sys.argv[2]))
    plot_h_dist(sys.argv[1])
    '''