# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 11/4/20

# This code generates graphs of the single water molecule system for the data
# we collected on convergence

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


std10_walk1=[0.00004853,0.00004517,0.00004462,0.00003331,0.00002573]
std10_walk5=[0.00003091,0.00001871,0.00002555,0.00001599,0.00000838]
std10_walk10=[.00002782,0.00001858,0.00001498,0.00000875,0.00001148]

std20_walk1=[0.00004561,0.00004025,0.00003564,0.00001815,0.00000990]
std20_walk5=[0.00002974,0.00001612,0.00001165,0.00000828,0.00000835]
std20_walk10=[0.00001505,0.00000767,0.00000852,0.00000577,0.00000673]

zp10_walk1=[0.062191604,0.062130088,0.062116094,0.061897123,0.061283425]
zp10_walk5=[0.06207806,0.062070326,0.062070945,0.061859571,0.061240847]
zp10_walk10=[0.062068792,0.062066651,0.062060804,0.061855164,0.061240559]

zp20_walk1=[0.062124328,0.062129101,0.062105626,0.061913392,0.061301559]
zp20_walk5=[0.062080402,0.062074165,0.062067523,0.061860314,0.061245072]
zp20_walk10=[0.062059289,0.06206329,0.062060331,0.061851937,0.061237289]

pop_percent=[.9999, .9991, .9965, .9208, .7164]



x1 = [.1,.5,1,5,10]
x = [1,2,3,4,5]
y1 = [10,20]
y = [1,2]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')


ax.axes.set_xticks(x)
ax.axes.set_xticklabels(x1)

ax.axes.set_yticks(y)
ax.axes.set_yticklabels(y1)


ax.plot(x,[1,1,1,1,1],std10_walk1,marker='o',label='1000 Walkers',color='red')
ax.plot(x,[1,1,1,1,1],std10_walk5,marker='o',label='5000 Walkers',color='blue')
ax.plot(x,[1,1,1,1,1],std10_walk10,marker='o',label='10000 Walkers',color='green')
ax.plot(x,[2,2,2,2,2],std20_walk1,marker='o',linestyle='dashed',color='red')
ax.plot(x,[2,2,2,2,2],std20_walk5,marker='o',linestyle='dashed',color='blue')
ax.plot(x,[2,2,2,2,2],std20_walk10,marker='o',linestyle='dashed',color='green')
ax.set_xlabel('Time step')
ax.set_ylabel('Number of Production Phases')
ax.set_zlabel('Zero-Point Energy Standard Deviation')
plt.title('Effects of Constants on Zero-Point Energy Variability')
ax.legend()

fig = plt.figure(2)
ax = fig.add_subplot(111,projection='3d')


ax.axes.set_xticks(x)
ax.axes.set_xticklabels(x1)
ax.axes.set_yticks(y)
ax.axes.set_yticklabels(y1)


ax.plot(x,[1,1,1,1,1],zp10_walk1,marker='o',label='1000 Walkers',color='red')
ax.plot(x,[1,1,1,1,1],zp10_walk5,marker='o',label='5000 Walkers',color='blue')
ax.plot(x,[1,1,1,1,1],zp10_walk10,marker='o',label='10000 Walkers',color='green')
ax.plot(x,[2,2,2,2,2],zp20_walk1,marker='o',linestyle='dashed',color='red')
ax.plot(x,[2,2,2,2,2],zp20_walk5,marker='o',linestyle='dashed',color='blue')
ax.plot(x,[2,2,2,2,2],zp20_walk10,marker='o',linestyle='dashed',color='green')
ax.plot(x,[2,2,2,2,2],zp20_walk10,marker='o',linestyle='dashed',color='green')

ax.set_xlabel('Time step')
ax.set_ylabel('Number of Production Phases')
ax.set_zlabel('Zero-Point Energy')
plt.title('Effects of Constants on Zero-Point Energy')
ax.legend()




fig = plt.figure(3)
ax = fig.add_subplot(111,projection='3d')

ax.axes.set_xticks(x)
ax.axes.set_xticklabels(x1)


ax.plot(x,pop_percent,zp10_walk10,marker='o',color='red')

ax.set_xlabel('Time step')
ax.set_zlabel('Zero-Point Energy')
ax.set_ylabel('Percent of Initial Walker Population')

plt.title('Correlation Between Walker Population and Zero-Point Energy')




fig = plt.figure(4)
ax = fig.add_subplot(111,projection='3d')


ax.axes.set_xticks(x)
ax.axes.set_xticklabels(x1)


x = [1,2,3,4,5]
x1 = [.1,.5,1,5,10]

std10_walk1=[0.00002573,0.00003331,0.00004462,0.00004517,0.00004853]
std10_walk5=[0.00000838,0.00001599,0.00002555,0.00001871,0.00003091]
std10_walk10=[0.00001148,0.00000875,0.00001498,0.00001858,.00002782]


zp10_walk1=[0.061283425,0.061897123,0.062116094,0.062130088,0.062191604]
zp10_walk5=[0.061240847,0.061859571,0.062070945,0.062070326,0.06207806]
zp10_walk10=[0.061240559,0.061855164,0.062060804,0.062066651,0.062068792]


ax.plot(x,std10_walk1,zp10_walk1,marker='o',label='1000 Walkers',color='red')
ax.plot(x,std10_walk5,zp10_walk5,marker='o',label='5000 Walkers',color='blue')
ax.plot(x,std10_walk10,zp10_walk10,marker='o',label='10000 Walkers',color='green')

ax.set_xlabel('Time step')
ax.set_ylabel('Zero-Point Energy Standard Deviation')
ax.set_zlabel('Zero-Point Energy')
plt.title('Effects of Time Step on Zero-Point Energy and Variability')
ax.legend()


plt.show()


