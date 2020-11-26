# Will Solow, Skye Rhomberg
# CS446 Fall 2020
# Diffusion Monte Carlo (DMC) Simulation
# Script Style
# Last Updated 11/4/20

# This code generates graphs of the water trimer system for the data
# we collected on convergence

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


std10_walk1=[]
std10_walk5=[]
std10_walk10=[]

std20_walk1=[]
std20_walk5=[]
std20_walk10=[]

zp10_walk1=[]
zp10_walk5=[]
zp10_walk10=[]

zp20_walk1=[]
zp20_walk5=[]
zp20_walk10=[]

pop_percent=[.9999, .9991, .9965, .9208, .7164]



x1 = [.05,.1,.5,1,5]
x = [1,2,3,4,5]
y1 = [10,20]
y = [1,2]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.axes.set_xticklabels(x1)
ax.axes.set_xticks(x)
ax.axes.set_yticklabels(y1)
ax.axes.set_yticks(y)
#ax.axes.ticklabel_format('both','plain')
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

ax.axes.set_xticklabels(x1)
ax.axes.set_xticks(x)
ax.axes.set_yticklabels(y1)
ax.axes.set_yticks(y)

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

ax.axes.set_xticklabels(x1)
ax.axes.set_xticks(x)

ax.plot(x,std10_walk1,zp10_walk1,marker='o',label='1000 Walkers',color='red')
ax.plot(x,std10_walk5,zp10_walk5,marker='o',label='5000 Walkers',color='blue')
ax.plot(x,std10_walk10,zp10_walk10,marker='o',label='10000 Walkers',color='green')

ax.set_xlabel('Time step')
ax.set_ylabel('Zero-Point Energy Standard Deviation')
ax.set_zlabel('Zero-Point Energy')
plt.title('Effects of Time Step on Zero-Point Energy and Variability')
ax.legend()

fig = plt.figure(4)
ax = fig.add_subplot(111,projection='3d')

ax.axes.set_xticklabels(x1)
ax.axes.set_xticks(x)
ax.plot(x,pop_percent,zp10_walk10,marker='o',color='red')

ax.set_xlabel('Time step')
ax.set_zlabel('Zero-Point Energy')
ax.set_ylabel('Percent of Initial Walker Population')

plt.title('Correlation Between Walker Population and Zero-Point Energy')

plt.show()


