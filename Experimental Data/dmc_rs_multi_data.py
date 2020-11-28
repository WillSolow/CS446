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


std10_walk1=[0.000223386,0.000240843,0.000252407,9.62612E-05]
std10_walk5=[0.000231928,9.61107E-05,0.00011202,7.36145E-05]
std10_walk10=[0.000157193,0.000176339,0.000152665,5.93053E-05]

std20_walk1=[0.000290794,0.000191455,0.000188728,0.000137379]
std20_walk5=[0.000122093,0.000208457,0.000102319,5.46887E-05]
std20_walk10=[0.0001011,9.59125E-05,0.000124843,3.75518E-05]

zp10_walk1=[0.185567197,0.185702524,0.185639948,0.187747629]
zp10_walk5=[0.185641904,0.185535115,0.185331114,0.187176671]
zp10_walk10=[0.186468431,0.185659258,0.185404298,0.187093136]


zp20_walk1=[0.185641904,0.185746141,0.185454332,0.18777013]
zp20_walk5=[0.185742507,0.185483241,0.185313117,0.187194938]
zp20_walk10=[0.1857000,0.185658667,0.185341121,0.187117592]

pop_percent=[0.9989,0.9973946,0.9896777,0.759241333]



x1 = [.1,.5,1,5]
x = [1,2,3,4]
y1 = [10,20]
y = [1,2]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

ax.axes.set_xticklabels(x1)
ax.axes.set_xticks(x)
ax.axes.set_yticklabels(y1)
ax.axes.set_yticks(y)
#ax.axes.ticklabel_format('both','plain')
ax.plot(x,[1,1,1,1],std10_walk1,marker='o',label='1000 Walkers',color='red')
ax.plot(x,[1,1,1,1],std10_walk5,marker='o',label='5000 Walkers',color='blue')
ax.plot(x,[1,1,1,1],std10_walk10,marker='o',label='10000 Walkers',color='green')
ax.plot(x,[2,2,2,2],std20_walk1,marker='o',linestyle='dashed',color='red')
ax.plot(x,[2,2,2,2],std20_walk5,marker='o',linestyle='dashed',color='blue')
ax.plot(x,[2,2,2,2],std20_walk10,marker='o',linestyle='dashed',color='green')
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

ax.plot(x,[1,1,1,1],zp10_walk1,marker='o',label='1000 Walkers',color='red')
ax.plot(x,[1,1,1,1],zp10_walk5,marker='o',label='5000 Walkers',color='blue')
ax.plot(x,[1,1,1,1],zp10_walk10,marker='o',label='10000 Walkers',color='green')
ax.plot(x,[2,2,2,2],zp20_walk1,marker='o',linestyle='dashed',color='red')
ax.plot(x,[2,2,2,2],zp20_walk5,marker='o',linestyle='dashed',color='blue')
ax.plot(x,[2,2,2,2],zp20_walk10,marker='o',linestyle='dashed',color='green')

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


