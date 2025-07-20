#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: D.Brueckner
"""

import numpy as np
import fns_data_wrapper as fns_data_wrapper
from fns_simulation_2d import simulate


#cell parameters
v_0 = 60 
D_r = 1 #rotational diffusion
chi = 70 #strength of gradient coupling
gamma = 50 #polarity repulsion
C0 = 3 #steric alignment to boundaries
lambda_C = 40   #length scale of steric boundary alignment
sigma_p = 40 #noise amplitude
length_cell = 60 #length of the cell
k_cell = 0.5 #cell length stiffness
sigma_L = 20 #noise on cell length

#micropattern parameters
rho0 = 85 #amplitude of gradient
ell = 84 #length scale of gradient decay (microns)
length = 500 #length of box (microns)
width = 200 #width of box (microns)
n = 8 #exponent of confinement potential
stiffness = 0.1 #stiffness of confinement potential

N_part = 100 #number of cells
N_t = 50 #number of time-steps
N_dim = 2
delta_t = 0.16 #time interval in hours
oversampling = 16 #number of intermediate time-steps

mode_hapto = 1

params = (v_0,D_r,chi,gamma,stiffness,n,length_cell,k_cell,sigma_L,rho0,ell,length,width,C0,lambda_C)
modes = (mode_hapto)
    
X,P,L,grad,theta = simulate(params,modes,N_part,N_dim,delta_t,N_t,oversampling)
    


import matplotlib.pyplot as plt
plt.close('all')
fig_size = [4,3]     
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)

plt.figure()

for j in range(0,N_part):        
    plt.plot(X[j,:,0],X[j,:,1])
        #data_snip.X[j,:data_snip.N_t[j]],data_snip.time[:data_snip.N_t[j]],color=colors_AH[data_snip.hapto[j]],lw=1,zorder=zorders[data_snip.hapto[j]])
  
plt.xlabel(r'x ($\mu$m)')
plt.ylabel(r'y ($\mu$m)')

plt.xlim([-length/2,length/2])
plt.xticks([-250,-125,0,125,250])
plt.ylim([-width/2,width/2])
plt.yticks([-width/2,width/2])

plt.tight_layout()