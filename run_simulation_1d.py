#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:15:08 2020

@author: D.Brueckner
"""

import numpy as np
import fns_data_wrapper as fns_data_wrapper
from fns_simulation_1d import simulate


#cell parameters
tau0 = 2 #persistence time (in hours)
chi = 70 #strength of gradient coupling
gamma = 10 #polarity repulsion at boundaries
sigma_p = 40 #noise amplitude
length_cell = 60 #length of the cell

#micropattern parameters
rho0 = 85 #amplitude of gradient
ell = 84 #length scale of gradient decay (microns)
length = 500 #length of box (microns)
width = 20 #width of box (microns)
n = 8 #exponent of confinement potential
stiffness = 0.1 #stiffness of confinement potential

N_part = 100 #number of cells
N_t = 125 #number of time-steps
N_dim = 1
delta_t = 0.16 #time interval
oversampling = 16 #number of intermediate time-steps


data_all = []
data_snip_all = []

for mode_hapto in [0,1]:

    params = (tau0,sigma_p,chi,gamma,stiffness,n,length_cell,rho0,ell,length,width)
    modes = (mode_hapto)
        
    data = simulate(params,modes,N_part,N_dim,delta_t,N_t,oversampling)
    data_all.append(data)

    data_snippets = fns_data_wrapper.StochasticSnippetData(data)
    data_snip_all.append(data_snippets)


import matplotlib.pyplot as plt
plt.close('all')
fig_size = [10,3]     
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
H,W = 1,3

colors_AH = ['y','lightseagreen']
labels = ['Homogeneous','Gradients']

plt.figure()
chrt=0

ratios = np.zeros(2)
for mode_hapto in [0,1]:
    data_snip = data_snip_all[mode_hapto]
    
    chrt+=1
    plt.subplot(H,W,chrt)
    plt.title(labels[mode_hapto])
    
    zorders = [1,-1]
    plt.plot([0,0],[0,15],'--k',lw=0.5,zorder=2)
    for j in range(0,N_part):        
        plt.plot(data_snip.X[j,:data_snip.N_t[j]],data_snip.time[:data_snip.N_t[j]],color=colors_AH[data_snip.hapto[j]],lw=1,zorder=zorders[data_snip.hapto[j]])
      
    plt.xlabel(r'position ($\mu$m)')
    plt.ylabel('time (h)')
    
    plt.xlim([-data.length/2,data.length/2])
    plt.ylim([0,15])
    plt.xticks([-250,-125,0,125,250])
    plt.yticks([0,5,10,15])

    ratios[mode_hapto] = np.sum(data_snip.hapto)/data_snip.N_part*100
    

chrt+=1
plt.subplot(H,W,chrt)
width = 0.6
for mode_hapto in [0,1]:
    p=plt.bar(mode_hapto, ratios[mode_hapto], width, color=colors_AH[1])
    p=plt.bar(mode_hapto, 100-ratios[mode_hapto], width, bottom = ratios[mode_hapto], color=colors_AH[0])

plt.xticks([0,1],labels=labels)
plt.ylabel('Percentage of cells (%)')
plt.ylim([0,100])
plt.tight_layout()

