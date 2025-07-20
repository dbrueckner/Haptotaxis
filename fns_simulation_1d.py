#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:41:56 2020

@author: D.Brueckner
"""
import numpy as np
import fns_data_wrapper as fns_data_wrapper

def simulate(params,modes,N_part,N_dim,delta_t,N_t,oversampling):
    
    (tau0,sigma_p,chi,gamma,stiffness,n,length_cell,rho0,ell,length,width) = params
    (mode_hapto) = modes
    
    if mode_hapto:
        mode_init_dist = 'triangular'
        def rho(x):
            rho = rho0*np.exp(-np.abs(x)/ell)
            return rho
    else:
        mode_init_dist = 'uniform'
        def rho(x):
            if np.abs(x)<length/2:
                rho = rho0
            else:
                rho = 0
            return rho
    
    dt = delta_t/oversampling
    sqrtdt = np.sqrt(dt)
    N_t_oversampling = int((N_t-1)*oversampling)
    
    x_boundary = length/2
    
    X = np.zeros((N_part,N_t-1,N_dim))
    P = np.zeros((N_part,N_t-1,N_dim))
    grad = np.zeros((N_part,N_t-1,N_dim))
    
    for j in range(0,N_part):
        
        x_prev = np.zeros(N_dim)
        
        if mode_init_dist == 'uniform':
            random_number = 0.98*x_boundary*(-1+2*np.random.rand())
        elif mode_init_dist == 'triangular':
            random_number = np.random.triangular(-0.98*x_boundary, 0, 0.98*x_boundary)

        x_prev[0] = random_number
        p_prev = 0
            
        count_t = 0
        for t in range(0,N_t_oversampling):

            F_boundaries = F_obstacle(x_prev,0,length/2,stiffness,n,N_dim) + F_obstacle(x_prev,0,-length/2,stiffness,n,N_dim) 

            if mode_hapto:
                rho_grad = ( ( rho(x_prev[0]+length_cell/2) - rho(x_prev[0]-length_cell/2) )/length_cell )
            else:
                rho_grad = 0
            
            x_next = x_prev + dt*( p_prev + chi*rho_grad + F_boundaries )
            
            p_next = p_prev + dt*( -(1/tau0)*p_prev + gamma*F_boundaries ) + sigma_p*np.random.randn(N_dim)*sqrtdt

            x_prev = x_next
            p_prev = p_next
            
            if(np.mod(t,oversampling)==0):
                X[j,count_t,:] = x_next
                P[j,count_t,:] = p_next
                grad[j,count_t,:] = rho_grad
                count_t += 1
    
    data = fns_data_wrapper.StochasticTrajectoryData(X,length,width,delta_t,P=P,grad=grad)
    
    return data


def confinement_force(x,n): return -x**(n-1)
def F_obstacle(x,dim,position_obstacle_in,stiffness,n,N_dim,space=10):
    if position_obstacle_in<0:
        position_obstacle = position_obstacle_in + space
        condition = x[dim] >= position_obstacle
    else:
        position_obstacle = position_obstacle_in - space
        condition = x[dim] <= position_obstacle
        
    
    if(condition):
        result = np.zeros(N_dim)
    else:
        vec = np.zeros(N_dim)
        vec[dim] = 1
        result = stiffness*confinement_force(x[dim]-position_obstacle,n)*vec
    
    return result  