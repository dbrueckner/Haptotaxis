#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: D.Brueckner
"""
import numpy as np
import fns_data_wrapper as fns_data_wrapper

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def simulate(params,modes,N_part,N_dim,delta_t,N_t,oversampling):
    
    (v_0,D_r,chi,gamma,stiffness,n,length_cell,k_cell,sigma_L,rho0,ell,length,width,C0,lambda_C) = params
    (mode_hapto) = modes
    
    C = C0*np.exp(-width/lambda_C)
    
    if mode_hapto == True:
        def rho(x,x_boundary,rho0,ell):
            if np.abs(x)<x_boundary:
                rho = rho0*np.exp(-np.abs(x)/ell)
            else:
                rho = 0
            return rho
    else:
        def rho(x,x_boundary,rho0,ell):
            if np.abs(x)<x_boundary:
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
    L = np.zeros((N_part,N_t-1,N_dim))
    grad = np.zeros((N_part,N_t-1,N_dim))
    theta = np.zeros((N_part,N_t-1))
    for j in range(0,N_part):
        
        x_prev = np.zeros(N_dim)
        x_prev[0] = (length/6)*np.random.randn() #initialize x positions in the center
        x_prev[1] = 0.8*(width/2)*(-1+2*np.random.rand()) #initialize y positions away from boundaries
        
        theta_prev = np.pi*(-1+2*np.random.rand()) #random initial orientation
        L_prev = length_cell
        
        count_t = 0
        for t in range(0,N_t_oversampling):
            
            rho_grad = ( ( rho(x_prev[0]+L_prev/2,x_boundary,rho0,ell) - rho(x_prev[0]-L_prev/2,x_boundary,rho0,ell) )/L_prev )

            F_boundaries = F_obstacle(x_prev,0,length/2,stiffness,n,N_dim) + F_obstacle(x_prev,0,-length/2,stiffness,n,N_dim) + F_obstacle(x_prev,1,-width/2,stiffness,n,N_dim) + F_obstacle(x_prev,1,width/2,stiffness,n,N_dim)
            rho_grad_vec = rho_grad*np.array([1,0])
            pol = v_0*np.array([np.cos(theta_prev), np.sin(theta_prev)])
            
            x_next = x_prev + dt*( pol + chi*rho_grad_vec + F_boundaries )
            L_next = np.abs(L_prev + dt*( -k_cell*(L_prev-length_cell) ) + sigma_L*np.random.randn()*sqrtdt)
            theta_next = theta_prev + ( - C*np.sin(2*theta_prev) + gamma*F_boundaries[0] )*dt + np.sqrt(D_r)*np.random.randn()*sqrtdt

            x_prev = x_next
            L_prev = L_next
            theta_prev = theta_next
            
            if(np.mod(t,oversampling)==0):
                X[j,count_t,:] = x_next
                P[j,count_t,:] = pol
                L[j,count_t,:] = L_next
                grad[j,count_t,:] = rho_grad
                theta[j,count_t] = theta_next
                count_t += 1

    return X,P,L,grad,theta

def confinement_force(x,n): return -x**(n-1)
def F_obstacle(x,dim,position_obstacle_in,stiffness,n,N_dim,space=5):
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