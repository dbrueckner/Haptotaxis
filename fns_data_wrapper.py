#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:52:27 2020

@author: D.Brueckner
"""

import numpy as np

class StochasticTrajectoryData(object):
 
    def __init__(self,X,length,width,delta_t,P=None,grad=None,theta=None,L=None):
        
        self.length = length
        self.width = width
        self.N_part = len([x for x in X[:,0,0] if not np.isnan(x)])
        self.N_t = []
        for j in range(0,self.N_part):
            self.N_t.append(len([x for x in X[j,:,0] if not np.isnan(x)]))
        self.N_part_max = X.shape[0]
        self.N_t_max = X.shape[1]
        self.N_variables = X.shape[2]
        
        self.dt = delta_t
        self.time = np.linspace(0,self.N_t_max-1,self.N_t_max)*self.dt
        self.t_max = self.N_t_max*self.dt

        self.X = X
        self.P = P
        self.grad = grad
        self.theta = theta
        self.L = L


class StochasticSnippetData(object):
 
    def __init__(self,data,mode_symmetrize=True,mode_mountain=True):
        
        self.length = data.length
        self.width = data.width
        self.N_part = data.N_part
        self.N_variables = data.N_variables
        self.dt = data.dt
        
        from scipy.signal import argrelextrema
        from scipy.ndimage.filters import gaussian_filter1d
        self.N_t = []
        for j in range(0,self.N_part):
            trajectory = data.X[j,:,0]
            trajectory_smooth = gaussian_filter1d(trajectory[:data.N_t[j]], sigma=4)

            maxima = argrelextrema(trajectory_smooth, np.greater)[0]
            minima = argrelextrema(trajectory_smooth, np.less)[0]
            if len(maxima)>0:
                loc_max = maxima[0]
            else:
                loc_max = data.N_t[j]
            if len(minima)>0:
                loc_min = minima[0]
            else:
                loc_min = data.N_t[j]
            self.N_t.append(min(loc_max,loc_min))
        
        self.N_t_max = np.max(self.N_t)
        
        self.time = np.linspace(0,self.N_t_max-1,self.N_t_max)*self.dt
        self.t_max = self.N_t_max*self.dt

        self.X = np.zeros((self.N_part,self.N_t_max))
        self.X[:] = np.nan
        
        self.P = np.zeros((self.N_part,self.N_t_max))
        self.P[:] = np.nan
        
        self.grad = np.zeros((self.N_part,self.N_t_max))
        self.grad[:] = np.nan
                
        for j in range(0,self.N_part):
            if mode_symmetrize:
                if data.X[j,0,0]>0:
                    fac = -1
                else:
                    fac = 1
            else:
                fac = 1
                
            self.X[j,:self.N_t[j]] = fac*data.X[j,:self.N_t[j],0]
            
            self.grad[j,:self.N_t[j]] = fac*data.grad[j,:self.N_t[j],0]
            self.P[j,:self.N_t[j]] = fac*data.P[j,:self.N_t[j],0]
                
        self.hapto = []
        for j in range(0,self.N_part):    
            hapto_dir = 0
            if mode_mountain:
                if self.X[j,self.N_t[j]-1]-self.X[j,0] > 0:
                    hapto_dir = 1
            else:
                if self.X[j,0]>=0:
                    if self.X[j,self.N_t[j]-1]-self.X[j,0] > 0:
                        hapto_dir = 1
                else:
                    if self.X[j,self.N_t[j]-1]-self.X[j,0] < 0:
                        hapto_dir = 1
            self.hapto.append(hapto_dir)
        
        self.V = np.diff(self.X,axis=1)/self.dt
        self.A = np.diff(self.V,axis=1)/self.dt


