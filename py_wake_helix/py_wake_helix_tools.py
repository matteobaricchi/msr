# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:04:51 2025

@author: matteobaricchi
"""

import numpy as np
from scipy.stats import norm
from numpy import newaxis as na
import pandas as pd

from pathos.multiprocessing import ProcessPool



#%% FUNCTIONS




def calculateAEP_withUncertainty(wfm,
                                 x,
                                 y,
                                 wd_array,
                                 ws_array,
                                 yaw,
                                 helix_amp,
                                 sigma = 5,
                                 delta_wd_min = None,
                                 delta_wd_max = None,
                                 n = 5):
    
    # check if the values are allowed
    eps = 1e-9
    if sigma==0: sigma += eps
    
    # assign values to delta_wd_min and delta_wd_max if they are not given as input
    if delta_wd_min==None or delta_wd_max==None:
        delta_wd_min = -2*sigma
        delta_wd_max = 2*sigma

    # create array with misalignment values
    delta_wd_values = np.linspace(delta_wd_min,delta_wd_max,n,endpoint=True)
    
    # intialize power matrix with misalignment, size:(wind directions, wind speeds , misalignment values)
    power_delta_wd_mat = np.zeros((len(wd_array),len(ws_array),len(delta_wd_values)))
    
    # calculate power for each misalignment
    for nn in np.arange(n):
        delta_wd = delta_wd_values[nn]
        power_delta_wd_mat[:,:,nn] = np.sum(wfm(x,y,wd=(wd_array+delta_wd)%360,ws=ws_array,yaw=yaw+delta_wd,tilt=np.zeros_like(yaw),helix_amp=helix_amp).Power.values,axis=(0))
        
    # calculate uncertainty array (normalized) and extend its dim
    p_uncertainty_temp = norm.pdf(delta_wd_values,0,sigma)
    p_uncertainty = p_uncertainty_temp/np.sum(p_uncertainty_temp)
    p_uncertainty_ext = np.tile(p_uncertainty[na,na,:],(len(wd_array),len(ws_array),1))
    
    # extract probability matrix from the wind rose and extedn its dim
    sim_res = wfm(x,y,wd=wd_array,ws=ws_array,yaw=np.zeros_like(yaw),tilt=np.zeros_like(yaw),helix_amp=np.zeros_like(yaw))
    p_windrose = sim_res.P.values
    p_windrose_ext = np.tile(p_windrose[:,:,na],(1,1,len(delta_wd_values)))
    
    # combine the probability matrices (uncertainty and windrose) and calculate AEP in GWh
    aep = 8760*np.sum(power_delta_wd_mat*p_uncertainty_ext*p_windrose_ext)/1e9
    
    return aep



def calculatePmat_withUncertainty(wfm,
                                 x,
                                 y,
                                 wd_array,
                                 ws_array,
                                 yaw,
                                 helix_amp,
                                 sigma = 5,
                                 delta_wd_min = None,
                                 delta_wd_max = None,
                                 n = 5):
    
    # check if the values are allowed
    eps = 1e-9
    if sigma==0: sigma += eps
    
    # assign values to delta_wd_min and delta_wd_max if they are not given as input
    if delta_wd_min==None or delta_wd_max==None:
        delta_wd_min = -2*sigma
        delta_wd_max = 2*sigma

    # create array with misalignment values
    delta_wd_values = np.linspace(delta_wd_min,delta_wd_max,n,endpoint=True)
    
    # intialize power matrix with misalignment, size:(x, wind directions, wind speeds , misalignment values)
    power_delta_wd_mat = np.zeros((len(x),len(wd_array),len(ws_array),len(delta_wd_values)))
    
    # calculate power for each misalignment
    for nn in np.arange(n):
        delta_wd = delta_wd_values[nn]

        if delta_wd>=0:
            ind_start_pre = len(wd_array)-len(wd_array[(wd_array+delta_wd)>=360])
            ind_start_post = len(wd_array[(wd_array+delta_wd)>=360])
        else:
            ind_start_pre = len(wd_array[(wd_array+delta_wd)<0])
            ind_start_post = len(wd_array)-len(wd_array[(wd_array+delta_wd)<0])
        ind_array_pre = np.concatenate((np.arange(ind_start_pre,len(wd_array),1),np.arange(0,ind_start_pre,1)))
        ind_array_post = np.concatenate((np.arange(ind_start_post,len(wd_array),1),np.arange(0,ind_start_post,1)))

        yaw_ordered = np.take_along_axis(yaw,ind_array_pre[na,:,na],axis=1)
        helix_amp_ordered = np.take_along_axis(helix_amp,ind_array_pre[na,:,na],axis=1)

        power_delta_wd_mat_temp = wfm(x,y,wd=(wd_array+delta_wd)%360,ws=ws_array,yaw=yaw_ordered+delta_wd,tilt=np.zeros_like(yaw),helix_amp=helix_amp_ordered).Power.values
        power_delta_wd_mat[:,:,:,nn] = np.take_along_axis(power_delta_wd_mat_temp,ind_array_post[na,:,na],axis=1)
                
    # calculate uncertainty array (normalized) and extend its dim
    p_uncertainty_temp = norm.pdf(delta_wd_values,0,sigma)
    p_uncertainty = p_uncertainty_temp/np.sum(p_uncertainty_temp)
    p_uncertainty_ext = np.tile(p_uncertainty[na,na,na,:],(len(x),len(wd_array),len(ws_array),1))
    
    # calculate power matrix dim:(x,wd,ws)
    p_mat = np.sum(power_delta_wd_mat*p_uncertainty_ext,axis=(3))
    
    return p_mat






# same principle of calculateAEP_withUncertainty but wd and ws are interpreted as timseries,
# so it returns array of power 

def calculatePower_withUncertainty(wfm,                     # wind farm model object
                                   x,                       # array of x coords, dim=(N)
                                   y,                       # array of y coords, dim=(N)
                                   wd_t,                    # wd timeseries, dim=(T)
                                   ws_t,                    # ws timeseries, dim=(T)
                                   yaw_t,                   # yaw timeseries, dim=(N,T)
                                   helix_amp_t,             # helix_amp timeseries, dim=(N,T)
                                   sigma = 5,               # std in deg
                                   delta_wd_min = None,     # min wd value considered in the Gaussian distribution centered in wd
                                   delta_wd_max = None,     # max wd value considered in the Gaussian distribution centered in wd
                                   n = 5):                  # number wd samples considered when applying uncertainty 
    
    # check if the values are allowed
    eps = 1e-9
    if sigma==0: sigma += eps
    
    # assign values to delta_wd_min and delta_wd_max if they are not given as input
    if delta_wd_min==None or delta_wd_max==None:
        delta_wd_min = -2*sigma
        delta_wd_max = 2*sigma

    # create array with misalignment values
    delta_wd_values = np.linspace(delta_wd_min,delta_wd_max,n,endpoint=True)
    
    # intialize power matrix with misalignment, size:(flow condition , misalignment values)
    power_delta_wd_mat = np.zeros((len(wd_t),len(delta_wd_values)))
    
    # calculate power for each misalignment
    for nn in np.arange(n):
        delta_wd = delta_wd_values[nn]
        power_delta_wd_mat[:,nn] = np.sum(wfm(x,y,wd=(wd_t+delta_wd)%360,ws=ws_t,yaw=yaw_t+delta_wd,tilt=np.zeros_like(yaw_t),helix_amp=helix_amp_t,time=True).Power.values,axis=(0))
        
    # calculate uncertainty array (normalized) and extend its dim
    p_uncertainty_temp = norm.pdf(delta_wd_values,0,sigma)
    p_uncertainty = p_uncertainty_temp/np.sum(p_uncertainty_temp)
    p_uncertainty_ext = np.tile(p_uncertainty[na,:],(len(wd_t),1))
        
    # combine the probability matrices (uncertainty) and calculate power
    p = np.sum(power_delta_wd_mat*p_uncertainty_ext,axis=1)
    
    return p






class WFFC_Optimizer_SR():
    
    def __init__(self,
                 x,
                 y,
                 wd,
                 f_obj,                     # f(yaw,helix_amp)
                 yaw_max = 30,
                 helix_amp_max = 5,
                 n_step = 3,
                 n_values = 5,
                 optimize_yaw = True,
                 optimize_helix_amp = True,
                 tol = 1e-5  # 0.001%
                 ):
        
        self.x = x
        self.y = y
        self.wd = wd
        self.f_obj = f_obj
        self.yaw_max = yaw_max
        self.helix_amp_max = helix_amp_max
        self.n_step = n_step
        self.n_values = n_values
        
        self.optimize_yaw = optimize_yaw
        self.optimize_helix_amp = optimize_helix_amp
        self.tol = tol
        
        self.yaw_opt =  np.zeros(len(x))
        self.helix_amp_opt =  np.zeros(len(x))
        

    def optimize(self):
        
        # order the turbines depending on the wind direction
        theta = np.pi*(270-self.wd)/180
        x_rot = self.x*np.cos(theta)+self.y*np.sin(theta)
        ind_turbine_ordered = np.argsort(x_rot)
        
        # initialize yaw and helix_amp
        n_wt = len(self.x)
        yaw_opt = np.zeros(n_wt)
        helix_amp_opt = np.zeros(n_wt)
        f_0 = self.f_obj(yaw_opt,helix_amp_opt)
        f_opt = f_0
        
        # initialize yaw and helix_amp values to test at the first step
        yaw_range = self.yaw_max*2/(self.n_values-1)
        yaw_values = np.linspace(-yaw_range*(self.n_values-1)/2,yaw_range*(self.n_values-1)/2,self.n_values)
        helix_amp_values = np.linspace(0,self.helix_amp_max,self.n_values)
        
        # initialize temporary optimal variables
        yaw_opt_y = yaw_opt.copy()
        yaw_opt_h = yaw_opt.copy()
        helix_amp_opt_y = helix_amp_opt.copy()
        helix_amp_opt_h = helix_amp_opt.copy()
        f_opt_y = f_opt.copy()
        f_opt_h = f_opt.copy()


        
        # iterate for each step
        for s in np.arange(self.n_step):
            
            # iterate for each turbine
            for tt in np.arange(n_wt):
                
                yaw_values_test = np.minimum(np.maximum(yaw_opt_y[ind_turbine_ordered[tt]]+yaw_values,-self.yaw_max),self.yaw_max)
                helix_amp_values_test = np.minimum(np.maximum(helix_amp_opt_h[ind_turbine_ordered[tt]]+helix_amp_values,0),self.helix_amp_max)
                
                # f_opt_y = f_opt.copy()
                # f_opt_h = f_opt.copy()
                
                
                if self.optimize_helix_amp:
                    # iterate for each helix_amp value
                    for hh in helix_amp_values_test:
                                
                        # test new yaw helix_amp
                        yaw_test = yaw_opt.copy()
                        helix_amp_test = helix_amp_opt.copy()
                        yaw_test[ind_turbine_ordered[tt]] = 0
                        helix_amp_test[ind_turbine_ordered[tt]] = hh
                        f_test = self.f_obj(yaw_test,helix_amp_test)
                        
                        # check improvement
                        if f_test>f_opt_h*(1+self.tol):
                            yaw_opt_h = yaw_test.copy()
                            helix_amp_opt_h = helix_amp_test.copy()
                            f_opt_h = f_test
                    

                if self.optimize_yaw:
                    # iterate for each yaw value
                    for yy in yaw_values_test:
                                
                        # test new yaw angle
                        yaw_test = yaw_opt.copy()
                        helix_amp_test = helix_amp_opt.copy()
                        yaw_test[ind_turbine_ordered[tt]] = yy
                        helix_amp_test[ind_turbine_ordered[tt]] = 0
                        f_test = self.f_obj(yaw_test,helix_amp_test)
                        
                        # check improvement
                        if f_test>f_opt_y*(1+self.tol):
                            yaw_opt_y = yaw_test.copy()
                            helix_amp_opt_y = helix_amp_test.copy()
                            f_opt_y = f_test
                                              
                # find best control strategy
                if f_opt_y>f_opt_h:
                    f_opt = f_opt_y
                    yaw_opt = yaw_opt_y.copy()
                    helix_amp_opt = helix_amp_opt_y.copy()
                elif f_opt_h>f_opt_y:
                    f_opt = f_opt_h
                    yaw_opt = yaw_opt_h.copy()
                    helix_amp_opt = helix_amp_opt_h.copy()

                #print(f'Step {s} - Turbine {tt} - Opt. yaw = {yaw_opt[ind_turbine_ordered[tt]]} - Opt. yaw = {helix_amp_opt[ind_turbine_ordered[tt]]} - Fyaw = {f_opt_y} - Fhelix = {f_opt_h} - F = {f_opt}')
            # print(f'Step {s} completed')
            
            # update yaw and helix_amp values for the next step
            yaw_range = ((np.max(yaw_values)/((self.n_values-1)/2))/2)/((self.n_values-1)/2)
            yaw_values = np.linspace(-yaw_range*(self.n_values-1)/2,yaw_range*(self.n_values-1)/2,self.n_values)
            helix_amp_range = ((np.max(helix_amp_values)/((self.n_values-1)/2))/2)/((self.n_values-1)/2)
            helix_amp_values = np.linspace(-helix_amp_range*(self.n_values-1)/2,helix_amp_range*(self.n_values-1)/2,self.n_values)

        # restore initial values if there is no improvement
        if f_0 >= f_opt*(1-self.tol):
            f_opt = f_0
            yaw_opt = np.zeros(n_wt)
            helix_amp_opt = np.zeros(n_wt)

        self.yaw_opt = yaw_opt
        self.helix_amp_opt = helix_amp_opt



class Power_wrapper():
    
    def __init__(self,**kwargs):
        self.x = kwargs.get('x')
        self.y = kwargs.get('y')
        self.wfm = kwargs.get('wfm')
        self.wd = kwargs.get('wd')
        self.ws = kwargs.get('ws')
        self.apply_uncertainty = kwargs.get('apply_uncertainty',False)
        self.sigma = kwargs.get('sigma',5)
        self.n = kwargs.get('n',5)
        
        
    def __call__(self,yaw,helix_amp):
        
        if self.apply_uncertainty:
            p = calculatePower_withUncertainty(wfm = self.wfm,
                                               x = self.x,
                                               y = self.y,
                                               wd_t = np.array([self.wd]),
                                               ws_t = np.array([self.ws]),
                                               yaw_t = yaw,
                                               helix_amp_t = helix_amp,
                                               sigma = self.sigma,
                                               n = self.n)
        else:
            p = np.sum(self.wfm(self.x,
                               self.y,
                               wd = np.array([self.wd]),
                               ws = np.array([self.ws]),
                               yaw = yaw,
                               tilt = 0,
                               helix_amp = helix_amp).Power.values)
        
        return p





def compute_WFFC_LUT(x,
                     y,
                     wfm,
                     wd,
                     ws,
                     timeseries = False,
                     yaw_max = 30,
                     helix_amp_max = 5,
                     n_step = 3,
                     n_values = 5,
                     optimize_yaw = True,
                     optimize_helix_amp = True,
                     tol = 1e-5,
                     apply_uncertainty = False,
                     sigma = 5,
                     n = 5,
                     parallel_execution = False,
                     n_cpu = None):
    
    # INDEX DIM:
    # i: turbine index
    # l: wind direction index
    # k: wind speed index
    # t: time index
    
    # define a wrapper function to run in parallel for each flow case
    def wffc_optimizer_wrapper(wd,ws):
        power_wrapper = Power_wrapper(x = x,
                                      y = y,
                                      wfm = wfm,
                                      wd = wd,
                                      ws = ws,
                                      apply_uncertainty = apply_uncertainty,
                                      sigma = sigma,
                                      n = n)

        optimizer = WFFC_Optimizer_SR(x = x,
                                      y = y,
                                      wd = wd,
                                      f_obj = power_wrapper,
                                      yaw_max = yaw_max,
                                      helix_amp_max = helix_amp_max,
                                      n_step = n_step,
                                      n_values = n_values,
                                      optimize_yaw = optimize_yaw,
                                      optimize_helix_amp = optimize_helix_amp,
                                      tol=tol)
        optimizer.optimize()
        yaw_opt = optimizer.yaw_opt
        helix_amp_opt = optimizer.helix_amp_opt
        return yaw_opt,helix_amp_opt

    
    if timeseries:
        # use wd and ws as timseries
        wd_list = wd.tolist()
        ws_list = ws.tolist()
    else:
        # use wd and ws as wind rose
        wd_list = np.tile(wd[:,na],(1,len(ws))).reshape(-1).tolist()
        ws_list = np.tile(ws[na,:],(len(wd),1)).reshape(-1).tolist()
    
    
    if parallel_execution:
        with ProcessPool(n_cpu) as pool:
            results = pool.map(wffc_optimizer_wrapper,wd_list,ws_list)
        res_yaw_opt,res_helix_amp_opt = zip(*results)
    else:
        results = map(wffc_optimizer_wrapper,wd_list,ws_list)
        res_yaw_opt,res_helix_amp_opt = zip(*results)


    # reconstruct array
    if timeseries:
        yaw_opt_array = np.array(list(res_yaw_opt)).T
        helix_amp_opt_array = np.array(list(res_helix_amp_opt)).T
    else:
        yaw_opt_array = np.reshape(np.array(list(res_yaw_opt)).T,(len(x),len(wd),len(ws)))
        helix_amp_opt_array = np.reshape(np.array(list(res_helix_amp_opt)).T,(len(x),len(wd),len(ws)))

    return yaw_opt_array,helix_amp_opt_array





def create_LUTdf(wd,
                 ws,
                 yaw_opt,
                 helix_amp_opt,
                 timeseries=False):

    n_wt = yaw_opt.shape[0]
    col_names = ['Wind direction','Wind speed']+[f'Turbine {i}' for i in np.arange(n_wt)]
    df_yaw = pd.DataFrame(columns=col_names)
    df_helix_amp = pd.DataFrame(columns=col_names)
    
    if timeseries:
        wd_t = wd
        ws_t = ws
        yaw_opt_it = yaw_opt
        helix_amp_opt_it = helix_amp_opt
        
    else:
        wd_t = np.tile(wd[:,na],(1,len(ws))).reshape(-1)
        ws_t = np.tile(ws[na,:],(len(wd),1)).reshape(-1)
        yaw_opt_it = np.reshape(yaw_opt,(n_wt,len(wd)*len(ws)))
        helix_amp_opt_it = np.reshape(helix_amp_opt,(n_wt,len(wd)*len(ws)))
        
    df_yaw['Wind speed'] = ws_t
    df_yaw['Wind direction'] = wd_t
    df_helix_amp['Wind speed'] = ws_t
    df_helix_amp['Wind direction'] = wd_t
    
    for i in np.arange(n_wt):
        df_yaw[f'Turbine {i}'] = yaw_opt_it[i,:]
        df_helix_amp[f'Turbine {i}'] = helix_amp_opt_it[i,:]
        
    return df_yaw,df_helix_amp





def extract_LUTdf(df_yaw,
                  df_helix_amp,
                  timeseries=False):

    cols = df_yaw.columns
    n_wt = len(cols)-2
    
    if timeseries:
        
        wd_t = np.array(df_yaw['Wind direction'])
        ws_t = np.array(df_yaw['Wind speed'])
    
        yaw_opt = np.zeros((n_wt,len(wd_t)))
        helix_amp_opt = np.zeros((n_wt,len(wd_t)))
        
        for i in np.arange(n_wt):
            yaw_opt[i,:] = np.array(df_yaw[f'Turbine {i}'])
            helix_amp_opt[i,:] = np.array(df_helix_amp[f'Turbine {i}'])
            
        return yaw_opt,helix_amp_opt,wd_t,ws_t
    
    else:
        
        wd_array = np.unique(np.array(df_yaw['Wind direction']))
        ws_array = np.unique(np.array(df_helix_amp['Wind speed']))
        
        yaw_opt = np.zeros((n_wt,len(wd_array),len(ws_array)))
        helix_amp_opt = np.zeros((n_wt,len(wd_array),len(ws_array)))
        
        for i in np.arange(n_wt):
            for i_wd in np.arange(len(wd_array)):
                fil_wd = df_yaw['Wind direction']==wd_array[i_wd]
                for i_ws in np.arange(len(ws_array)):
                    fil_ws = df_yaw['Wind speed']==ws_array[i_ws]
                    df_yaw_temp = df_yaw[fil_wd&fil_ws]
                    df_helix_amp_temp = df_helix_amp[fil_wd&fil_ws]
                    yaw_opt[i,i_wd,i_ws] = df_yaw_temp[f'Turbine {i}'].iloc[0]
                    helix_amp_opt[i,i_wd,i_ws] = df_helix_amp_temp[f'Turbine {i}'].iloc[0]

        return yaw_opt,helix_amp_opt,wd_array,ws_array




class Obj_flow_wrapper():

    def __init__(self,**kwargs):
        self.wd = kwargs.get('wd')
        self.ws = kwargs.get('ws')
        self.obj_flow_function = kwargs.get('obj_flow_function')

    def __call__(self,yaw,helix_amp):
        f = self.obj_flow_function(wd = self.wd,
                                   ws = self.ws,
                                   yaw = yaw,
                                   helix_amp = helix_amp)
        return f



def compute_WFFC_LUT_generalObj(x,
                                y,
                                wd,
                                ws,
                                obj_flow_function,
                                timeseries = False,
                                yaw_max = 30,
                                helix_amp_max = 5,
                                n_step = 3,
                                n_values = 5,
                                optimize_yaw = True,
                                optimize_helix_amp = True,
                                tol = 1e-5,
                                parallel_execution = False,
                                n_cpu = None):
    
    # INDEX DIM:
    # i: turbine index
    # l: wind direction index
    # k: wind speed index
    # t: time index
    
    # define a wrapper function to run in parallel for each flow case
    def wffc_optimizer_wrapper(wd,ws):
        obj_wrapper = Obj_flow_wrapper(wd = wd,
                                       ws = ws,
                                       obj_flow_function = obj_flow_function)

        optimizer = WFFC_Optimizer_SR(x = x,
                                      y = y,
                                      wd = wd,
                                      f_obj = obj_wrapper,
                                      yaw_max = yaw_max,
                                      helix_amp_max = helix_amp_max,
                                      n_step = n_step,
                                      n_values = n_values,
                                      optimize_yaw = optimize_yaw,
                                      optimize_helix_amp = optimize_helix_amp,
                                      tol=tol)
        optimizer.optimize()
        yaw_opt = optimizer.yaw_opt
        helix_amp_opt = optimizer.helix_amp_opt
        return yaw_opt,helix_amp_opt

    
    if timeseries:
        # use wd and ws as timseries
        wd_list = wd.tolist()
        ws_list = ws.tolist()
    else:
        # use wd and ws as wind rose
        wd_list = np.tile(wd[:,na],(1,len(ws))).reshape(-1).tolist()
        ws_list = np.tile(ws[na,:],(len(wd),1)).reshape(-1).tolist()
    
    
    if parallel_execution:
        with ProcessPool(n_cpu) as pool:
            results = pool.map(wffc_optimizer_wrapper,wd_list,ws_list)
        res_yaw_opt,res_helix_amp_opt = zip(*results)
    else:
        results = map(wffc_optimizer_wrapper,wd_list,ws_list)
        res_yaw_opt,res_helix_amp_opt = zip(*results)


    # reconstruct array
    if timeseries:
        yaw_opt_array = np.array(list(res_yaw_opt)).T
        helix_amp_opt_array = np.array(list(res_helix_amp_opt)).T
    else:
        yaw_opt_array = np.reshape(np.array(list(res_yaw_opt)).T,(len(x),len(wd),len(ws)))
        helix_amp_opt_array = np.reshape(np.array(list(res_helix_amp_opt)).T,(len(x),len(wd),len(ws)))

    return yaw_opt_array,helix_amp_opt_array









#%% TEST

# from functools import partial
# import time

# # import new models
# from py_wake_helix.py_wake_helix import helix_power_ct_function
# from py_wake_helix.py_wake_helix import PropagateDownwind_helix
# from py_wake_helix.py_wake_helix import EmpiricalGaussianDeficit
# from py_wake_helix.py_wake_helix import EmpiricalGaussianDeflection

# # import py_pywake models
# from py_wake.wind_turbines import WindTurbine
# from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
# from py_wake.site import UniformWeibullSite
# from py_wake.rotor_avg_models import GaussianOverlapAvgModel
# from py_wake.superposition_models import SquaredSum


# if __name__ == "__main__":


#     # deifne site (HKN) - NOT RELEVANT FOR THE TUNING
#     wd_site = np.linspace(0,360,12,endpoint=False)
#     p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
#     a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
#     k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
#     site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)
    
#     # define turbine
#     powerCtFunction = PowerCtFunction(
#         input_keys=['ws','helix_amp'],
#         power_ct_func = partial(helix_power_ct_function,
#                                 helix_a = 1.907,
#                                 helix_power_b = 1.376e-3,
#                                 helix_power_c = 4.017e-11,  # not tuned
#                                 helix_thrust_b = 0.8371e-3,
#                                 helix_thrust_c = 5.084e-4),  # not tuned
#         power_unit='kW',
#     )
#     wind_turbine = WindTurbine(name='IEA22MW_helix',
#                     diameter=283.2,
#                     hub_height=170.0,
#                     powerCtFunction=powerCtFunction)    
#     diameter = wind_turbine.diameter()
    
#     # define wind farm model (EMPGAUSS - OPT COEFF.)
#     wfm = PropagateDownwind_helix(site, wind_turbine,
#                                                wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[0.01213,0.008],
#                                                                                           sigma_0_D=0.3042,
#                                                                                           mixing_gain_velocity=0.2166,
#                                                                                           awc_wake_exp=1.130,
#                                                                                           awc_wake_denominator=145.5),
#                                                superpositionModel=SquaredSum(),
#                                                deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=2.1,
#                                                                               deflection_rate=12.017,
#                                                                               mixing_gain_deflection=0.),
#                                                turbulenceModel=None,
#                                                rotorAvgModel=GaussianOverlapAvgModel())
    
    
    
#     # define case study
    
#     x = np.array([0,4.5,9])*diameter
#     y = np.array([0,0,0])*diameter
    
#     # wd_t = np.array([265,270])
#     # ws_t = np.array([8,8])
    
#     # wd_array = np.arange(0,360,5)
#     # ws_array = np.arange(4,26,1)
    
    
    
    
#     wd_array = np.array([265,270,275,280])
#     ws_array = np.array([8,9])
    
#     wd_t = np.array([265,270,275,200])
#     ws_t = np.array([8,8,8,8])
    
#     wd = wd_array
#     ws = ws_array
#     timeseries = False
    
#     t = time.time()
#     yaw_opt,helix_amp_opt =  compute_WFFC_LUT(x,
#                                               y,
#                                               wfm,
#                                               wd,
#                                               ws,
#                                               timeseries = timeseries,
#                                               yaw_max = 30,
#                                               helix_amp_max = 5,
#                                               n_step = 3,
#                                               n_values = 5,
#                                               optimize_yaw = True,
#                                               optimize_helix_amp = True,
#                                               apply_uncertainty = True,
#                                               sigma = 5,
#                                               parallel_execution = False,
#                                               n_cpu = 12)
#     print(f'Time: {time.time()-t}')

#     df_yaw,df_helix_amp = create_LUTdf(wd, ws, yaw_opt, helix_amp_opt, timeseries)
#     yaw_opt,helix_amp_opt,ws,ws = extract_LUTdf(df_yaw,df_helix_amp)






