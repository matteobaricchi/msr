# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:52:33 2025

@author: matteobaricchi
"""
#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import re
from functools import partial
import xarray as xr
from numpy import newaxis as na
import time


# import new models
from py_wake_helix.py_wake_helix import helix_power_ct_function
from py_wake_helix.py_wake_helix import PropagateDownwind_helix
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeficit
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeflection

# import py_pywake models
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site import UniformWeibullSite
from py_wake.rotor_avg_models import GaussianOverlapAvgModel,RotorCenter
from py_wake.superposition_models import LinearSum,SquaredSum
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.deficit_models import BastankhahGaussianDeficit
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.turbulence_models import STF2017TurbulenceModel
from py_wake.wind_farm_models import PropagateDownwind




#%% IMPORT DATA - WAKE

#os.chdir(r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\data_tuning\DATA_HelixEngineeringModel_tuning_20250220')
path_name = r'data_tuning\DATA_HelixEngineeringModel_tuning_20250220\\'

df_baseline_1wt = pd.read_csv(path_name+'mean_flow_1turb_U10ms_baseline.csv')                     # Daan - n_wt=1 d=- ws=10m/s TI=0.04

df_baseline_3wt_aligned = pd.read_csv(path_name+'mean_flow_3turbs_U10ms_baseline.csv')            # Daan - n_wt=3 d=4.5D ws=10m/s TI=0.04
df_baseline_3wt_misaligned = pd.read_csv(path_name+'mean_flow_3turbs_U10ms_baseline_WD206.csv')   # Daan - n_wt=3 d=4.5D ws=10m/s TI=0.04 wd=+5deg

df_helix2deg_1wt = pd.read_csv(path_name+'mean_flow_1turb_U10ms_helix_A2.0deg.csv')               # Daan - n_wt=1 d=- ws=10m/s TI=0.04 helix_amp=2deg
df_helix3deg_1wt = pd.read_csv(path_name+'mean_flow_1turb_U10ms_helix_A3.0deg.csv')               # Daan - n_wt=1 d=- ws=10m/s TI=0.04 helix_amp=3deg
df_helix4deg_1wt = pd.read_csv(path_name+'mean_flow_1turb_U10ms_helix_A4.0deg.csv')               # Daan - n_wt=1 d=- ws=10m/s TI=0.04 helix_amp=4deg

df_helix3deg_3wt = pd.read_csv(path_name+'mean_flow_3turbs_U10ms_helix_A3.0deg.csv')               # Daan - n_wt=3 d=4.5D ws=10m/s TI=0.04 helix_amp=[3,0,0]deg
df_helix4deg_3wt = pd.read_csv(path_name+'mean_flow_3turbs_U10ms_helix_A4.0deg.csv')               # Daan - n_wt=3 d=4.5D ws=10m/s TI=0.04 helix_amp=[4,0,0]deg

df_yaw20deg_1wt = pd.read_csv(path_name+'mean_flow_1turb_U10ms_yaw_A20.0deg.csv')                 # Daan - n_wt=1 d=- ws=10m/s TI=0.04 yaw=20deg
df_yawmin20deg_1wt = pd.read_csv(path_name+'mean_flow_1turb_U10ms_yaw_Aneg20.0deg.csv')           # Daan - n_wt=1 d=- ws=10m/s TI=0.04 yaw=-20deg

df_yawGeom_3wt = pd.read_csv(path_name+'mean_flow_3turbs_U10ms_yaw.csv')                         # Daan - n_wt=3 d=4.5D ws=10m/s TI=0.04 yaw=geometric_yaw wd=201deg HKN corner


#%% FUNCTIONS

def extract_data_df(df):
    def extract_downstream_distance(expression: str) -> float:
        return float(re.search(r'[-+]?[0-9]*\.?[0-9]+', expression).group())
    x_D_array = np.array([extract_downstream_distance(col_name) for col_name in df.columns.tolist()[1:]])
    y_D_array = np.array(df['y/D'].tolist())
    u_mat = df.iloc[:,1:].to_numpy()
    return u_mat,x_D_array,y_D_array


# this function should be used when the effect of RotorAvg models is neglected (e.g. 1 wt simulation)
def compute_flow(x_D_array,y_D_array,diameter,x,y,wfm,wd,ws,yaw,helix_amp):
    
    # initialize u_mat
    u_mat = np.zeros((len(y_D_array),len(x_D_array)))
    
    # iterate for each position of the flow field
    for y_ind in np.arange(len(y_D_array)):
        for x_ind in np.arange(len(x_D_array)):
            x_new = np.concatenate((x,np.array([x_D_array[x_ind]*diameter])))
            y_new = np.concatenate((y,np.array([y_D_array[y_ind]*diameter])))
            yaw_new = np.concatenate((yaw,np.array([0.])))
            helix_amp_new = np.concatenate((helix_amp,np.array([0.])))
            simres = wfm(x_new,y_new,wd=wd,ws=ws,yaw=yaw_new,tilt=0,helix_amp=helix_amp_new)
            u_mat[y_ind,x_ind] = simres.WS_eff_ilk[-1]
            
        # print(f'Iteration {y_ind+1} of {len(y_D_array)} completed')
            
    return u_mat


def compute_flow_Bastankhah(x_D_array,y_D_array,diameter,x,y,wfm,wd,ws,yaw):
    
    # initialize u_mat
    u_mat = np.zeros((len(y_D_array),len(x_D_array)))
    
    # iterate for each position of the flow field
    for y_ind in np.arange(len(y_D_array)):
        for x_ind in np.arange(len(x_D_array)):
            x_new = np.concatenate((x,np.array([x_D_array[x_ind]*diameter])))
            y_new = np.concatenate((y,np.array([y_D_array[y_ind]*diameter])))
            yaw_new = np.concatenate((yaw,np.array([0.])))
            simres = wfm(x_new,y_new,wd=wd,ws=ws,yaw=yaw_new,tilt=0)
            u_mat[y_ind,x_ind] = simres.WS_eff_ilk[-1]
            
        # print(f'Iteration {y_ind+1} of {len(y_D_array)} completed')
            
    return u_mat




def plot_U_comparison(cw_D_plot,
                      dw_D_plot,
                      u_mat_LES,
                      x_D_array_LES,
                      y_D_array_LES,
                      diameter,
                      x,
                      y,
                      wfm_list,
                      wfm_label_list,
                      wd,
                      ws,
                      yaw,
                      helix_amp,
                      savefig=False,
                      name_path=None,
                      name_fig=None):
    
    colors = ['b','k']
    
    fig, axes = plt.subplots(len(dw_D_plot), figsize=(10, 10), sharex=True)
    for i in np.arange(len(dw_D_plot)):
        
        dw = dw_D_plot[i]
        axes[i].set_title(f'Downstream distance: {dw} D')
        
        dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
        u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
        axes[i].scatter(y_D_array_LES,u_LES_slice,c='r',marker='.',label='LES')
        
        y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
        for wfm_ind in np.arange(len(wfm_list)):
            u_pywake_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_list[wfm_ind],wd,ws,yaw,helix_amp).reshape(-1)
            axes[i].plot(y_D_array_pywake,u_pywake_slice,c=colors[wfm_ind],label='pywake - '+wfm_label_list[wfm_ind])
        
        axes[i].set_ylabel('Wind speed [m/s]')
        axes[i].legend()
    
    axes[len(dw_D_plot)-1].set_xlabel('Cross-stream distance [D]')
    plt.tight_layout()
    if savefig: plt.savefig(name_path+'cw_section'+name_fig,format='svg')
    plt.show()
        
    
    fig, axes = plt.subplots(len(cw_D_plot), figsize=(10, 10), sharex=True)
    for i in np.arange(len(cw_D_plot)):
        
        cw = cw_D_plot[i]
        axes[i].set_title(f'Crosswind distance: {cw} D')
        
        cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
        u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
        axes[i].scatter(x_D_array_LES,u_LES_slice,c='r',marker='.',label='LES')
        
        x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
        for wfm_ind in np.arange(len(wfm_list)):
            u_pywake_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_list[wfm_ind],wd,ws,yaw,helix_amp).reshape(-1)
            axes[i].plot(x_D_array_pywake,u_pywake_slice,c=colors[wfm_ind],label='pywake - '+wfm_label_list[wfm_ind])
            
        axes[i].set_ylabel('Wind speed [m/s]')
        axes[i].legend()
    
    axes[len(cw_D_plot)-1].set_xlabel('Downstream distance [D]')
    plt.tight_layout()
    if savefig: plt.savefig(name_path+'dw_section'+name_fig,format='svg')
    plt.show()
    
    

# this function should be used when the effect of RotorAvg models is corrected to obtain the wind speed field (i.e. "RotorCenter values")
def compute_flow_with_conversion(x_D_array,
                                 y_D_array,
                                 diameter,
                                 x,
                                 y,
                                 wd,
                                 ws,
                                 yaw,
                                 tilt,
                                 helix_amp,
                                 sigma_0_D,
                                 k_1,
                                 k_2,
                                 mixing_gain_velocity,
                                 awc_wake_exp,
                                 awc_wake_denominator,
                                 hcw_deflection_gain_D,
                                 deflection_rate,
                                 mixing_gain_deflection,
                                 table_GaussOverlap):
    
    # initialize u_mat
    u_mat = np.zeros((len(y_D_array),len(x_D_array)))
    
    # iterate for each position of the flow field
    for y_ind in np.arange(len(y_D_array)):
        for x_ind in np.arange(len(x_D_array)):
            
            x_P = x_D_array[x_ind]*diameter
            y_P = y_D_array[y_ind]*diameter
            u_mat[y_ind,x_ind] = convert_u_from_GaussOverlap_to_RotorCenter(x_P,
                                                                            y_P,
                                                                            wd,
                                                                            ws,
                                                                            x,
                                                                            y,
                                                                            yaw,
                                                                            tilt,
                                                                            helix_amp,
                                                                            sigma_0_D,
                                                                            k_1,
                                                                            k_2,
                                                                            mixing_gain_velocity,
                                                                            awc_wake_exp,
                                                                            awc_wake_denominator,
                                                                            hcw_deflection_gain_D,
                                                                            deflection_rate,
                                                                            mixing_gain_deflection,
                                                                            table_GaussOverlap,
                                                                            superimposition_model='SquaredSum')
            
        # print(f'Iteration {y_ind+1} of {len(y_D_array)} completed')

    return u_mat

    
    
    
    
    
# fucntion used to extract the RotorCenter wind speed in a point P for a pywake simulation where GaussianOverlapAvgModel() is used
# the following superposition methods are supported: Linear, SquaredSum
def convert_u_from_GaussOverlap_to_RotorCenter(x_P,
                                               y_P,
                                               wd,
                                               ws,
                                               x,
                                               y,
                                               yaw,
                                               tilt,
                                               helix_amp,
                                               sigma_0_D,
                                               k_1,
                                               k_2,
                                               mixing_gain_velocity,
                                               awc_wake_exp,
                                               awc_wake_denominator,
                                               hcw_deflection_gain_D,
                                               deflection_rate,
                                               mixing_gain_deflection,
                                               table_GaussOverlap,
                                               superimposition_model='SquaredSum'):
    
    # LinearSum is assumed as superimposition model
    
    # define wfm (GaussOverlap)
    if superimposition_model=='Linear':
        wfm_GaussOverlap = PropagateDownwind_helix(site, wind_turbine,
                                                  wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
                                                                                             sigma_0_D=sigma_0_D,
                                                                                             mixing_gain_velocity=mixing_gain_velocity,
                                                                                             awc_wake_exp=awc_wake_exp,
                                                                                             awc_wake_denominator=awc_wake_denominator),
                                                  superpositionModel=LinearSum(),
                                                  deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
                                                                                              deflection_rate=deflection_rate,
                                                                                              mixing_gain_deflection=mixing_gain_deflection),
                                                  turbulenceModel=None,
                                                  rotorAvgModel=GaussianOverlapAvgModel())
    elif superimposition_model=='SquaredSum':
        wfm_GaussOverlap = PropagateDownwind_helix(site, wind_turbine,
                                                  wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
                                                                                             sigma_0_D=sigma_0_D,
                                                                                             mixing_gain_velocity=mixing_gain_velocity,
                                                                                             awc_wake_exp=awc_wake_exp,
                                                                                             awc_wake_denominator=awc_wake_denominator),
                                                  superpositionModel=SquaredSum(),
                                                  deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
                                                                                              deflection_rate=deflection_rate,
                                                                                              mixing_gain_deflection=mixing_gain_deflection),
                                                  turbulenceModel=None,
                                                  rotorAvgModel=GaussianOverlapAvgModel())
    else:
        TypeError('The superimposition model name given as input is not correct')

    
    # define Empirical Gaussian Model to extract its functions
    empgauss = EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
                                               sigma_0_D=sigma_0_D,
                                               mixing_gain_velocity=mixing_gain_velocity,
                                               awc_wake_exp=awc_wake_exp,
                                               awc_wake_denominator=awc_wake_denominator)
    
    
    # calculate teh contribution of every upstream turbine to a virtual turbine in P ------------------------------------------------------------------
    
    # initialize facotr for conversion and deficit
    f_conversion = np.zeros(len(x))
    defict_GaussOverlap_P = np.zeros(len(x))
    defict_RotorCenter_P = np.zeros(len(x))
    
    # extract ct
    simres = wfm_GaussOverlap(x,y,wd=wd,ws=ws,yaw=yaw,tilt=0,helix_amp=helix_amp)
    ct = simres.ct_ilk
    
    # calculate dw and cw
    x_mat_1 = x[:,na]
    x_mat_2 = x[na,:]
    y_mat_1 = y[:,na]
    y_mat_2 = y[na,:]
    d = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)
    theta = np.arctan2(y_mat_2-y_mat_1,x_mat_2-x_mat_1)
    gamma = wd*(np.pi/180)-(3/2)*np.pi+theta
    dw_mat = d*np.cos(gamma)
    cw_mat = d*np.sin(gamma)
    
    # order the turbines
    ind_dw_order = np.argsort(dw_mat[0,:])
    
    # initialize mixing (relevant only for the turbine positions)
    mixing_tot = empgauss._awc_added_mixing(helix_amp)
    
    
    # iterate for each turbine in downstream direction
    for i_dw in np.arange(len(ind_dw_order)):
        
        # iterate for each downstream turbine
        ind_dw_turb = ind_dw_order[i_dw+1:]
        for j in np.arange(len(ind_dw_turb)):
        
            # calculate mixing on the next turbine
            dw = np.array([dw_mat[ind_dw_order[i_dw],ind_dw_turb[j]]])
            cw = np.array([cw_mat[ind_dw_order[i_dw],ind_dw_turb[j]]])
            
            wake_radius = 2.0*empgauss.sigma_ijlk(dw[na,:,na,na],np.array([diameter])[na,:],mixing_tot[ind_dw_order[i_dw],na,na,na],yaw[ind_dw_order[i_dw],na,na,na],tilt[ind_dw_order[i_dw],na,na,na])
            mixing_tot[ind_dw_turb[j]] += empgauss._calc_mixing_i_to_j(np.array([diameter])[:,na,na],dw[na,:,na,na],cw[na,:,na,na],ct[ind_dw_order[i_dw],na,na,na],wake_radius,helix_amp[:,na,na])
        
        
        # calculate conversion factor
        
        d_P = np.sqrt((x_P-x[ind_dw_order[i_dw]])**2+(y_P-y[ind_dw_order[i_dw]])**2)
        theta_P = np.arctan2(y_P-y[ind_dw_order[i_dw]],x_P-x[ind_dw_order[i_dw]])
        gamma_P = wd*(np.pi/180)-(3/2)*np.pi+theta_P
        dw_P = d_P*np.cos(gamma_P)
        cw_P = d_P*np.sin(gamma_P)
        
        if dw_P>0:
            
            # calculate sigma at P position
            sigma_P = empgauss.sigma_ijlk(dw_P[na,:,na,na],np.array([diameter])[na,:],mixing_tot[ind_dw_order[i_dw],na,na,na],yaw[ind_dw_order[i_dw],na,na,na],tilt[ind_dw_order[i_dw],na,na,na])
        
            # create interpolator object from lookup table
            R_sigma = np.arange(0, 20.001, 0.01)
            CW_sigma = np.arange(0, 10.01, 0.01)
            dat = table_GaussOverlap.interp(R_sigma=R_sigma, CW_sigma=CW_sigma, method='cubic')
            overlap_interpolator = GridInterpolator([R_sigma, CW_sigma], dat, bounds='limit')
        
            # extract coefficient
            r_sigma = (diameter/2)/sigma_P
            cw_sigma = np.abs(cw_P/sigma_P)
            f_conversion[ind_dw_order[i_dw]] = overlap_interpolator(np.array([r_sigma.item(),cw_sigma.item()]))
    
    
            # calculate deficit from the upstream turbines up to this point
            x_temp = np.concatenate((x[:ind_dw_order[i_dw]+1],np.array([x_P])))
            y_temp = np.concatenate((y[:ind_dw_order[i_dw]+1],np.array([y_P])))
            yaw_temp = np.concatenate((yaw[:ind_dw_order[i_dw]+1],np.array([0])))
            tilt_temp = np.concatenate((tilt[:ind_dw_order[i_dw]+1],np.array([0])))
            helix_amp_temp = np.concatenate((helix_amp[:ind_dw_order[i_dw]+1],np.array([0])))
            ws_eff_P_GaussOverlap = wfm_GaussOverlap(x_temp,y_temp,wd=wd,ws=ws,yaw=yaw_temp,tilt=tilt_temp,helix_amp=helix_amp_temp).WS_eff_ilk[-1]
    
            # isolate the effect of only the iterating turbine (subtracting previous deficit -> LinearSum)
            if superimposition_model=='Linear':
                defict_GaussOverlap_P[ind_dw_order[i_dw]] = (ws-ws_eff_P_GaussOverlap)-np.sum(defict_GaussOverlap_P)
            elif superimposition_model=='SquaredSum':
                defict_GaussOverlap_P[ind_dw_order[i_dw]] = np.sqrt((ws-ws_eff_P_GaussOverlap)**2-np.sum(defict_GaussOverlap_P**2))
            else:
                TypeError('The superimposition model name given as input is not correct')
            
            # covert and save this defict into RotorCenter effective wind speed deficit
            defict_RotorCenter_P[ind_dw_order[i_dw]] = defict_GaussOverlap_P[ind_dw_order[i_dw]]/f_conversion[ind_dw_order[i_dw]]
    
    
    # sum up the deficits to obtain the RotorCenter effective wind speed (based on LinearSum)
    if superimposition_model=='Linear':
        ws_eff_P_RotorCenter = ws-np.sum(defict_RotorCenter_P)
    elif superimposition_model=='SquaredSum':
        ws_eff_P_RotorCenter = ws-np.sqrt(np.sum(defict_RotorCenter_P**2))
    else:
        TypeError('The superimposition model name given as input is not correct')

    return ws_eff_P_RotorCenter

    





#%% TUNE WAKE DEFICIT MODEL ====================================================================================
# simulation: Baseline - 1 wt  =================================================================================
# coefficients (3): sigma_0_D, k_1, k_2  =======================================================================
# ==============================================================================================================

# extract data -------------------------------------------------------------------------------------------------
u_mat_LES,x_D_array,y_D_array = extract_data_df(df_baseline_1wt)


# define case study --------------------------------------------------------------------------------------------

# deifne site (HKN) - NOT RELEVANT FOR THE TUNING
wd_site = np.linspace(0,360,12,endpoint=False)
p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)

# define turbine
powerCtFunction = PowerCtFunction(
    input_keys=['ws','helix_amp'],
    power_ct_func=helix_power_ct_function,
    power_unit='kW',
)
wind_turbine = WindTurbine(name='IEA22MW_helix',
                diameter=283.2,
                hub_height=170.0,
                powerCtFunction=powerCtFunction)    
diameter = wind_turbine.diameter()

# define INPUT
x = np.array([0])*diameter
y = np.array([0])*diameter
wd = np.array([270])
ws = np.array([10])
yaw = np.array([0])
helix_amp = np.array([0])


# function to minimize -------------------------------------------------------------------------------------------

# function of residual between wind fields cubed
def residuals_Ucube(coeff):
    
    # define model with the input coefficients
    sigma_0_D = coeff[0]
    k_1 = coeff[1]
    k_2 = coeff[2]
    wfm = PropagateDownwind_helix(site, wind_turbine,
                                  wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
                                                                             sigma_0_D=sigma_0_D),
                                  superpositionModel=LinearSum(),
                                  deflectionModel=EmpiricalGaussianDeflection(),
                                  turbulenceModel=None,
                                  rotorAvgModel=RotorCenter())

    # compute wind field with pywake
    u_mat_pywake = compute_flow(x_D_array = x_D_array,
                                y_D_array = y_D_array,
                                diameter = diameter,
                                x = x,
                                y = y,
                                wfm = wfm,
                                wd = wd,
                                ws = ws,
                                yaw = yaw,
                                helix_amp = helix_amp)

    # compute residuals of U cubed
    residuals = (np.abs(u_mat_LES**3-u_mat_pywake**3)).reshape(-1)
    
    return residuals


#%% optimize ------------------------------------------------------------------------------------------------------

# run optmization
coeff_0 = np.array([0.28,0.023,0.008])
t = time.time()
res = least_squares(residuals_Ucube,coeff_0,verbose=2)
print(f'Optimization completed - Time: {time.time()-t}')    # around 6 min in this case
coeff_opt = res.x
sigma_0_D_opt = coeff_opt[0]
k_1_opt = coeff_opt[1]
k_2_opt = coeff_opt[2]

print('Optimal coefficients:')
print(f'sigma_0_D: \t {sigma_0_D_opt}')
print(f'k_1: \t \t {k_1_opt}')
print(f'k_2: \t \t {k_2_opt}')


#%% plot ------------------------------------------------------------------------------------------------------

# define downstream and cross-stream slices
dw_D_plot = np.array([3.,4.5,6.,7.5])
cw_D_plot = np.array([-0.75,0.,0.75])

# wind farm model with OPTIMAL coefficients
wfm_opt = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[0.01213,0.008],
                                                                         sigma_0_D=0.3042),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())

# wind farm model with INITIAL coefficients
wfm_initial = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())


# plot comparison
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'tuningDeficit__baseline1wt.svg'
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
x_D_array_LES = x_D_array
y_D_array_LES = y_D_array
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES,
                  x_D_array_LES,
                  y_D_array_LES,
                  diameter,
                  x,
                  y,
                  wfm_list,
                  wfm_label_list,
                  wd,
                  ws,
                  yaw,
                  helix_amp,
                  savefig=True,
                  name_path=name_path,
                  name_fig=name_fig)





#%% TUNE WAKE DEFICIT MODEL ====================================================================================
# simulation: Baseline - 3 wt  =================================================================================
# coefficients (1): mixing_gain_velocity  ======================================================================
# ==============================================================================================================

# extract data -------------------------------------------------------------------------------------------------
u_mat_LES_1,x_D_array_1,y_D_array_1 = extract_data_df(df_baseline_3wt_aligned)
#u_mat_LES_2,x_D_array_2,y_D_array_2 = extract_data_df(df_baseline_3wt_misaligned)


# define case study --------------------------------------------------------------------------------------------

# deifne site (HKN) - NOT RELEVANT FOR THE TUNING
wd_site = np.linspace(0,360,12,endpoint=False)
p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)

# define turbine
powerCtFunction = PowerCtFunction(
    input_keys=['ws','helix_amp'],
    power_ct_func = partial(helix_power_ct_function,
                            helix_a = 1.907,
                            helix_power_b = 1.376e-3,
                            helix_power_c = 4.017e-11,  # not tuned
                            helix_thrust_b = 0.8371e-3,
                            helix_thrust_c = 5.084e-4),  # not tuned
    power_unit='kW',
)
wind_turbine = WindTurbine(name='IEA22MW_helix',
                diameter=283.2,
                hub_height=170.0,
                powerCtFunction=powerCtFunction)    
diameter = wind_turbine.diameter()

# define INPUT
x = np.array([0.,4.5,9.])*diameter
y = np.array([0.,0.,0.])*diameter
wd = np.array([270])
ws = np.array([10])
yaw = np.array([0,0,0])
tilt = np.array([0,0,0])
helix_amp = np.array([0,0,0])

# other inputs
os.chdir(r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation')
filename = 'gaussian_overlap_.02_.02_128_512.nc'
table_GaussOverlap = xr.load_dataarray(filename, engine='h5netcdf')

# model coefficients
sigma_0_D = 0.3042          # tuned
k_1 = 0.01213               # tuned
k_2 = 0.008                 # tuned
mixing_gain_velocity = 2.   # NOT tuned yet
awc_wake_exp = 1.2          # NOT tuned yet
awc_wake_denominator = 400. # NOT tuned yet
hcw_deflection_gain_D = 3.  # NOT tuned yet
deflection_rate = 22.       # NOT tuned yet
mixing_gain_deflection = 0. # NOT tuned yet


# function to minimize -------------------------------------------------------------------------------------------

u_mat_LES = u_mat_LES_1
x_D_array = x_D_array_1
y_D_array = y_D_array_1

# function of residual between wind fields cubed
def residuals_Ucube(coeff):
    
    # define model with the input coefficients
    mixing_gain_velocity = coeff[0]
    wfm = PropagateDownwind_helix(site, wind_turbine,
                                  wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
                                                                             sigma_0_D=sigma_0_D,
                                                                             mixing_gain_velocity=mixing_gain_velocity,
                                                                             awc_wake_exp=awc_wake_exp,
                                                                             awc_wake_denominator=awc_wake_denominator),
                                  superpositionModel=SquaredSum(),
                                  deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
                                                                              deflection_rate=deflection_rate,
                                                                              mixing_gain_deflection=mixing_gain_deflection),
                                  turbulenceModel=None,
                                  rotorAvgModel=RotorCenter())

    # compute wind field with pywake
    u_mat_pywake = compute_flow(x_D_array = x_D_array,
                                y_D_array = y_D_array,
                                diameter = diameter,
                                x = x,
                                y = y,
                                wfm = wfm,
                                wd = wd,
                                ws = ws,
                                yaw = yaw,
                                helix_amp = helix_amp)

    # compute residuals of U cubed
    residuals = (np.abs(u_mat_LES**3-u_mat_pywake**3)).reshape(-1)
    
    return residuals



#%% optimize ------------------------------------------------------------------------------------------------------

# run optmization
coeff_0 = np.array([2.])
t = time.time()
res = least_squares(residuals_Ucube,coeff_0,verbose=2)
print(f'Optimization completed - Time: {time.time()-t}')    # around 3.5 min in this case
coeff_opt = res.x
mixing_gain_velocity_opt = coeff_opt[0]

print('Optimal coefficients:')
print(f'mixing_gain_velocity: \t {mixing_gain_velocity_opt}')



#%% plot ------------------------------------------------------------------------------------------------------

# define downstream and cross-stream slices
dw_D_plot = np.array([7.,8.5,11.5,13.])
cw_D_plot = np.array([-0.75,0.,0.75])

# wind farm model with OPTIMAL coefficients
wfm_opt = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[0.01213,0.008],
                                                                         sigma_0_D=0.3042,
                                                                         mixing_gain_velocity=0.2119,
                                                                         awc_wake_exp=1.2,
                                                                         awc_wake_denominator=400.),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=3.,
                                                                          deflection_rate=22.,
                                                                          mixing_gain_deflection=0.),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())

# wind farm model with INITIAL coefficients
wfm_initial = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())


# plot comparison
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'tuningDeficit__baseline3wt.svg'
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
x_D_array_LES = x_D_array
y_D_array_LES = y_D_array
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES,
                  x_D_array_LES,
                  y_D_array_LES,
                  diameter,
                  x,
                  y,
                  wfm_list,
                  wfm_label_list,
                  wd,
                  ws,
                  yaw,
                  helix_amp,
                  savefig=True,
                  name_path=name_path,
                  name_fig=name_fig)



#%% TUNE WAKE DEFICIT MODEL (with conversion from RotorAVg model) ==============================================
# simulation: Baseline - 3 wt  =================================================================================
# coefficients (1): mixing_gain_velocity  ======================================================================
# ==============================================================================================================

# extract data -------------------------------------------------------------------------------------------------
u_mat_LES_1,x_D_array_1,y_D_array_1 = extract_data_df(df_baseline_3wt_aligned)
#u_mat_LES_2,x_D_array_2,y_D_array_2 = extract_data_df(df_baseline_3wt_misaligned)


# define case study --------------------------------------------------------------------------------------------

# deifne site (HKN) - NOT RELEVANT FOR THE TUNING
wd_site = np.linspace(0,360,12,endpoint=False)
p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)

# define turbine
powerCtFunction = PowerCtFunction(
    input_keys=['ws','helix_amp'],
    power_ct_func = partial(helix_power_ct_function,
                            helix_a = 1.907,
                            helix_power_b = 1.376e-3,
                            helix_power_c = 4.017e-11,  # not tuned
                            helix_thrust_b = 0.8371e-3,
                            helix_thrust_c = 5.084e-4),  # not tuned
    power_unit='kW',
)
wind_turbine = WindTurbine(name='IEA22MW_helix',
                diameter=283.2,
                hub_height=170.0,
                powerCtFunction=powerCtFunction)    
diameter = wind_turbine.diameter()

# define INPUT
x = np.array([0.,4.5,9.])*diameter
y = np.array([0.,0.,0.])*diameter
wd = np.array([270])
ws = np.array([10])
yaw = np.array([0,0,0])
tilt = np.array([0,0,0])
helix_amp = np.array([0,0,0])

# other inputs
os.chdir(r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation')
filename = 'gaussian_overlap_.02_.02_128_512.nc'
table_GaussOverlap = xr.load_dataarray(filename, engine='h5netcdf')

# model coefficients
sigma_0_D = 0.3042          # tuned
k_1 = 0.01213               # tuned
k_2 = 0.008                 # tuned
mixing_gain_velocity = 2.   # NOT tuned yet
awc_wake_exp = 1.2          # NOT tuned yet
awc_wake_denominator = 400. # NOT tuned yet
hcw_deflection_gain_D = 3.  # NOT tuned yet
deflection_rate = 22.       # NOT tuned yet
mixing_gain_deflection = 0. # NOT tuned yet


# function to minimize -------------------------------------------------------------------------------------------

u_mat_LES = u_mat_LES_1
x_D_array = x_D_array_1
y_D_array = y_D_array_1

# function of residual between wind fields cubed
def residuals_Ucube(coeff):
    
    # define model with the input coefficients
    mixing_gain_velocity = coeff[0]
    wfm = PropagateDownwind_helix(site, wind_turbine,
                                  wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
                                                                             sigma_0_D=sigma_0_D,
                                                                             mixing_gain_velocity=mixing_gain_velocity,
                                                                             awc_wake_exp=awc_wake_exp,
                                                                             awc_wake_denominator=awc_wake_denominator),
                                  superpositionModel=SquaredSum(),
                                  deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
                                                                              deflection_rate=deflection_rate,
                                                                              mixing_gain_deflection=mixing_gain_deflection),
                                  turbulenceModel=None,
                                  rotorAvgModel=RotorCenter())

    # compute wind field with pywake
    u_mat_pywake = compute_flow(x_D_array = x_D_array,
                                y_D_array = y_D_array,
                                diameter = diameter,
                                x = x,
                                y = y,
                                wfm = wfm,
                                wd = wd,
                                ws = ws,
                                yaw = yaw,
                                helix_amp = helix_amp)
    
    u_mat_pywake = compute_flow_with_conversion(x_D_array,
                                                y_D_array,
                                                diameter,
                                                x,
                                                y,
                                                wd,
                                                ws,
                                                yaw,
                                                tilt,
                                                helix_amp,
                                                sigma_0_D,
                                                k_1,
                                                k_2,
                                                mixing_gain_velocity,
                                                awc_wake_exp,
                                                awc_wake_denominator,
                                                hcw_deflection_gain_D,
                                                deflection_rate,
                                                mixing_gain_deflection,
                                                table_GaussOverlap)

    # compute residuals of U cubed
    residuals = (np.abs(u_mat_LES**3-u_mat_pywake**3)).reshape(-1)
    
    return residuals


#% optimize ------------------------------------------------------------------------------------------------------

# run optmization
coeff_0 = np.array([2.])
t = time.time()
res = least_squares(residuals_Ucube,coeff_0,verbose=2)
print(f'Optimization completed - Time: {time.time()-t}')    # around 3.5 min in this case
coeff_opt = res.x
mixing_gain_velocity_opt = coeff_opt[0]

print('Optimal coefficients:')
print(f'mixing_gain_velocity: \t {mixing_gain_velocity_opt}')




# #%% ----- TEST 1 (around 9 seconds)


# wfm_RotorCenter_SS = PropagateDownwind_helix(site, wind_turbine,
#                                           wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
#                                                                                      sigma_0_D=sigma_0_D,
#                                                                                      awc_wake_exp=awc_wake_exp,
#                                                                                      awc_wake_denominator=awc_wake_denominator),
#                                           superpositionModel=SquaredSum(),
#                                           deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
#                                                                                       deflection_rate=deflection_rate,
#                                                                                       mixing_gain_deflection=mixing_gain_deflection),
#                                           turbulenceModel=None,
#                                           rotorAvgModel=RotorCenter())

# t = time.time()
# u_mat_1 = compute_flow(x_D_array_1,y_D_array_1,diameter,x,y,wfm_RotorCenter_SS,wd,ws,yaw,helix_amp)
# print(f'Simulation without conversion completed - Time: {time.time()-t}')



# #%% ----- TEST 2 (around 523 seconds -> 60 times more)

# t = time.time()
# u_mat_2 = compute_flow_with_conversion(x_D_array_1,
#                                      y_D_array_1,
#                                      diameter,
#                                      x,
#                                      y,
#                                      wd,
#                                      ws,
#                                      yaw,
#                                      tilt,
#                                      helix_amp,
#                                      sigma_0_D,
#                                      k_1,
#                                      k_2,
#                                      mixing_gain_velocity,
#                                      awc_wake_exp,
#                                      awc_wake_denominator,
#                                      hcw_deflection_gain_D,
#                                      deflection_rate,
#                                      mixing_gain_deflection,
#                                      table_GaussOverlap)

# print(f'Simulation with conversion completed - Time: {time.time()-t}')


# #%%

# error_mat = u_mat_1-u_mat_2
# # max difference = 0.018 m/s

#%% TUNE WAKE DEFICIT MODEL (HELIX) ============================================================================
# simulation: Helix - 1 and 3 wt  ==============================================================================
# coefficients (2): awc_wake_exponent, awc_wake_denominator  ===================================================
# ==============================================================================================================

# extract data -------------------------------------------------------------------------------------------------
u_mat_LES_list = [None]*5
x_D_array_list = [None]*5
y_D_array_list = [None]*5
u_mat_LES_list[0],x_D_array_list[0],y_D_array_list[0] = extract_data_df(df_helix2deg_1wt)
u_mat_LES_list[1],x_D_array_list[1],y_D_array_list[1] = extract_data_df(df_helix3deg_1wt)
u_mat_LES_list[2],x_D_array_list[2],y_D_array_list[2] = extract_data_df(df_helix4deg_1wt)
u_mat_LES_list[3],x_D_array_list[3],y_D_array_list[3] = extract_data_df(df_helix3deg_3wt)
u_mat_LES_list[4],x_D_array_list[4],y_D_array_list[4] = extract_data_df(df_helix4deg_3wt)


# define case study --------------------------------------------------------------------------------------------

# deifne site (HKN) - NOT RELEVANT FOR THE TUNING
wd_site = np.linspace(0,360,12,endpoint=False)
p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)

# define turbine
powerCtFunction = PowerCtFunction(
    input_keys=['ws','helix_amp'],
    power_ct_func = partial(helix_power_ct_function,
                            helix_a = 1.907,
                            helix_power_b = 1.376e-3,
                            helix_power_c = 4.017e-11,  # not tuned
                            helix_thrust_b = 0.8371e-3,
                            helix_thrust_c = 5.084e-4),  # not tuned
    power_unit='kW',
)
wind_turbine = WindTurbine(name='IEA22MW_helix',
                diameter=283.2,
                hub_height=170.0,
                powerCtFunction=powerCtFunction)    
diameter = wind_turbine.diameter()


# define INPUT -----------------------------
x_list = [None]*5
y_list = [None]*5
wd_list = [np.array([270])]*5
ws_list = [np.array([10])]*5
yaw_list = [None]*5
tilt_list = [None]*5
helix_amp_list = [None]*5

# case 1
x_list[0] = np.array([0.])*diameter
y_list[0] = np.array([0.])*diameter
yaw_list[0] = np.array([0.])
tilt_list[0] = np.array([0.])
helix_amp_list[0] = np.array([2.])

# case 2
x_list[1] = np.array([0.])*diameter
y_list[1] = np.array([0.])*diameter
yaw_list[1] = np.array([0.])
tilt_list[1] = np.array([0.])
helix_amp_list[1] = np.array([3.])

# case 3
x_list[2] = np.array([0.])*diameter
y_list[2] = np.array([0.])*diameter
yaw_list[2] = np.array([0.])
tilt_list[2] = np.array([0.])
helix_amp_list[2] = np.array([4.])

# case 4
x_list[3] = np.array([0.,4.5,9.])*diameter
y_list[3] = np.array([0.,0.,0.])*diameter
yaw_list[3] = np.array([0.,0.,0.])
tilt_list[3] = np.array([0.,0.,0.])
helix_amp_list[3] = np.array([3.,0.,0.])

# case 5
x_list[4] = np.array([0.,4.5,9.])*diameter
y_list[4] = np.array([0.,0.,0.])*diameter
yaw_list[4] = np.array([0.,0.,0.])
tilt_list[4] = np.array([0.,0.,0.])
helix_amp_list[4] = np.array([4.,0.,0.])

# ------------------------------------


# other inputs
# os.chdir(r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation')
# filename = 'gaussian_overlap_.02_.02_128_512.nc'
# table_GaussOverlap = xr.load_dataarray(filename, engine='h5netcdf')

# model coefficients
sigma_0_D = 0.3042              # tuned
k_1 = 0.01213                   # tuned
k_2 = 0.008                     # tuned
mixing_gain_velocity = 0.2119   # tuned
awc_wake_exp = 1.2              # NOT tuned yet
awc_wake_denominator = 400.     # NOT tuned yet
hcw_deflection_gain_D = 3.      # NOT tuned yet
deflection_rate = 22.           # NOT tuned yet
mixing_gain_deflection = 0.     # NOT tuned yet


# function to minimize -------------------------------------------------------------------------------------------

# function of residual between wind fields cubed
def residuals_Ucube(coeff):
    
    # initialize resilduals
    residuals = np.array([])
    
    # iterate for each case
    for i in np.arange(len(u_mat_LES_list)):
        
        # extract case
        u_mat_LES = u_mat_LES_list[i]
        x_D_array = x_D_array_list[i]
        y_D_array = y_D_array_list[i]
        x = x_list[i]
        y = y_list[i]
        wd = wd_list[i]
        ws = ws_list[i]
        yaw = yaw_list[i]
        helix_amp = helix_amp_list[i]
    
        # define model with the input coefficients
        awc_wake_exp = coeff[0]
        awc_wake_denominator = coeff[1]
        wfm = PropagateDownwind_helix(site, wind_turbine,
                                      wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
                                                                                 sigma_0_D=sigma_0_D,
                                                                                 mixing_gain_velocity=mixing_gain_velocity,
                                                                                 awc_wake_exp=awc_wake_exp,
                                                                                 awc_wake_denominator=awc_wake_denominator),
                                      superpositionModel=SquaredSum(),
                                      deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
                                                                                  deflection_rate=deflection_rate,
                                                                                  mixing_gain_deflection=mixing_gain_deflection),
                                      turbulenceModel=None,
                                      rotorAvgModel=RotorCenter())
    
        # compute wind field with pywake
        u_mat_pywake = compute_flow(x_D_array = x_D_array,
                                    y_D_array = y_D_array,
                                    diameter = diameter,
                                    x = x,
                                    y = y,
                                    wfm = wfm,
                                    wd = wd,
                                    ws = ws,
                                    yaw = yaw,
                                    helix_amp = helix_amp)
    
        # compute residuals of U cubed
        residuals_temp = (np.abs(u_mat_LES**3-u_mat_pywake**3)).reshape(-1)
        residuals = np.concatenate((residuals,residuals_temp))
    
    return residuals


    

#%% optimize ------------------------------------------------------------------------------------------------------

# run optmization
coeff_0 = np.array([1.2,400.])
t = time.time()
res = least_squares(residuals_Ucube,coeff_0,verbose=2)
print(f'Optimization completed - Time: {time.time()-t}')    # around 40 min in this case
coeff_opt = res.x
awc_wake_exp_opt = coeff_opt[0]
awc_wake_denominator_opt = coeff_opt[1]

print('Optimal coefficients:')
print(f'awc_wake_exp: \t {awc_wake_exp_opt}')
print(f'awc_wake_denominator: \t {awc_wake_denominator_opt}')



#%% plot ------------------------------------------------------------------------------------------------------

# wind farm model with OPTIMAL coefficients
wfm_opt = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[0.01213,0.008],
                                                                         sigma_0_D=0.3042,
                                                                         mixing_gain_velocity=0.2119,
                                                                         awc_wake_exp=1.119,
                                                                         awc_wake_denominator=137.21),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=3.,
                                                                          deflection_rate=22.,
                                                                          mixing_gain_deflection=0.),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())

# wind farm model with INITIAL coefficients
wfm_initial = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())


# plot comparison (1wt - helix=2deg)
ind_case = 0
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'_tuningDeficit_helix2deg1wt.svg'
dw_D_plot = np.array([3.,4.5,6.,7.5])
cw_D_plot = np.array([-0.75,0.,0.75])
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES = u_mat_LES_list[ind_case],
                  x_D_array_LES = x_D_array_list[ind_case],
                  y_D_array_LES = y_D_array_list[ind_case],
                  diameter = diameter,
                  x = x_list[ind_case],
                  y = y_list[ind_case],
                  wfm_list = wfm_list,
                  wfm_label_list = wfm_label_list,
                  wd = wd_list[ind_case],
                  ws = ws_list[ind_case],
                  yaw = yaw_list[ind_case],
                  helix_amp = helix_amp_list[ind_case],
                  savefig=False,
                  name_path=name_path,
                  name_fig=name_fig)


# plot comparison (1wt - helix=3deg)
ind_case = 1
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'_tuningDeficit_helix3deg1wt.svg'
dw_D_plot = np.array([3.,4.5,6.,7.5])
cw_D_plot = np.array([-0.75,0.,0.75])
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES = u_mat_LES_list[ind_case],
                  x_D_array_LES = x_D_array_list[ind_case],
                  y_D_array_LES = y_D_array_list[ind_case],
                  diameter = diameter,
                  x = x_list[ind_case],
                  y = y_list[ind_case],
                  wfm_list = wfm_list,
                  wfm_label_list = wfm_label_list,
                  wd = wd_list[ind_case],
                  ws = ws_list[ind_case],
                  yaw = yaw_list[ind_case],
                  helix_amp = helix_amp_list[ind_case],
                  savefig=False,
                  name_path=name_path,
                  name_fig=name_fig)


# plot comparison (1wt - helix=4deg)
ind_case = 2
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'_tuningDeficit_helix4deg1wt.svg'
dw_D_plot = np.array([3.,4.5,6.,7.5])
cw_D_plot = np.array([-0.75,0.,0.75])
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES = u_mat_LES_list[ind_case],
                  x_D_array_LES = x_D_array_list[ind_case],
                  y_D_array_LES = y_D_array_list[ind_case],
                  diameter = diameter,
                  x = x_list[ind_case],
                  y = y_list[ind_case],
                  wfm_list = wfm_list,
                  wfm_label_list = wfm_label_list,
                  wd = wd_list[ind_case],
                  ws = ws_list[ind_case],
                  yaw = yaw_list[ind_case],
                  helix_amp = helix_amp_list[ind_case],
                  savefig=False,
                  name_path=name_path,
                  name_fig=name_fig)


# plot comparison (3wt - helix=3deg)
ind_case = 3
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'_tuningDeficit_helix3deg3wt.svg'
dw_D_plot = np.array([7.,8.5,11.5,13.])
cw_D_plot = np.array([-0.75,0.,0.75])
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES = u_mat_LES_list[ind_case],
                  x_D_array_LES = x_D_array_list[ind_case],
                  y_D_array_LES = y_D_array_list[ind_case],
                  diameter = diameter,
                  x = x_list[ind_case],
                  y = y_list[ind_case],
                  wfm_list = wfm_list,
                  wfm_label_list = wfm_label_list,
                  wd = wd_list[ind_case],
                  ws = ws_list[ind_case],
                  yaw = yaw_list[ind_case],
                  helix_amp = helix_amp_list[ind_case],
                  savefig=False,
                  name_path=name_path,
                  name_fig=name_fig)


# plot comparison (3wt - helix=4deg)
ind_case = 4
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'_tuningDeficit_helix4deg3wt.svg'
dw_D_plot = np.array([7.,8.5,11.5,13.])
cw_D_plot = np.array([-0.75,0.,0.75])
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES = u_mat_LES_list[ind_case],
                  x_D_array_LES = x_D_array_list[ind_case],
                  y_D_array_LES = y_D_array_list[ind_case],
                  diameter = diameter,
                  x = x_list[ind_case],
                  y = y_list[ind_case],
                  wfm_list = wfm_list,
                  wfm_label_list = wfm_label_list,
                  wd = wd_list[ind_case],
                  ws = ws_list[ind_case],
                  yaw = yaw_list[ind_case],
                  helix_amp = helix_amp_list[ind_case],
                  savefig=False,
                  name_path=name_path,
                  name_fig=name_fig)



#%% TUNE WAKE DEFLECTION MODEL (YAW) ==========================================================================
# simulation: Yaw - 1 wt  =====================================================================================
# coefficients (2): hcw_deflection_gain_D, deflection_rate  ===================================================
# =============================================================================================================

# extract data -------------------------------------------------------------------------------------------------
u_mat_LES_list = [None]*2
x_D_array_list = [None]*2
y_D_array_list = [None]*2
u_mat_LES_list[0],x_D_array_list[0],y_D_array_list[0] = extract_data_df(df_yaw20deg_1wt)
u_mat_LES_list[1],x_D_array_list[1],y_D_array_list[1] = extract_data_df(df_yawmin20deg_1wt)


# define case study --------------------------------------------------------------------------------------------

# deifne site (HKN) - NOT RELEVANT FOR THE TUNING
wd_site = np.linspace(0,360,12,endpoint=False)
p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)

# define turbine
powerCtFunction = PowerCtFunction(
    input_keys=['ws','helix_amp'],
    power_ct_func = partial(helix_power_ct_function,
                            helix_a = 1.907,
                            helix_power_b = 1.376e-3,
                            helix_power_c = 4.017e-11,  # not tuned
                            helix_thrust_b = 0.8371e-3,
                            helix_thrust_c = 5.084e-4),  # not tuned
    power_unit='kW',
)
wind_turbine = WindTurbine(name='IEA22MW_helix',
                diameter=283.2,
                hub_height=170.0,
                powerCtFunction=powerCtFunction)    
diameter = wind_turbine.diameter()


# define INPUT -----------------------------
x_list = [None]*2
y_list = [None]*2
wd_list = [np.array([270])]*2
ws_list = [np.array([10])]*2
yaw_list = [None]*2
tilt_list = [None]*2
helix_amp_list = [None]*2

# case 1
x_list[0] = np.array([0.])*diameter
y_list[0] = np.array([0.])*diameter
yaw_list[0] = np.array([20.])
tilt_list[0] = np.array([0.])
helix_amp_list[0] = np.array([0.])

# case 2
x_list[1] = np.array([0.])*diameter
y_list[1] = np.array([0.])*diameter
yaw_list[1] = np.array([-20.])
tilt_list[1] = np.array([0.])
helix_amp_list[1] = np.array([0.])


# ------------------------------------


# other inputs
# os.chdir(r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation')
# filename = 'gaussian_overlap_.02_.02_128_512.nc'
# table_GaussOverlap = xr.load_dataarray(filename, engine='h5netcdf')

# model coefficients
sigma_0_D = 0.3042              # tuned
k_1 = 0.01213                   # tuned
k_2 = 0.008                     # tuned
mixing_gain_velocity = 0.2119   # tuned
awc_wake_exp = 1.119            # tuned
awc_wake_denominator = 137.21   # tuned
hcw_deflection_gain_D = 3.      # NOT tuned yet
deflection_rate = 22.           # NOT tuned yet
mixing_gain_deflection = 0.     # NOT tuned yet


# function to minimize -------------------------------------------------------------------------------------------

# function of residual between wind fields cubed
def residuals_Ucube(coeff):
    
    # initialize resilduals
    residuals = np.array([])
    
    # iterate for each case
    for i in np.arange(len(u_mat_LES_list)):
        
        # extract case
        u_mat_LES = u_mat_LES_list[i]
        x_D_array = x_D_array_list[i]
        y_D_array = y_D_array_list[i]
        x = x_list[i]
        y = y_list[i]
        wd = wd_list[i]
        ws = ws_list[i]
        yaw = yaw_list[i]
        helix_amp = helix_amp_list[i]
    
        # define model with the input coefficients
        hcw_deflection_gain_D = coeff[0]
        deflection_rate = coeff[1]
        wfm = PropagateDownwind_helix(site, wind_turbine,
                                      wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
                                                                                 sigma_0_D=sigma_0_D,
                                                                                 mixing_gain_velocity=mixing_gain_velocity,
                                                                                 awc_wake_exp=awc_wake_exp,
                                                                                 awc_wake_denominator=awc_wake_denominator),
                                      superpositionModel=SquaredSum(),
                                      deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
                                                                                  deflection_rate=deflection_rate,
                                                                                  mixing_gain_deflection=mixing_gain_deflection),
                                      turbulenceModel=None,
                                      rotorAvgModel=RotorCenter())
    
        # compute wind field with pywake
        u_mat_pywake = compute_flow(x_D_array = x_D_array,
                                    y_D_array = y_D_array,
                                    diameter = diameter,
                                    x = x,
                                    y = y,
                                    wfm = wfm,
                                    wd = wd,
                                    ws = ws,
                                    yaw = yaw,
                                    helix_amp = helix_amp)
    
        # compute residuals of U cubed
        residuals_temp = (np.abs(u_mat_LES**3-u_mat_pywake**3)).reshape(-1)
        residuals = np.concatenate((residuals,residuals_temp))
    
    return residuals



#%% optimize ------------------------------------------------------------------------------------------------------

# run optmization
coeff_0 = np.array([3.,22.])
t = time.time()
res = least_squares(residuals_Ucube,coeff_0,verbose=2)
print(f'Optimization completed - Time: {time.time()-t}')    # around 4 min in this case
coeff_opt = res.x
hcw_deflection_gain_D_opt = coeff_opt[0]
deflection_rate_opt = coeff_opt[1]

print('Optimal coefficients:')
print(f'hcw_deflection_gain_D: \t {hcw_deflection_gain_D_opt}')
print(f'deflection_rate: \t {deflection_rate_opt}')


#%% plot ------------------------------------------------------------------------------------------------------

# wind farm model with OPTIMAL coefficients
wfm_opt = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[0.01213,0.008],
                                                                         sigma_0_D=0.3042,
                                                                         mixing_gain_velocity=0.2166,
                                                                         awc_wake_exp=1.130,
                                                                         awc_wake_denominator=145.5),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=2.0984,
                                                                          deflection_rate=12.018,
                                                                          mixing_gain_deflection=0.),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())

# wind farm model with INITIAL coefficients
wfm_initial = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())



# plot comparison (1wt - yaw=20deg)
ind_case = 0
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'_tuningDeflection_yaw20deg1wt.svg'
dw_D_plot = np.array([3.,4.5,6.,7.5])
cw_D_plot = np.array([-0.75,0.,0.75])
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES = u_mat_LES_list[ind_case],
                  x_D_array_LES = x_D_array_list[ind_case],
                  y_D_array_LES = y_D_array_list[ind_case],
                  diameter = diameter,
                  x = x_list[ind_case],
                  y = y_list[ind_case],
                  wfm_list = wfm_list,
                  wfm_label_list = wfm_label_list,
                  wd = wd_list[ind_case],
                  ws = ws_list[ind_case],
                  yaw = yaw_list[ind_case],
                  helix_amp = helix_amp_list[ind_case],
                  savefig=False,
                  name_path=name_path,
                  name_fig=name_fig)


# plot comparison (1wt - yaw=-20deg)
ind_case = 1
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'_tuningDeflection_yawmin20deg1wt.svg'
dw_D_plot = np.array([3.,4.5,6.,7.5])
cw_D_plot = np.array([-0.75,0.,0.75])
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES = u_mat_LES_list[ind_case],
                  x_D_array_LES = x_D_array_list[ind_case],
                  y_D_array_LES = y_D_array_list[ind_case],
                  diameter = diameter,
                  x = x_list[ind_case],
                  y = y_list[ind_case],
                  wfm_list = wfm_list,
                  wfm_label_list = wfm_label_list,
                  wd = wd_list[ind_case],
                  ws = ws_list[ind_case],
                  yaw = yaw_list[ind_case],
                  helix_amp = helix_amp_list[ind_case],
                  savefig=False,
                  name_path=name_path,
                  name_fig=name_fig)




#%% TUNE WAKE DEFLECTION MODEL (YAW) ============================================================================
# simulation: yaw - 3 wt  ======================================================================================
# coefficients (1): mixing_gain_deflection  ====================================================================
# ==============================================================================================================

# extract data -------------------------------------------------------------------------------------------------
u_mat_LES_list = [None]*1
x_D_array_list = [None]*1
y_D_array_list = [None]*1
u_mat_LES_list[0],x_D_array_list[0],y_D_array_list[0] = extract_data_df(df_yawGeom_3wt)


# define case study --------------------------------------------------------------------------------------------

# deifne site (HKN) - NOT RELEVANT FOR THE TUNING
wd_site = np.linspace(0,360,12,endpoint=False)
p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)

# define turbine
powerCtFunction = PowerCtFunction(
    input_keys=['ws','helix_amp'],
    power_ct_func = partial(helix_power_ct_function,
                            helix_a = 1.907,
                            helix_power_b = 1.376e-3,
                            helix_power_c = 4.017e-11,  # not tuned
                            helix_thrust_b = 0.8371e-3,
                            helix_thrust_c = 5.084e-4),  # not tuned
    power_unit='kW',
)
wind_turbine = WindTurbine(name='IEA22MW_helix',
                diameter=283.2,
                hub_height=170.0,
                powerCtFunction=powerCtFunction)    
diameter = wind_turbine.diameter()


# define INPUT -----------------------------
x_list = [None]*1
y_list = [None]*1
wd_list = [np.array([270])]*1
ws_list = [np.array([10])]*1
yaw_list = [None]*1
tilt_list = [None]*1
helix_amp_list = [None]*1

# case 1
x_list[0] = np.array([0.,4.5,9.])*diameter
y_list[0] = np.array([0.,0.,0.])*diameter
yaw_list[0] = np.array([15.,18.,0.])
tilt_list[0] = np.array([0.,0.,0.])
helix_amp_list[0] = np.array([0.,0.,0.])

# ------------------------------------


# other inputs
# os.chdir(r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation')
# filename = 'gaussian_overlap_.02_.02_128_512.nc'
# table_GaussOverlap = xr.load_dataarray(filename, engine='h5netcdf')

# model coefficients
sigma_0_D = 0.3042              # tuned
k_1 = 0.01213                   # tuned
k_2 = 0.008                     # tuned
mixing_gain_velocity = 0.2119   # tuned
awc_wake_exp = 1.119            # tuned
awc_wake_denominator = 137.21   # tuned
hcw_deflection_gain_D = 2.0984  # tuned
deflection_rate = 12.018        # tuned
mixing_gain_deflection = 0.     # NOT tuned yet


# function to minimize -------------------------------------------------------------------------------------------

# function of residual between wind fields cubed
def residuals_Ucube(coeff):
    
    # initialize resilduals
    residuals = np.array([])
    
    # iterate for each case
    for i in np.arange(len(u_mat_LES_list)):
        
        # extract case
        u_mat_LES = u_mat_LES_list[i]
        x_D_array = x_D_array_list[i]
        y_D_array = y_D_array_list[i]
        x = x_list[i]
        y = y_list[i]
        wd = wd_list[i]
        ws = ws_list[i]
        yaw = yaw_list[i]
        helix_amp = helix_amp_list[i]
    
        # define model with the input coefficients
        mixing_gain_deflection = coeff[0]
        wfm = PropagateDownwind_helix(site, wind_turbine,
                                      wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
                                                                                 sigma_0_D=sigma_0_D,
                                                                                 mixing_gain_velocity=mixing_gain_velocity,
                                                                                 awc_wake_exp=awc_wake_exp,
                                                                                 awc_wake_denominator=awc_wake_denominator),
                                      superpositionModel=SquaredSum(),
                                      deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
                                                                                  deflection_rate=deflection_rate,
                                                                                  mixing_gain_deflection=mixing_gain_deflection),
                                      turbulenceModel=None,
                                      rotorAvgModel=RotorCenter())
    
        # compute wind field with pywake
        u_mat_pywake = compute_flow(x_D_array = x_D_array,
                                    y_D_array = y_D_array,
                                    diameter = diameter,
                                    x = x,
                                    y = y,
                                    wfm = wfm,
                                    wd = wd,
                                    ws = ws,
                                    yaw = yaw,
                                    helix_amp = helix_amp)
    
        # compute residuals of U cubed
        residuals_temp = (np.abs(u_mat_LES**3-u_mat_pywake**3)).reshape(-1)
        residuals = np.concatenate((residuals,residuals_temp))
    
    return residuals


#%% optimize ------------------------------------------------------------------------------------------------------

# run optmization
coeff_0 = np.array([0.])
t = time.time()
res = least_squares(residuals_Ucube,coeff_0,verbose=2,bounds=(0,np.inf))
print(f'Optimization completed - Time: {time.time()-t}')    # around 1 min in this case
coeff_opt = res.x
mixing_gain_deflection_opt = coeff_opt[0]

print('Optimal coefficients:')
print(f'mixing_gain_deflection: \t {mixing_gain_deflection_opt}')


#%% plot ------------------------------------------------------------------------------------------------------


# wind farm model with OPTIMAL coefficients
wfm_opt = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[0.01213,0.008],
                                                                         sigma_0_D=0.3042,
                                                                         mixing_gain_velocity=0.2119,
                                                                         awc_wake_exp=1.119,
                                                                         awc_wake_denominator=137.21),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=2.0984,
                                                                          deflection_rate=12.018,
                                                                          mixing_gain_deflection=0.),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())


# wind farm model with INITIAL coefficients
wfm_initial = PropagateDownwind_helix(site, wind_turbine,
                              wake_deficitModel=EmpiricalGaussianDeficit(),
                              superpositionModel=SquaredSum(),
                              deflectionModel=EmpiricalGaussianDeflection(),
                              turbulenceModel=None,
                              rotorAvgModel=RotorCenter())


# plot comparison (3wt - helix=3deg)
ind_case = 0
name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_fig = r'_tuningDeflection_yawGeom3wt.svg'
dw_D_plot = np.array([7.,8.5,11.5,13.])
cw_D_plot = np.array([-0.75,0.,0.75])
wfm_list = [wfm_opt,wfm_initial]
wfm_label_list = ['Opt.','Init.']
plot_U_comparison(cw_D_plot,
                  dw_D_plot,
                  u_mat_LES = u_mat_LES_list[ind_case],
                  x_D_array_LES = x_D_array_list[ind_case],
                  y_D_array_LES = y_D_array_list[ind_case],
                  diameter = diameter,
                  x = x_list[ind_case],
                  y = y_list[ind_case],
                  wfm_list = wfm_list,
                  wfm_label_list = wfm_label_list,
                  wd = wd_list[ind_case],
                  ws = ws_list[ind_case],
                  yaw = yaw_list[ind_case],
                  helix_amp = helix_amp_list[ind_case],
                  savefig=True,
                  name_path=name_path,
                  name_fig=name_fig)






#%% COMPARE MODELS =========================================================================
# ==========================================================================================
# ==========================================================================================

# deifne site (HKN) - NOT RELEVANT FOR THE TUNING
wd_site = np.linspace(0,360,12,endpoint=False)
p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)

# define turbine
powerCtFunction = PowerCtFunction(
    input_keys=['ws','helix_amp'],
    power_ct_func = partial(helix_power_ct_function,
                            helix_a = 1.907,
                            helix_power_b = 1.376e-3,
                            helix_power_c = 4.017e-11,  # not tuned
                            helix_thrust_b = 0.8371e-3,
                            helix_thrust_c = 5.084e-4),  # not tuned
    power_unit='kW',
)
wind_turbine = WindTurbine(name='IEA22MW_helix',
                diameter=283.2,
                hub_height=170.0,
                powerCtFunction=powerCtFunction)    
diameter = wind_turbine.diameter()

# define wind farm model (EMPGAUSS - OPT COEFF.)
wfm_empgauss_opt = PropagateDownwind_helix(site, wind_turbine,
                                           wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[0.01213,0.008],
                                                                                      sigma_0_D=0.3042,
                                                                                      mixing_gain_velocity=0.2119,
                                                                                      awc_wake_exp=1.119,
                                                                                      awc_wake_denominator=137.21),
                                           superpositionModel=SquaredSum(),
                                           deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=2.0984,
                                                                          deflection_rate=12.018,
                                                                          mixing_gain_deflection=0.),
                                           turbulenceModel=None,
                                           rotorAvgModel=RotorCenter())


# define wind farm model (EMPGAUSS - INITIAL COEFF.)
wfm_empgauss_0 = PropagateDownwind_helix(site, wind_turbine,
                                         wake_deficitModel=EmpiricalGaussianDeficit(),
                                         superpositionModel=SquaredSum(),
                                         deflectionModel=EmpiricalGaussianDeflection(),
                                         turbulenceModel=None,
                                         rotorAvgModel=RotorCenter())





# define wind farm model (BASTANKAHAH and JIMENEZ)
ws_turbine = np.array([0,2,3,4,5,6,7,8,9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24,25])
p_baseline = np.array([0, 0,466, 1103, 2158, 3722, 5878, 8657, 12110, 16910, 22000, 22000, 22000, 22000, 22000, 22000, 22000, 22000, 22000, 22000, 22000,22000 ,22000 ,22000 ,22000])
ct_baseline = np.array([0, 0,0.8234, 0.8298, 0.8459, 0.8646, 0.8767, 0.8866, 0.8861, 0.8099, 0.7569, 0.45, 0.34, 0.27, 0.24, 0.2, 0.17, 0.13, 0.10, 0.09, 0.075,0.065, 0.06, 0.05, 0.046])
wind_turbine = WindTurbine(name='IEA22MW_helix',
                diameter=283.2,
                hub_height=170.0,
                powerCtFunction=PowerCtTabular(ws_turbine,p_baseline,'kW',ct_baseline))    
diameter = wind_turbine.diameter()
wfm_bast_1wt = PropagateDownwind(site, wind_turbine,
                             wake_deficitModel=BastankhahGaussianDeficit(),
                             superpositionModel=SquaredSum(),
                             deflectionModel=JimenezWakeDeflection(),
                             turbulenceModel=STF2017TurbulenceModel(),
                             rotorAvgModel=RotorCenter())




#%% plot 1 - 1wt simulation -----------------------------------------------------

from matplotlib.patches import Patch

x = np.array([0])*diameter
y = np.array([0])*diameter
wd = np.array([270])
ws = np.array([10])

# colors = ['#001221','#538de5','#41c3d3','#bee4dc','#ea9bd5','#ff9887']
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']

dw_D_plot = np.array([4.,6.,8.])
cw_D_plot = np.array([0.])

savefig = False
#name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_path = r"..\figures_WES_paper\\"
name_fig = r'_comparison_1wt_v2.pdf'


# stream section -----

#fig, axes = plt.subplots(len(dw_D_plot), figsize=(15, 11), sharex=True)
fig, axes = plt.subplots(len(dw_D_plot), figsize=(11,11), sharex=True)
for i in np.arange(len(dw_D_plot)):
    
    dw = dw_D_plot[i]
    axes[i].set_title(fr'Downstream distance: ${dw}\ D$')
    
    
    # baseline ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_baseline_1wt)
    dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
    u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
    axes[i].scatter(y_D_array_LES,u_LES_slice,c=colors[0],marker='o',s=5,label=r'Baseline',alpha=0.5)
    
    # py_wake
    yaw = np.array([0])
    helix_amp = np.array([0])
    y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(y_D_array_pywake,u_empgauss_slice,c=colors[0],label=r'Baseline',linestyle='-',linewidth=2)
    
    #u_bast_slice = compute_flow_Bastankhah(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    #axes[i].plot(y_D_array_pywake,u_bast_slice,c=colors[0],label='Baseline - Bastankhah',linestyle='--')
    
    
    # yaw=20deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_yaw20deg_1wt)
    dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
    u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
    axes[i].scatter(y_D_array_LES,u_LES_slice,c=colors[1],marker='o',s=5,label=r'Wake steering (+) ($\gamma=20^\circ$)',alpha=0.5)
    
    # py_wake
    yaw = np.array([20.])
    helix_amp = np.array([0])
    y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(y_D_array_pywake,u_empgauss_slice,c=colors[1],label=r'Wake steering (+) ($\gamma=20^\circ$)',linestyle='-',linewidth=2)
    
    #u_bast_slice = compute_flow_Bastankhah(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    #axes[i].plot(y_D_array_pywake,u_bast_slice,c=colors[1],label='Yaw: 20deg - Bastankhah',linestyle='--')
    
    
    # yaw=-20deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_yawmin20deg_1wt)
    dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
    u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
    axes[i].scatter(y_D_array_LES,u_LES_slice,c=colors[2],marker='o',s=5,label=r'Wake steering (-) ($\gamma=-20^\circ$)',alpha=0.5)
    
    # py_wake
    yaw = np.array([-20.])
    helix_amp = np.array([0])
    y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(y_D_array_pywake,u_empgauss_slice,c=colors[2],label=r'Wake steering (-) ($\gamma=-20^\circ$)',linestyle='-',linewidth=2)
    
    #u_bast_slice = compute_flow_Bastankhah(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    #axes[i].plot(y_D_array_pywake,u_bast_slice,c=colors[2],label='Yaw: -20deg - Bastankhah',linestyle='--')
    
    
    # helix=2deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_helix2deg_1wt)
    dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
    u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
    axes[i].scatter(y_D_array_LES,u_LES_slice,c=colors[3],marker='o',s=5,label=r'Helix A2 ($A=2^\circ$)',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.])
    helix_amp = np.array([2.])
    y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(y_D_array_pywake,u_empgauss_slice,c=colors[3],label=r'Helix A2 ($A=2^\circ$)',linestyle='-',linewidth=2)
    
    
    # helix=4deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_helix4deg_1wt)
    dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
    u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
    axes[i].scatter(y_D_array_LES,u_LES_slice,c=colors[4],marker='o',s=5,label=r'Helix A4 ($A=4^\circ$)',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.])
    helix_amp = np.array([4.])
    y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(y_D_array_pywake,u_empgauss_slice,c=colors[4],label=r'Helix A4 ($A=4^\circ$)',linestyle='-',linewidth=2)
        

    axes[i].set_xlim([-1.5,1.5])
    axes[i].set_ylim([4.5,10.5])
    axes[i].set_ylabel(r'Wind speed [$\mathrm{m\,s^{-1}}$]')

axes[len(dw_D_plot)-1].set_xlabel('Cross-stream distance [D]')


#axes[len(dw_D_plot)-2].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=1)
#plt.tight_layout()


handles_temp, labels_temp = axes[0].get_legend_handles_labels()
custom_handles = [Patch(color='none')]+handles_temp[0:10:2]+[Patch(color='none')]+[Patch(color='none')]+handles_temp[1:10:2]
custom_labels = [r'$\bf{LES}$']+labels_temp[0:10:2]+['']+[r'$\bf{Empirical\ Gaussian\ Model}$']+labels_temp[1:10:2]

fig.legend(
    handles=custom_handles,
    labels=custom_labels,
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    #loc='lower center',
    #bbox_to_anchor=(0.5, 1.1),
    ncol=1,
    frameon=False
)

plt.subplots_adjust(right=0.97)
#plt.tight_layout(rect=[0, 0, 1, 0.95])

if savefig: plt.savefig(name_path+'cw_section'+name_fig,format='pdf',bbox_inches='tight')
plt.show()

#%%
# cross section -----

fig, axes = plt.subplots(len(cw_D_plot), figsize=(15, 4), sharex=True)
for i in np.arange(len(cw_D_plot)):
    
    cw = cw_D_plot[i]
    axes.set_title(f'Crosswind distance: {cw} D')
    
    # baseline ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_baseline_1wt)
    cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
    u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
    axes.scatter(x_D_array_LES,u_LES_slice,c=colors[0],marker='*',s=50,label='Baseline - LES',alpha=0.5)
    
    # py_wake
    yaw = np.array([0])
    helix_amp = np.array([0])
    x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes.plot(x_D_array_pywake,u_empgauss_slice,c=colors[0],label='Baseline - EmpGauss',linestyle='-',linewidth=2)
    
    u_bast_slice = compute_flow_Bastankhah(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    axes.plot(x_D_array_pywake,u_bast_slice,c=colors[0],label='Baseline - Bastankhah',linestyle='--')


    # yaw=20deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_yaw20deg_1wt)
    cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
    u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
    axes.scatter(x_D_array_LES,u_LES_slice,c=colors[1],marker='*',s=50,label='Yaw: 20deg - LES',alpha=0.5)
    
    # py_wake
    yaw = np.array([20.])
    helix_amp = np.array([0])
    x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes.plot(x_D_array_pywake,u_empgauss_slice,c=colors[1],label='Yaw: 20deg - EmpGauss',linestyle='-',linewidth=2)
    
    u_bast_slice = compute_flow_Bastankhah(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    axes.plot(x_D_array_pywake,u_bast_slice,c=colors[1],label='Yaw: 20deg - Bastankhah',linestyle='--')
    
    
    # yaw=-20deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_yawmin20deg_1wt)
    cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
    u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
    axes.scatter(x_D_array_LES,u_LES_slice,c=colors[2],marker='*',s=50,label='Yaw: -20deg - LES',alpha=0.5)
    
    # py_wake
    yaw = np.array([-20.])
    helix_amp = np.array([0])
    x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes.plot(x_D_array_pywake,u_empgauss_slice,c=colors[2],label='Yaw: -20deg - EmpGauss',linestyle='-',linewidth=2)
    
    u_bast_slice = compute_flow_Bastankhah(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    axes.plot(x_D_array_pywake,u_bast_slice,c=colors[2],label='Yaw: -20deg - Bastankhah',linestyle='--')
    
    
    # helix=2deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_helix2deg_1wt)
    cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
    u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
    axes.scatter(x_D_array_LES,u_LES_slice,c=colors[3],marker='*',s=50,label='Helix: 2deg - LES',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.])
    helix_amp = np.array([2.])
    x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes.plot(x_D_array_pywake,u_empgauss_slice,c=colors[3],label='Helix: 2deg - EmpGauss',linestyle='-',linewidth=2)
    
    
    # helix=4deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_helix4deg_1wt)
    cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
    u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
    axes.scatter(x_D_array_LES,u_LES_slice,c=colors[4],marker='*',s=50,label='Helix: 4deg - LES',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.])
    helix_amp = np.array([4.])
    x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes.plot(x_D_array_pywake,u_empgauss_slice,c=colors[4],label='Helix: 4deg - EmpGauss',linestyle='-',linewidth=2)

        
    axes.set_ylabel('Wind speed [m/s]')

axes.set_xlabel('Downstream distance [D]')
axes.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=1)
plt.tight_layout()
if savefig: plt.savefig(name_path+'dw_section'+name_fig,format='svg')
plt.show()




#%% plot 2 - 3wt simulation -----------------------------------------------------

x = np.array([0.,4.5,9.])*diameter
y = np.array([0.,0.,0.])*diameter
wd = np.array([270])
ws = np.array([10])

# colors = ['#001221','#538de5','#41c3d3','#bee4dc','#ea9bd5','#ff9887']
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']

dw_D_plot = np.array([7.,8.5,11.5,13.])
cw_D_plot = np.array([-0.5,0.,0.5])

savefig = False
#name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_path = r"..\figures_WES_paper\\"
name_fig = r'_comparison_3wt_v2.pdf'


# stream section -----

fig, axes = plt.subplots(len(dw_D_plot), figsize=(11, 14), sharex=True)
for i in np.arange(len(dw_D_plot)):
    
    dw = dw_D_plot[i]
    axes[i].set_title(fr'Downstream distance: ${dw}\ D$')
    
    
    # baseline ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_baseline_3wt_aligned)
    dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
    u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
    axes[i].scatter(y_D_array_LES,u_LES_slice,c=colors[0],marker='o',s=5,label='Baseline array',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.,0.,0.])
    helix_amp = np.array([0.,0.,0.])
    y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(y_D_array_pywake,u_empgauss_slice,c=colors[0],label='Baseline array',linestyle='-',linewidth=2)
    
    #u_bast_slice = compute_flow_Bastankhah(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    #axes[i].plot(y_D_array_pywake,u_bast_slice,c=colors[0],label='Baseline - Bastankhah',linestyle='--')
    
    
    # yaw=YawGeom ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_yawGeom_3wt)
    dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
    u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
    axes[i].scatter(y_D_array_LES,u_LES_slice,c=colors[1],marker='o',s=5,label=r'Wake steering array ($\gamma=[15^\circ,18^\circ,0^\circ]$)',alpha=0.5)
    
    # py_wake
    yaw = np.array([15.,18.,0.])
    helix_amp = np.array([0.,0.,0.])
    y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(y_D_array_pywake,u_empgauss_slice,c=colors[1],label=r'Wake steering array ($\gamma_i=[15^\circ,18^\circ,0^\circ]$)',linestyle='-',linewidth=2)
    
    #u_bast_slice = compute_flow_Bastankhah(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    #axes[i].plot(y_D_array_pywake,u_bast_slice,c=colors[1],label='Yaw: [15,18,0]deg - Bastankhah',linestyle='--')
    
        
    # helix=3deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_helix3deg_3wt)
    dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
    u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
    axes[i].scatter(y_D_array_LES,u_LES_slice,c=colors[3],marker='o',s=5,label=r'Helix array A3 ($A_i=[3^\circ,0^\circ,0^\circ]$)',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.,0.,0.])
    helix_amp = np.array([3.,0.,0.])
    y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(y_D_array_pywake,u_empgauss_slice,c=colors[3],label=r'Helix array A3 ($A_i=[3^\circ,0^\circ,0^\circ]$)',linestyle='-',linewidth=2)
    
    
    # helix=4deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_helix4deg_3wt)
    dw_ind_LES = np.argmin(np.abs(x_D_array_LES-dw))
    u_LES_slice = u_mat_LES[:,dw_ind_LES].reshape(-1)
    axes[i].scatter(y_D_array_LES,u_LES_slice,c=colors[4],marker='o',s=5,label=r'Helix array A4 ($A_i=[4^\circ,0^\circ,0^\circ]$)',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.,0.,0.])
    helix_amp = np.array([4.,0.,0.])
    y_D_array_pywake = np.linspace(np.min(y_D_array_LES),np.max(y_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(np.array([dw]),y_D_array_pywake,diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(y_D_array_pywake,u_empgauss_slice,c=colors[4],label=r'Helix array A4 ($A_i=[4^\circ,0^\circ,0^\circ]$)',linestyle='-',linewidth=2)
        

    axes[i].set_xlim([-1.5,1.5])
    axes[i].set_ylim([2.5,10.5])
    axes[i].set_ylabel(r'Wind speed [$\mathrm{m\,s^{-1}}$]')

axes[len(dw_D_plot)-1].set_xlabel('Cross-stream distance [D]')

#axes[len(dw_D_plot)-2].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=1)
#plt.tight_layout()


handles_temp, labels_temp = axes[0].get_legend_handles_labels()
custom_handles = [Patch(color='none')]+handles_temp[0:8:2]+[Patch(color='none')]+[Patch(color='none')]+handles_temp[1:8:2]
custom_labels = [r'$\bf{LES}$']+labels_temp[0:8:2]+['']+[r'$\bf{Empirical\ Gaussian\ Model}$']+labels_temp[1:8:2]

fig.legend(
    handles=custom_handles,
    labels=custom_labels,
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    #loc='lower center',
    #bbox_to_anchor=(0.5, 1.1),
    ncol=1,
    frameon=False
)

plt.subplots_adjust(right=0.97)

if savefig: plt.savefig(name_path+'cw_section'+name_fig,format='pdf',bbox_inches='tight')
plt.show()

#%%
# cross section -----

fig, axes = plt.subplots(len(cw_D_plot), figsize=(15, 11), sharex=True)
for i in np.arange(len(cw_D_plot)):
    
    cw = cw_D_plot[i]
    axes[i].set_title(f'Crosswind distance: {cw} D')
    
    # baseline ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_baseline_3wt_aligned)
    cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
    u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
    axes[i].scatter(x_D_array_LES,u_LES_slice,c=colors[0],marker='*',s=50,label='Baseline - LES',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.,0.,0.,])
    helix_amp = np.array([0.,0.,0.])
    x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(x_D_array_pywake,u_empgauss_slice,c=colors[0],label='Baseline - EmpGauss',linestyle='-',linewidth=2)
    
    u_bast_slice = compute_flow_Bastankhah(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    axes[i].plot(x_D_array_pywake,u_bast_slice,c=colors[0],label='Baseline - Bastankhah',linestyle='--')


    # yaw=YawGeom ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_yawGeom_3wt)
    cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
    u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
    axes[i].scatter(x_D_array_LES,u_LES_slice,c=colors[1],marker='*',s=50,label='Yaw: [15,18,0]deg - LES',alpha=0.5)
    
    # py_wake
    yaw = np.array([15.,18.,0.])
    helix_amp = np.array([0.,0.,0.])
    x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(x_D_array_pywake,u_empgauss_slice,c=colors[1],label='Yaw: [15,18,0]deg - EmpGauss',linestyle='-',linewidth=2)
    
    u_bast_slice = compute_flow_Bastankhah(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_bast_1wt,wd,ws,yaw).reshape(-1)
    axes[i].plot(x_D_array_pywake,u_bast_slice,c=colors[1],label='Yaw: [15,18,0]deg - Bastankhah',linestyle='--')
        
    
    # helix=3deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_helix3deg_3wt)
    cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
    u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
    axes[i].scatter(x_D_array_LES,u_LES_slice,c=colors[3],marker='*',s=50,label='Helix: 3deg - LES',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.,0.,0.])
    helix_amp = np.array([3.,0.,0.])
    x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(x_D_array_pywake,u_empgauss_slice,c=colors[3],label='Helix: 3deg - EmpGauss',linestyle='-',linewidth=2)
    
    
    # helix=4deg ----------------------
    
    # LES
    u_mat_LES,x_D_array_LES,y_D_array_LES = extract_data_df(df_helix4deg_3wt)
    cw_ind_LES = np.argmin(np.abs(y_D_array_LES-cw))
    u_LES_slice = u_mat_LES[cw_ind_LES,:].reshape(-1)
    axes[i].scatter(x_D_array_LES,u_LES_slice,c=colors[4],marker='*',s=50,label='Helix: 4deg - LES',alpha=0.5)
    
    # py_wake
    yaw = np.array([0.,0.,0.])
    helix_amp = np.array([4.,0.,0.])
    x_D_array_pywake = np.linspace(np.min(x_D_array_LES),np.max(x_D_array_LES),500,endpoint=True)
    
    u_empgauss_slice = compute_flow(x_D_array_pywake,np.array([cw]),diameter,x,y,wfm_empgauss_opt,wd,ws,yaw,helix_amp).reshape(-1)
    axes[i].plot(x_D_array_pywake,u_empgauss_slice,c=colors[4],label='Helix: 4deg - EmpGauss',linestyle='-',linewidth=2)

    axes[i].set_ylim([0,10.5])
    axes[i].set_ylabel('Wind speed [m/s]')

axes[2].set_xlabel('Downstream distance [D]')
axes[1].legend(loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=1)
plt.tight_layout()
if savefig: plt.savefig(name_path+'dw_section'+name_fig,format='svg')
plt.show()
































#%% TEST FUNCTION: (convert_u_from_GaussOverlap_to_RotorCenter)

# os.chdir(r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation')
# filename = 'gaussian_overlap_.02_.02_128_512.nc'
# table_GaussOverlap = xr.load_dataarray(filename, engine='h5netcdf')


# x_P = 12.*diameter
# y_P = 0.*diameter

# wd = np.array([270])
# ws = np.array([10.])

# # intial farm
# x = np.array([0,4.5,9])*diameter
# y = np.array([0.,0.,0.])*diameter
# yaw = np.zeros(len(x))
# tilt = np.zeros(len(x))
# helix_amp = np.zeros(len(x))


# # coefficients EmpGauss

# sigma_0_D = 0.3042
# k_1 = 0.01213
# k_2 = 0.008

# mixing_gain_velocity = 2.

# awc_wake_exp = 1.2
# awc_wake_denominator = 400

# hcw_deflection_gain_D = 3.
# deflection_rate = 22.

# mixing_gain_deflection = 0.




# u_Linear = convert_u_from_GaussOverlap_to_RotorCenter(x_P,
#                                                y_P,
#                                                wd,
#                                                ws,
#                                                x,
#                                                y,
#                                                yaw,
#                                                tilt,
#                                                helix_amp,
#                                                sigma_0_D,
#                                                k_1,
#                                                k_2,
#                                                mixing_gain_velocity,
#                                                awc_wake_exp,
#                                                awc_wake_denominator,
#                                                hcw_deflection_gain_D,
#                                                deflection_rate,
#                                                mixing_gain_deflection,
#                                                table_GaussOverlap,
#                                                superimposition_model='Linear')

# u_SS = convert_u_from_GaussOverlap_to_RotorCenter(x_P,
#                                                y_P,
#                                                wd,
#                                                ws,
#                                                x,
#                                                y,
#                                                yaw,
#                                                tilt,
#                                                helix_amp,
#                                                sigma_0_D,
#                                                k_1,
#                                                k_2,
#                                                mixing_gain_velocity,
#                                                awc_wake_exp,
#                                                awc_wake_denominator,
#                                                hcw_deflection_gain_D,
#                                                deflection_rate,
#                                                mixing_gain_deflection,
#                                                table_GaussOverlap,
#                                                superimposition_model='SquaredSum')




# wfm_GaussOverlap_Linear = PropagateDownwind_helix(site, wind_turbine,
#                                           wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
#                                                                                      sigma_0_D=sigma_0_D,
#                                                                                      awc_wake_exp=awc_wake_exp,
#                                                                                      awc_wake_denominator=awc_wake_denominator),
#                                           superpositionModel=LinearSum(),
#                                           deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
#                                                                                       deflection_rate=deflection_rate,
#                                                                                       mixing_gain_deflection=mixing_gain_deflection),
#                                           turbulenceModel=None,
#                                           rotorAvgModel=GaussianOverlapAvgModel())

# wfm_RotorCenter_Linear = PropagateDownwind_helix(site, wind_turbine,
#                                           wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
#                                                                                      sigma_0_D=sigma_0_D,
#                                                                                      awc_wake_exp=awc_wake_exp,
#                                                                                      awc_wake_denominator=awc_wake_denominator),
#                                           superpositionModel=LinearSum(),
#                                           deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
#                                                                                       deflection_rate=deflection_rate,
#                                                                                       mixing_gain_deflection=mixing_gain_deflection),
#                                           turbulenceModel=None,
#                                           rotorAvgModel=RotorCenter())

# wfm_GaussOverlap_SS = PropagateDownwind_helix(site, wind_turbine,
#                                           wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
#                                                                                      sigma_0_D=sigma_0_D,
#                                                                                      awc_wake_exp=awc_wake_exp,
#                                                                                      awc_wake_denominator=awc_wake_denominator),
#                                           superpositionModel=SquaredSum(),
#                                           deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
#                                                                                       deflection_rate=deflection_rate,
#                                                                                       mixing_gain_deflection=mixing_gain_deflection),
#                                           turbulenceModel=None,
#                                           rotorAvgModel=GaussianOverlapAvgModel())

# wfm_RotorCenter_SS = PropagateDownwind_helix(site, wind_turbine,
#                                           wake_deficitModel=EmpiricalGaussianDeficit(wake_expansion_rates=[k_1,k_2],
#                                                                                      sigma_0_D=sigma_0_D,
#                                                                                      awc_wake_exp=awc_wake_exp,
#                                                                                      awc_wake_denominator=awc_wake_denominator),
#                                           superpositionModel=SquaredSum(),
#                                           deflectionModel=EmpiricalGaussianDeflection(hcw_deflection_gain_D=hcw_deflection_gain_D,
#                                                                                       deflection_rate=deflection_rate,
#                                                                                       mixing_gain_deflection=mixing_gain_deflection),
#                                           turbulenceModel=None,
#                                           rotorAvgModel=RotorCenter())



# x_temp = np.concatenate((x,np.array([x_P])))
# y_temp = np.concatenate((y,np.array([y_P])))
# yaw_temp = np.concatenate((yaw,np.array([0])))
# tilt_temp = np.concatenate((tilt,np.array([0])))
# helix_amp_temp = np.concatenate((helix_amp,np.array([0])))

# ws_eff_P_GaussOverlap_Linear = wfm_GaussOverlap_Linear(x_temp,y_temp,wd=wd,ws=ws,yaw=yaw_temp,tilt=tilt_temp,helix_amp=helix_amp_temp).WS_eff_ilk[-1].reshape(-2)
# ws_eff_P_RotorCenter_Linear = wfm_RotorCenter_Linear(x_temp,y_temp,wd=wd,ws=ws,yaw=yaw_temp,tilt=tilt_temp,helix_amp=helix_amp_temp).WS_eff_ilk[-1].reshape(-2)
# print('Linear ------------')
# print(f'GaussOverlap: Pywake: {ws_eff_P_GaussOverlap_Linear}')
# print(f'RotorCenter: Pywake: {ws_eff_P_RotorCenter_Linear} - Conversion: {u_Linear}')

# ws_eff_P_GaussOverlap_SS = wfm_GaussOverlap_SS(x_temp,y_temp,wd=wd,ws=ws,yaw=yaw_temp,tilt=tilt_temp,helix_amp=helix_amp_temp).WS_eff_ilk[-1].reshape(-2)
# ws_eff_P_RotorCenter_SS = wfm_RotorCenter_SS(x_temp,y_temp,wd=wd,ws=ws,yaw=yaw_temp,tilt=tilt_temp,helix_amp=helix_amp_temp).WS_eff_ilk[-1].reshape(-2)
# print('SquaredSum ------------')
# print(f'GaussOverlap: Pywake: {ws_eff_P_GaussOverlap_SS}')
# print(f'RotorCenter: Pywake: {ws_eff_P_RotorCenter_SS} - Conversion: {u_SS}')


