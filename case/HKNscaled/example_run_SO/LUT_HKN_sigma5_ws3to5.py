# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:18:53 2025

@author: matteobaricchi
"""

import numpy as np
from functools import partial
import time
import pandas as pd
import matplotlib.pyplot as plt
import utm
import xarray as xr
import pickle


# import py_wake_helix models
from py_wake_helix.py_wake_helix import helix_power_ct_function
from py_wake_helix.py_wake_helix import PropagateDownwind_helix
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeficit
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeflection

# import py_wake_helix_tools models
from py_wake_helix.py_wake_helix_tools import calculateAEP_withUncertainty
from py_wake_helix.py_wake_helix_tools import calculatePower_withUncertainty
from py_wake_helix.py_wake_helix_tools import WFFC_Optimizer_SR
from py_wake_helix.py_wake_helix_tools import Power_wrapper
from py_wake_helix.py_wake_helix_tools import compute_WFFC_LUT
from py_wake_helix.py_wake_helix_tools import create_LUTdf
from py_wake_helix.py_wake_helix_tools import extract_LUTdf


# import py_pywake models
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.site import UniformWeibullSite
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.superposition_models import SquaredSum
from py_wake.site import XRSite



#%% TEST

if __name__ == "__main__":
    
    # extract HKN data
    with open(f'HKN_data_and_tools/HKN_data.pkl', 'rb') as f:
        HKN_data = pickle.load(f)
    hkn_site = HKN_data['hkn_site']
    hkn_ws_mean = HKN_data['hkn_ws_mean']
    hkn_site_bathymetry_grid = HKN_data['hkn_site_bathymetry_grid']
    hkn_site_x_grid = HKN_data['hkn_site_x_grid']
    hkn_site_y_grid = HKN_data['hkn_site_y_grid']
    hkn_boundaries_x = HKN_data['hkn_boundaries_x']
    hkn_boundaries_y = HKN_data['hkn_boundaries_y']
    hkn_wt_x = HKN_data['hkn_wt_x']
    hkn_wt_y = HKN_data['hkn_wt_y']

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



    # scale HKN data (turbine positions and wind resource)
    coord_sub = utm.from_latlon(52.70,4.29)
    x_sub = coord_sub[0]
    y_sub = coord_sub[1]
    diameter_hkn = 200.
    x = (hkn_wt_x-x_sub)*(diameter/diameter_hkn)
    y = (hkn_wt_y-y_sub)*(diameter/diameter_hkn)
    ds_hkn_scaled = xr.Dataset(
        data_vars={
            'Sector_frequency':(['x','y','wd'],hkn_site.ds['Sector_frequency'].values),
            'Weibull_A':(['x','y','wd'],hkn_site.ds['Weibull_A'].values),
            'Weibull_k':(['x','y','wd'],hkn_site.ds['Weibull_k'].values),
            'TI':0.04    
            },
        coords={
            'x':(hkn_site.ds['x'].values-x_sub)*(diameter/diameter_hkn),
            'y':(hkn_site.ds['y'].values-y_sub)*(diameter/diameter_hkn),
            'wd':hkn_site.ds['wd'].values
            }
        )
    hkn_site_scaled = XRSite(ds_hkn_scaled)


    # # deifne site (HKN)
    # wd_site = np.linspace(0,360,12,endpoint=False)
    # p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
    # a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
    # k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
    # site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)
    

    # define wind farm model (EMPGAUSS - OPT COEFF.)
    wfm = PropagateDownwind_helix(hkn_site_scaled, wind_turbine,
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
                                                rotorAvgModel=GaussianOverlapAvgModel())
    
    
    
    
    
    # define wind conditions
    wd_array = np.arange(0,360,1)
    ws_array = np.arange(3,6,1)

    # uncertainty parameters
    sigma = 5
    n = 17
    
    t_tot = time.time()

    # MIXED ============================================================================================================
    t = time.time()
    yaw_opt,helix_amp_opt =  compute_WFFC_LUT(x,
                                              y,
                                              wfm,
                                              wd_array,
                                              ws_array,
                                              optimize_yaw = True,
                                              optimize_helix_amp = True,
                                              apply_uncertainty = True,
                                              sigma = sigma,
                                              n = n,
                                              parallel_execution = True,
                                              n_cpu = 48)
    print(f'Mixed control - 5std - simulation completed: {time.time()-t}')
    df_yaw,df_helix_amp = create_LUTdf(wd_array,ws_array,yaw_opt,helix_amp_opt)
    df_yaw.to_csv('yawLUT_HKN_mixed_sigma5_ws3to5.csv',sep=',',index=False,encoding='utf-8')
    df_helix_amp.to_csv('helixLUT_HKN_mixed_sigma5_ws3to5.csv',sep=',',index=False,encoding='utf-8')
    # ====================================================================================================================


    # YAW ================================================================================================================
    t = time.time()
    yaw_opt,helix_amp_opt =  compute_WFFC_LUT(x,
                                              y,
                                              wfm,
                                              wd_array,
                                              ws_array,
                                              optimize_yaw = True,
                                              optimize_helix_amp = False,
                                              apply_uncertainty = True,
                                              sigma = sigma,
                                              n = n,
                                              parallel_execution = True,
                                              n_cpu = 48)
    print(f'Yaw control - 5std - simulation completed: {time.time()-t}')
    df_yaw,df_helix_amp = create_LUTdf(wd_array,ws_array,yaw_opt,helix_amp_opt)
    df_yaw.to_csv('yawLUT_HKN_yaw_sigma5_ws3to5.csv',sep=',',index=False,encoding='utf-8')
    df_helix_amp.to_csv('helixLUT_HKN_yaw_sigma5_ws3to5.csv',sep=',',index=False,encoding='utf-8')
    # ====================================================================================================================


    # HELIX ==============================================================================================================
    t = time.time()
    yaw_opt,helix_amp_opt =  compute_WFFC_LUT(x,
                                              y,
                                              wfm,
                                              wd_array,
                                              ws_array,
                                              optimize_yaw = False,
                                              optimize_helix_amp = True,
                                              apply_uncertainty = True,
                                              sigma = sigma,
                                              n = n,
                                              parallel_execution = True,
                                              n_cpu = 48)
    print(f'Helix control - 5std - simulation completed: {time.time()-t}')
    df_yaw,df_helix_amp = create_LUTdf(wd_array,ws_array,yaw_opt,helix_amp_opt)
    df_yaw.to_csv('yawLUT_HKN_helix_sigma5_ws3to5.csv',sep=',',index=False,encoding='utf-8')
    df_helix_amp.to_csv('helixLUT_HKN_helix_sigma5_ws3to5.csv',sep=',',index=False,encoding='utf-8')
    # ====================================================================================================================

    print(f'Total time: {time.time()-t_tot}')

