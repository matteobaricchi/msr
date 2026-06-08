#%%

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
from py_wake_helix.py_wake_helix_tools import Power_wrapper
from py_wake_helix.py_wake_helix_tools import compute_WFFC_LUT_generalObj


# import py_pywake models
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.site import UniformWeibullSite
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.superposition_models import SquaredSum
from py_wake.site import XRSite


#%%

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
    ws_array = np.arange(6,9,1)
    ws_min = np.min(ws_array)
    ws_max = np.max(ws_array)

    # uncertainty parameters
    apply_uncertainty = True
    sigma = 2.5
    n = 9

    # optimizer parameters
    weight_coefficient = 0.25
    weight = weight_coefficient*1e5
    helix_amp_max = 5


    # define objective fucntion (flow dipendent)    

    class Obj_flow_function():

        def __init__(self,**kwargs):
            self.x = kwargs.get('x')
            self.y = kwargs.get('y')
            self.wfm = kwargs.get('wfm')
            self.apply_uncertainty = kwargs.get('apply_uncertainty')
            self.sigma = kwargs.get('sigma')
            self.n = kwargs.get('n')
            self.weight = kwargs.get('weight')

        def __call__(self,wd,ws,yaw,helix_amp):
            
            power_wrapper = Power_wrapper(x = self.x,
                                          y = self.y,
                                          wfm = self.wfm,
                                          wd = wd,
                                          ws = ws,
                                          apply_uncertainty = self.apply_uncertainty,
                                          sigma = self.sigma,
                                          n = self.n)
            perc_control = np.zeros(len(self.x))
            fil_control = (np.abs(yaw)>0.)|(helix_amp>0.)
            perc_control[fil_control] = 1
            f_obj = power_wrapper(yaw,helix_amp) - self.weight*np.sum(perc_control)

            return f_obj
        
    obj_flow_function = Obj_flow_function(x = x,
                                          y = y,
                                          wfm = wfm,
                                          apply_uncertainty = apply_uncertainty,
                                          sigma = sigma,
                                          n = n,
                                          weight = weight)



    t_tot = time.time()

    # MIXED ============================================================================================================
    t = time.time()
    yaw_opt,helix_amp_opt = compute_WFFC_LUT_generalObj(x,
                                                        y,
                                                        wd_array,
                                                        ws_array,
                                                        obj_flow_function,
                                                        yaw_max = 30,
                                                        helix_amp_max = helix_amp_max,
                                                        optimize_yaw = True,
                                                        optimize_helix_amp = True,
                                                        tol = 1e-5,
                                                        parallel_execution = True,
                                                        n_cpu = 48)
    print(f'Mixed control - 2.5std - simulation completed: {time.time()-t}')
    with open(f'wffcLUT_HKN_mixed_sigma25_ws{ws_min}to{ws_max}_w{weight_coefficient}e5.pkl', 'wb') as f:
        pickle.dump({'wd_array' : wd_array,
                     'ws_array' : ws_array,
                     'yaw_opt': yaw_opt,
                     'helix_amp_opt' : helix_amp_opt
                    }, f)
    # ====================================================================================================================

    # YAW ============================================================================================================
    t = time.time()
    yaw_opt,helix_amp_opt = compute_WFFC_LUT_generalObj(x,
                                                        y,
                                                        wd_array,
                                                        ws_array,
                                                        obj_flow_function,
                                                        yaw_max = 30,
                                                        helix_amp_max = helix_amp_max,
                                                        optimize_yaw = True,
                                                        optimize_helix_amp = False,
                                                        tol = 1e-5,
                                                        parallel_execution = True,
                                                        n_cpu = 48)
    print(f'Yaw control - 2.5std - simulation completed: {time.time()-t}')
    with open(f'wffcLUT_HKN_yaw_sigma25_ws{ws_min}to{ws_max}_w{weight_coefficient}e5.pkl', 'wb') as f:
        pickle.dump({'wd_array' : wd_array,
                     'ws_array' : ws_array,
                     'yaw_opt': yaw_opt,
                     'helix_amp_opt' : helix_amp_opt
                    }, f)
    # ====================================================================================================================

    # HELIX ============================================================================================================
    t = time.time()
    yaw_opt,helix_amp_opt = compute_WFFC_LUT_generalObj(x,
                                                        y,
                                                        wd_array,
                                                        ws_array,
                                                        obj_flow_function,
                                                        yaw_max = 30,
                                                        helix_amp_max = helix_amp_max,
                                                        optimize_yaw = False,
                                                        optimize_helix_amp = True,
                                                        tol = 1e-5,
                                                        parallel_execution = True,
                                                        n_cpu = 48)
    print(f'Helix control - 2.5std - simulation completed: {time.time()-t}')
    with open(f'wffcLUT_HKN_helix_sigma25_ws{ws_min}to{ws_max}_w{weight_coefficient}e5.pkl', 'wb') as f:
        pickle.dump({'wd_array' : wd_array,
                     'ws_array' : ws_array,
                     'yaw_opt': yaw_opt,
                     'helix_amp_opt' : helix_amp_opt
                    }, f)
    # ====================================================================================================================

    print(f'Total time: {time.time()-t_tot}')

