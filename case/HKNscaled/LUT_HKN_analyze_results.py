
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:41:30 2025

@author: matteobaricchi
"""
#%%
import numpy as np
from functools import partial
import time
import pandas as pd
import matplotlib.pyplot as plt
import utm
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
from numpy import newaxis as na
import pickle

# import py_wake_helix models
from py_wake_helix.py_wake_helix import helix_power_ct_function
from py_wake_helix.py_wake_helix import PropagateDownwind_helix
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeficit
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeflection

# import py_wake_helix_tools models
from py_wake_helix.py_wake_helix_tools import calculateAEP_withUncertainty
from py_wake_helix.py_wake_helix_tools import calculatePmat_withUncertainty
from py_wake_helix.py_wake_helix_tools import calculatePower_withUncertainty
from py_wake_helix.py_wake_helix_tools import WFFC_Optimizer_SR
from py_wake_helix.py_wake_helix_tools import Power_wrapper
from py_wake_helix.py_wake_helix_tools import compute_WFFC_LUT
from py_wake_helix.py_wake_helix_tools import create_LUTdf
from py_wake_helix.py_wake_helix_tools import extract_LUTdf


# import py_pywake models
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.site import UniformWeibullSite, XRSite
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.superposition_models import SquaredSum

from HKN_data_and_tools.hkn_data_extraction import extract_HKNsite


#%%


# extract HKN data
filename_site = r'HKN_data_and_tools/nonuniform_vortex_and_bathymetry_data_grid_980190969.csv'
filename_boundaries = r'HKN_data_and_tools/HKN_area.csv'
filename_layout = r'HKN_data_and_tools/layoutHKN.csv'
hkn_site,hkn_ws_mean,hkn_site_bathymetry_grid,hkn_site_x_grid,hkn_site_y_grid,hkn_boundaries_x,hkn_boundaries_y,hkn_wt_x,hkn_wt_y = extract_HKNsite(filename_site,filename_boundaries,filename_layout)

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
        'Weibull_A':(['x','y','wd'],hkn_site.ds['Weibull_A'].values*((170./115.)**0.1)),
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



#with open(f'layout_HKNscaled.pkl', 'wb') as f:
#    pickle.dump({'x_HKNscaled' : x_sub+(hkn_wt_x-x_sub)*(diameter/diameter_hkn),
#                 'y_HKNscaled' : y_sub+(hkn_wt_y-y_sub)*(diameter/diameter_hkn),
#                 }, f)




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






#%% EXTRACT DATA (around 40 min)

# sigma = 0 =================================================================================================================================================

t = time.time()

# mixed operation
df_yaw = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma0/yawLUT_HKN_mixed_sigma0_ws3to25.csv')
df_helix_amp = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma0/helixLUT_HKN_mixed_sigma0_ws3to25.csv')
yaw_mixedOpt_0std,helix_amp_mixedOpt_0std,wd_array,ws_array = extract_LUTdf(df_yaw,df_helix_amp)
simres_mixedOpt_0std = wfm(x,y,wd=wd_array,ws=ws_array,yaw=yaw_mixedOpt_0std,tilt=0,helix_amp=helix_amp_mixedOpt_0std)

# wake steering
df_yaw = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma0/yawLUT_HKN_yaw_sigma0_ws3to25.csv')
df_helix_amp = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma0/helixLUT_HKN_yaw_sigma0_ws3to25.csv')
yaw_yawOpt_0std,helix_amp_yawOpt_0std,wd_array,ws_array = extract_LUTdf(df_yaw,df_helix_amp)
simres_yawOpt_0std = wfm(x,y,wd=wd_array,ws=ws_array,yaw=yaw_yawOpt_0std,tilt=0,helix_amp=helix_amp_yawOpt_0std)

# helix
df_yaw = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma0/yawLUT_HKN_helix_sigma0_ws3to25.csv')
df_helix_amp = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma0/helixLUT_HKN_helix_sigma0_ws3to25.csv')
yaw_helixOpt_0std,helix_amp_helixOpt_0std,wd_array,ws_array = extract_LUTdf(df_yaw,df_helix_amp)
simres_helixOpt_0std = wfm(x,y,wd=wd_array,ws=ws_array,yaw=yaw_helixOpt_0std,tilt=0,helix_amp=helix_amp_helixOpt_0std)

# baseline
simres_baseline_0std = wfm(x,y,wd=wd_array,ws=ws_array,yaw=np.zeros_like(yaw_helixOpt_0std),tilt=0,helix_amp=np.zeros_like(helix_amp_helixOpt_0std))

print(f'Data extraction sigma 0 completed - Time: {time.time()-t}')



# sigma = 2.5 =================================================================================================================================================

t = time.time()

sigma = 2.5
n_values = 9
ws_array = np.arange(3,26,1)
wd_array = np.arange(0,360,1)
fil_ws = np.ones((len(x),len(wd_array),len(ws_array)),dtype=bool)
fil_ws[:,:,12:] = False
fil_ws_temp = np.ones((len(x),len(wd_array),12),dtype=bool)

# mixed operation
df_yaw_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_mixed_sigma25_ws3to5.csv')
df_yaw_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_mixed_sigma25_ws6to8.csv')
df_yaw_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_mixed_sigma25_ws9to11.csv')
df_yaw_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_mixed_sigma25_ws12to14.csv')
df_helix_amp_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_mixed_sigma25_ws3to5.csv')
df_helix_amp_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_mixed_sigma25_ws6to8.csv')
df_helix_amp_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_mixed_sigma25_ws9to11.csv')
df_helix_amp_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_mixed_sigma25_ws12to14.csv')
df_yaw = pd.concat([df_yaw_1,df_yaw_2,df_yaw_3,df_yaw_4],ignore_index=True)
df_helix_amp = pd.concat([df_helix_amp_1,df_helix_amp_2,df_helix_amp_3,df_helix_amp_4],ignore_index=True)
yaw_mixedOpt_25std_temp,helix_amp_mixedOpt_25std_temp,wd_array,ws_array_temp = extract_LUTdf(df_yaw,df_helix_amp)
yaw_mixedOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_mixedOpt_25std[fil_ws] = yaw_mixedOpt_25std_temp[fil_ws_temp]
helix_amp_mixedOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_mixedOpt_25std[fil_ws] = helix_amp_mixedOpt_25std_temp[fil_ws_temp]
p_mat_mixedOpt_25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_25std,helix_amp_mixedOpt_25std,sigma=sigma,n=n_values)

# wake steering
df_yaw_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_yaw_sigma25_ws3to5.csv')
df_yaw_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_yaw_sigma25_ws6to8.csv')
df_yaw_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_yaw_sigma25_ws9to11.csv')
df_yaw_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_yaw_sigma25_ws12to14.csv')
df_helix_amp_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_yaw_sigma25_ws3to5.csv')
df_helix_amp_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_yaw_sigma25_ws6to8.csv')
df_helix_amp_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_yaw_sigma25_ws9to11.csv')
df_helix_amp_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_yaw_sigma25_ws12to14.csv')
df_yaw = pd.concat([df_yaw_1,df_yaw_2,df_yaw_3,df_yaw_4],ignore_index=True)
df_helix_amp = pd.concat([df_helix_amp_1,df_helix_amp_2,df_helix_amp_3,df_helix_amp_4],ignore_index=True)
yaw_yawOpt_25std_temp,helix_amp_yawOpt_25std_temp,wd_array,ws_array_temp = extract_LUTdf(df_yaw,df_helix_amp)
yaw_yawOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_yawOpt_25std[fil_ws] = yaw_yawOpt_25std_temp[fil_ws_temp]
helix_amp_yawOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_yawOpt_25std[fil_ws] = helix_amp_yawOpt_25std_temp[fil_ws_temp]
p_mat_yawOpt_25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_25std,helix_amp_yawOpt_25std,sigma=sigma,n=n_values)

# helix
df_yaw_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_helix_sigma25_ws3to5.csv')
df_yaw_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_helix_sigma25_ws6to8.csv')
df_yaw_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_helix_sigma25_ws9to11.csv')
df_yaw_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/yawLUT_HKN_helix_sigma25_ws12to14.csv')
df_helix_amp_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_helix_sigma25_ws3to5.csv')
df_helix_amp_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_helix_sigma25_ws6to8.csv')
df_helix_amp_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_helix_sigma25_ws9to11.csv')
df_helix_amp_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma25/helixLUT_HKN_helix_sigma25_ws12to14.csv')
df_yaw = pd.concat([df_yaw_1,df_yaw_2,df_yaw_3,df_yaw_4],ignore_index=True)
df_helix_amp = pd.concat([df_helix_amp_1,df_helix_amp_2,df_helix_amp_3,df_helix_amp_4],ignore_index=True)
yaw_helixOpt_25std_temp,helix_amp_helixOpt_25std_temp,wd_array,ws_array_temp = extract_LUTdf(df_yaw,df_helix_amp)
yaw_helixOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_helixOpt_25std[fil_ws] = yaw_helixOpt_25std_temp[fil_ws_temp]
helix_amp_helixOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_helixOpt_25std[fil_ws] = helix_amp_helixOpt_25std_temp[fil_ws_temp]
p_mat_helixOpt_25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_25std,helix_amp_helixOpt_25std,sigma=sigma,n=n_values)

# baseline
p_mat_baseline_25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,np.zeros_like(yaw_mixedOpt_25std),np.zeros_like(helix_amp_mixedOpt_25std),sigma=sigma,n=n_values)


print(f'Data extraction sigma 2.5 completed - Time: {time.time()-t}')


# sigma = 5 =================================================================================================================================================

t = time.time()

sigma = 5
n_values = 9
ws_array = np.arange(3,26,1)
wd_array = np.arange(0,360,1)
fil_ws = np.ones((len(x),len(wd_array),len(ws_array)),dtype=bool)
fil_ws[:,:,12:] = False
fil_ws_temp = np.ones((len(x),len(wd_array),12),dtype=bool)

# mixed operation
df_yaw_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_mixed_sigma5_ws3to5.csv')
df_yaw_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_mixed_sigma5_ws6to8.csv')
df_yaw_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_mixed_sigma5_ws9to11.csv')
df_yaw_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_mixed_sigma5_ws12to14.csv')
df_helix_amp_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_mixed_sigma5_ws3to5.csv')
df_helix_amp_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_mixed_sigma5_ws6to8.csv')
df_helix_amp_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_mixed_sigma5_ws9to11.csv')
df_helix_amp_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_mixed_sigma5_ws12to14.csv')
df_yaw = pd.concat([df_yaw_1,df_yaw_2,df_yaw_3,df_yaw_4],ignore_index=True)
df_helix_amp = pd.concat([df_helix_amp_1,df_helix_amp_2,df_helix_amp_3,df_helix_amp_4],ignore_index=True)
yaw_mixedOpt_5std_temp,helix_amp_mixedOpt_5std_temp,wd_array,ws_array_temp = extract_LUTdf(df_yaw,df_helix_amp)
yaw_mixedOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_mixedOpt_5std[fil_ws] = yaw_mixedOpt_5std_temp[fil_ws_temp]
helix_amp_mixedOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_mixedOpt_5std[fil_ws] = helix_amp_mixedOpt_5std_temp[fil_ws_temp]
p_mat_mixedOpt_5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_5std,helix_amp_mixedOpt_5std,sigma=sigma,n=n_values)

# wake steering
df_yaw_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_yaw_sigma5_ws3to5.csv')
df_yaw_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_yaw_sigma5_ws6to8.csv')
df_yaw_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_yaw_sigma5_ws9to11.csv')
df_yaw_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_yaw_sigma5_ws12to14.csv')
df_helix_amp_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_yaw_sigma5_ws3to5.csv')
df_helix_amp_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_yaw_sigma5_ws6to8.csv')
df_helix_amp_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_yaw_sigma5_ws9to11.csv')
df_helix_amp_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_yaw_sigma5_ws12to14.csv')
df_yaw = pd.concat([df_yaw_1,df_yaw_2,df_yaw_3,df_yaw_4],ignore_index=True)
df_helix_amp = pd.concat([df_helix_amp_1,df_helix_amp_2,df_helix_amp_3,df_helix_amp_4],ignore_index=True)
yaw_yawOpt_5std_temp,helix_amp_yawOpt_5std_temp,wd_array,ws_array_temp = extract_LUTdf(df_yaw,df_helix_amp)
yaw_yawOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_yawOpt_5std[fil_ws] = yaw_yawOpt_5std_temp[fil_ws_temp]
helix_amp_yawOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_yawOpt_5std[fil_ws] = helix_amp_yawOpt_5std_temp[fil_ws_temp]
p_mat_yawOpt_5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_5std,helix_amp_yawOpt_5std,sigma=sigma,n=n_values)

# helix
df_yaw_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_helix_sigma5_ws3to5.csv')
df_yaw_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_helix_sigma5_ws6to8.csv')
df_yaw_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_helix_sigma5_ws9to11.csv')
df_yaw_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/yawLUT_HKN_helix_sigma5_ws12to14.csv')
df_helix_amp_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_helix_sigma5_ws3to5.csv')
df_helix_amp_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_helix_sigma5_ws6to8.csv')
df_helix_amp_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_helix_sigma5_ws9to11.csv')
df_helix_amp_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v3/data/sigma5/helixLUT_HKN_helix_sigma5_ws12to14.csv')
df_yaw = pd.concat([df_yaw_1,df_yaw_2,df_yaw_3,df_yaw_4],ignore_index=True)
df_helix_amp = pd.concat([df_helix_amp_1,df_helix_amp_2,df_helix_amp_3,df_helix_amp_4],ignore_index=True)
yaw_helixOpt_5std_temp,helix_amp_helixOpt_5std_temp,wd_array,ws_array_temp = extract_LUTdf(df_yaw,df_helix_amp)
yaw_helixOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_helixOpt_5std[fil_ws] = yaw_helixOpt_5std_temp[fil_ws_temp]
helix_amp_helixOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_helixOpt_5std[fil_ws] = helix_amp_helixOpt_5std_temp[fil_ws_temp]
p_mat_helixOpt_5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_5std,helix_amp_helixOpt_5std,sigma=sigma,n=n_values)

# baseline
p_mat_baseline_5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,np.zeros_like(yaw_mixedOpt_5std),np.zeros_like(helix_amp_mixedOpt_5std),sigma=sigma,n=n_values)

print(f'Data extraction sigma 5 completed - Time: {time.time()-t}')


#%%
# SAVE DATA (in pickle file)

import pickle

with open(f'LUT_HKN.pkl', 'wb') as f:
    pickle.dump({'yaw_mixedOpt_0std' : yaw_mixedOpt_0std,
                 'helix_amp_mixedOpt_0std' : helix_amp_mixedOpt_0std,
                 'p_mat_mixedOpt_0std' : simres_mixedOpt_0std.power_ilk,
                 'simres_mixedOpt_0std' : simres_mixedOpt_0std,
                 'yaw_yawOpt_0std' : yaw_yawOpt_0std,
                 'helix_amp_yawOpt_0std' : helix_amp_yawOpt_0std,
                 'p_mat_yawOpt_0std' : simres_yawOpt_0std.power_ilk,
                 'simres_yawOpt_0std' : simres_yawOpt_0std,
                 'yaw_helixOpt_0std' : yaw_helixOpt_0std,
                 'helix_amp_helixOpt_0std' : helix_amp_helixOpt_0std,
                 'p_mat_helixOpt_0std' : simres_helixOpt_0std.power_ilk,
                 'simres_helixOpt_0std' : simres_helixOpt_0std,
                 'p_mat_baseline_0std' : simres_baseline_0std.power_ilk,
                 'simres_baseline_0std' : simres_baseline_0std,
                 'yaw_mixedOpt_25std' : yaw_mixedOpt_25std,
                 'helix_amp_mixedOpt_25std' : helix_amp_mixedOpt_25std,
                 'p_mat_mixedOpt_25std' : p_mat_mixedOpt_25std,
                 'yaw_yawOpt_25std' : yaw_yawOpt_25std,
                 'helix_amp_yawOpt_25std' : helix_amp_yawOpt_25std,
                 'p_mat_yawOpt_25std' : p_mat_yawOpt_25std,
                 'yaw_helixOpt_25std' : yaw_helixOpt_25std,
                 'helix_amp_helixOpt_25std' : helix_amp_helixOpt_25std,
                 'p_mat_helixOpt_25std' : p_mat_helixOpt_25std,
                 'p_mat_baseline_25std' : p_mat_baseline_25std,
                 'yaw_mixedOpt_5std' : yaw_mixedOpt_5std,
                 'helix_amp_mixedOpt_5std' : helix_amp_mixedOpt_5std,
                 'p_mat_mixedOpt_5std' : p_mat_mixedOpt_5std,
                 'yaw_yawOpt_5std' : yaw_yawOpt_5std,
                 'helix_amp_yawOpt_5std' : helix_amp_yawOpt_5std,
                 'p_mat_yawOpt_5std' : p_mat_yawOpt_5std,
                 'yaw_helixOpt_5std' : yaw_helixOpt_5std,
                 'helix_amp_helixOpt_5std' : helix_amp_helixOpt_5std,
                 'p_mat_helixOpt_5std' : p_mat_helixOpt_5std,
                 'p_mat_baseline_5std' : p_mat_baseline_5std,
                 }, f)


#%% EXTRACT DATA (FAST) -- OLD

with open(f'LUT_HKN.pkl', 'rb') as f:
    data = pickle.load(f)

yaw_mixedOpt_0std = data['yaw_mixedOpt_0std']
helix_amp_mixedOpt_0std = data['helix_amp_mixedOpt_0std']
p_mat_mixedOpt_0std = data['p_mat_mixedOpt_0std']
simres_mixedOpt_0std = data['simres_mixedOpt_0std']
yaw_yawOpt_0std = data['yaw_yawOpt_0std'] 
helix_amp_yawOpt_0std = data['helix_amp_yawOpt_0std']
p_mat_yawOpt_0std = data['p_mat_yawOpt_0std']
simres_yawOpt_0std = data['simres_yawOpt_0std']
yaw_helixOpt_0std = data['yaw_helixOpt_0std']
helix_amp_helixOpt_0std = data['helix_amp_helixOpt_0std']
p_mat_helixOpt_0std = data['p_mat_helixOpt_0std']
simres_helixOpt_0std = data['simres_helixOpt_0std']
p_mat_baseline_0std = data['p_mat_baseline_0std']
simres_baseline_0std = data['simres_baseline_0std']
yaw_mixedOpt_25std = data['yaw_mixedOpt_25std']
helix_amp_mixedOpt_25std = data['helix_amp_mixedOpt_25std']
p_mat_mixedOpt_25std = data['p_mat_mixedOpt_25std']
yaw_yawOpt_25std = data['yaw_yawOpt_25std']
helix_amp_yawOpt_25std = data['helix_amp_yawOpt_25std']
p_mat_yawOpt_25std = data['p_mat_yawOpt_25std']
yaw_helixOpt_25std = data['yaw_helixOpt_25std']
helix_amp_helixOpt_25std = data['helix_amp_helixOpt_25std']
p_mat_helixOpt_25std = data['p_mat_helixOpt_25std']
p_mat_baseline_25std = data['p_mat_baseline_25std']
yaw_mixedOpt_5std = data['yaw_mixedOpt_5std']
helix_amp_mixedOpt_5std = data['helix_amp_mixedOpt_5std']
p_mat_mixedOpt_5std = data['p_mat_mixedOpt_5std']
yaw_yawOpt_5std = data['yaw_yawOpt_5std']
helix_amp_yawOpt_5std = data['helix_amp_yawOpt_5std']
p_mat_yawOpt_5std = data['p_mat_yawOpt_5std']
yaw_helixOpt_5std = data['yaw_helixOpt_5std']
helix_amp_helixOpt_5std = data['helix_amp_helixOpt_5std']
p_mat_helixOpt_5std = data['p_mat_helixOpt_5std']
p_mat_baseline_5std = data['p_mat_baseline_5std']



#%% UPDATE ONLY sigma=5 obtained with N=17

# sigma = 5 =================================================================================================================================================

t = time.time()

sigma = 5
n_values = 17
ws_array = np.arange(3,26,1)
wd_array = np.arange(0,360,1)
fil_ws = np.ones((len(x),len(wd_array),len(ws_array)),dtype=bool)
fil_ws[:,:,12:] = False
fil_ws_temp = np.ones((len(x),len(wd_array),12),dtype=bool)

# mixed operation
df_yaw_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_mixed_sigma5_ws3to5.csv')
df_yaw_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_mixed_sigma5_ws6to8.csv')
df_yaw_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_mixed_sigma5_ws9to11.csv')
df_yaw_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_mixed_sigma5_ws12to14.csv')
df_helix_amp_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_mixed_sigma5_ws3to5.csv')
df_helix_amp_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_mixed_sigma5_ws6to8.csv')
df_helix_amp_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_mixed_sigma5_ws9to11.csv')
df_helix_amp_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_mixed_sigma5_ws12to14.csv')
df_yaw = pd.concat([df_yaw_1,df_yaw_2,df_yaw_3,df_yaw_4],ignore_index=True)
df_helix_amp = pd.concat([df_helix_amp_1,df_helix_amp_2,df_helix_amp_3,df_helix_amp_4],ignore_index=True)
yaw_mixedOpt_5std_temp,helix_amp_mixedOpt_5std_temp,wd_array,ws_array_temp = extract_LUTdf(df_yaw,df_helix_amp)
yaw_mixedOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_mixedOpt_5std[fil_ws] = yaw_mixedOpt_5std_temp[fil_ws_temp]
helix_amp_mixedOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_mixedOpt_5std[fil_ws] = helix_amp_mixedOpt_5std_temp[fil_ws_temp]
p_mat_mixedOpt_5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_5std,helix_amp_mixedOpt_5std,sigma=sigma,n=n_values)

# wake steering
df_yaw_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_yaw_sigma5_ws3to5.csv')
df_yaw_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_yaw_sigma5_ws6to8.csv')
df_yaw_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_yaw_sigma5_ws9to11.csv')
df_yaw_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_yaw_sigma5_ws12to14.csv')
df_helix_amp_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_yaw_sigma5_ws3to5.csv')
df_helix_amp_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_yaw_sigma5_ws6to8.csv')
df_helix_amp_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_yaw_sigma5_ws9to11.csv')
df_helix_amp_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_yaw_sigma5_ws12to14.csv')
df_yaw = pd.concat([df_yaw_1,df_yaw_2,df_yaw_3,df_yaw_4],ignore_index=True)
df_helix_amp = pd.concat([df_helix_amp_1,df_helix_amp_2,df_helix_amp_3,df_helix_amp_4],ignore_index=True)
yaw_yawOpt_5std_temp,helix_amp_yawOpt_5std_temp,wd_array,ws_array_temp = extract_LUTdf(df_yaw,df_helix_amp)
yaw_yawOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_yawOpt_5std[fil_ws] = yaw_yawOpt_5std_temp[fil_ws_temp]
helix_amp_yawOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_yawOpt_5std[fil_ws] = helix_amp_yawOpt_5std_temp[fil_ws_temp]
p_mat_yawOpt_5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_5std,helix_amp_yawOpt_5std,sigma=sigma,n=n_values)

# helix
df_yaw_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_helix_sigma5_ws3to5.csv')
df_yaw_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_helix_sigma5_ws6to8.csv')
df_yaw_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_helix_sigma5_ws9to11.csv')
df_yaw_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/yawLUT_HKN_helix_sigma5_ws12to14.csv')
df_helix_amp_1 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_helix_sigma5_ws3to5.csv')
df_helix_amp_2 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_helix_sigma5_ws6to8.csv')
df_helix_amp_3 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_helix_sigma5_ws9to11.csv')
df_helix_amp_4 = pd.read_csv('LUT_HKN_DelftBlue_simulations/LUT_HKN_v5/data/sigma5/helixLUT_HKN_helix_sigma5_ws12to14.csv')
df_yaw = pd.concat([df_yaw_1,df_yaw_2,df_yaw_3,df_yaw_4],ignore_index=True)
df_helix_amp = pd.concat([df_helix_amp_1,df_helix_amp_2,df_helix_amp_3,df_helix_amp_4],ignore_index=True)
yaw_helixOpt_5std_temp,helix_amp_helixOpt_5std_temp,wd_array,ws_array_temp = extract_LUTdf(df_yaw,df_helix_amp)
yaw_helixOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_helixOpt_5std[fil_ws] = yaw_helixOpt_5std_temp[fil_ws_temp]
helix_amp_helixOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_helixOpt_5std[fil_ws] = helix_amp_helixOpt_5std_temp[fil_ws_temp]
p_mat_helixOpt_5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_5std,helix_amp_helixOpt_5std,sigma=sigma,n=n_values)

# baseline
p_mat_baseline_5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,np.zeros_like(yaw_mixedOpt_5std),np.zeros_like(helix_amp_mixedOpt_5std),sigma=sigma,n=n_values)

print(f'Data extraction sigma 5 completed - Time: {time.time()-t}')


# save data
with open(f'LUT_HKN_v2.pkl', 'wb') as f:
    pickle.dump({'yaw_mixedOpt_0std' : yaw_mixedOpt_0std,
                 'helix_amp_mixedOpt_0std' : helix_amp_mixedOpt_0std,
                 'p_mat_mixedOpt_0std' : simres_mixedOpt_0std.power_ilk,
                 'simres_mixedOpt_0std' : simres_mixedOpt_0std,
                 'yaw_yawOpt_0std' : yaw_yawOpt_0std,
                 'helix_amp_yawOpt_0std' : helix_amp_yawOpt_0std,
                 'p_mat_yawOpt_0std' : simres_yawOpt_0std.power_ilk,
                 'simres_yawOpt_0std' : simres_yawOpt_0std,
                 'yaw_helixOpt_0std' : yaw_helixOpt_0std,
                 'helix_amp_helixOpt_0std' : helix_amp_helixOpt_0std,
                 'p_mat_helixOpt_0std' : simres_helixOpt_0std.power_ilk,
                 'simres_helixOpt_0std' : simres_helixOpt_0std,
                 'p_mat_baseline_0std' : simres_baseline_0std.power_ilk,
                 'simres_baseline_0std' : simres_baseline_0std,

                 'yaw_mixedOpt_25std' : yaw_mixedOpt_25std,
                 'helix_amp_mixedOpt_25std' : helix_amp_mixedOpt_25std,
                 'p_mat_mixedOpt_25std' : p_mat_mixedOpt_25std,
                 'yaw_yawOpt_25std' : yaw_yawOpt_25std,
                 'helix_amp_yawOpt_25std' : helix_amp_yawOpt_25std,
                 'p_mat_yawOpt_25std' : p_mat_yawOpt_25std,
                 'yaw_helixOpt_25std' : yaw_helixOpt_25std,
                 'helix_amp_helixOpt_25std' : helix_amp_helixOpt_25std,
                 'p_mat_helixOpt_25std' : p_mat_helixOpt_25std,
                 'p_mat_baseline_25std' : p_mat_baseline_25std,

                 'yaw_mixedOpt_5std' : yaw_mixedOpt_5std,
                 'helix_amp_mixedOpt_5std' : helix_amp_mixedOpt_5std,
                 'p_mat_mixedOpt_5std' : p_mat_mixedOpt_5std,
                 'yaw_yawOpt_5std' : yaw_yawOpt_5std,
                 'helix_amp_yawOpt_5std' : helix_amp_yawOpt_5std,
                 'p_mat_yawOpt_5std' : p_mat_yawOpt_5std,
                 'yaw_helixOpt_5std' : yaw_helixOpt_5std,
                 'helix_amp_helixOpt_5std' : helix_amp_helixOpt_5std,
                 'p_mat_helixOpt_5std' : p_mat_helixOpt_5std,
                 'p_mat_baseline_5std' : p_mat_baseline_5std,
                 }, f)


#%% EXTRACT DATA (FAST)

with open(f'LUT_HKN_v2.pkl', 'rb') as f:
    data = pickle.load(f)

yaw_mixedOpt_0std = data['yaw_mixedOpt_0std']
helix_amp_mixedOpt_0std = data['helix_amp_mixedOpt_0std']
p_mat_mixedOpt_0std = data['p_mat_mixedOpt_0std']
simres_mixedOpt_0std = data['simres_mixedOpt_0std']
yaw_yawOpt_0std = data['yaw_yawOpt_0std'] 
helix_amp_yawOpt_0std = data['helix_amp_yawOpt_0std']
p_mat_yawOpt_0std = data['p_mat_yawOpt_0std']
simres_yawOpt_0std = data['simres_yawOpt_0std']
yaw_helixOpt_0std = data['yaw_helixOpt_0std']
helix_amp_helixOpt_0std = data['helix_amp_helixOpt_0std']
p_mat_helixOpt_0std = data['p_mat_helixOpt_0std']
simres_helixOpt_0std = data['simres_helixOpt_0std']
p_mat_baseline_0std = data['p_mat_baseline_0std']
simres_baseline_0std = data['simres_baseline_0std']
yaw_mixedOpt_25std = data['yaw_mixedOpt_25std']
helix_amp_mixedOpt_25std = data['helix_amp_mixedOpt_25std']
p_mat_mixedOpt_25std = data['p_mat_mixedOpt_25std']
yaw_yawOpt_25std = data['yaw_yawOpt_25std']
helix_amp_yawOpt_25std = data['helix_amp_yawOpt_25std']
p_mat_yawOpt_25std = data['p_mat_yawOpt_25std']
yaw_helixOpt_25std = data['yaw_helixOpt_25std']
helix_amp_helixOpt_25std = data['helix_amp_helixOpt_25std']
p_mat_helixOpt_25std = data['p_mat_helixOpt_25std']
p_mat_baseline_25std = data['p_mat_baseline_25std']
yaw_mixedOpt_5std = data['yaw_mixedOpt_5std']
helix_amp_mixedOpt_5std = data['helix_amp_mixedOpt_5std']
p_mat_mixedOpt_5std = data['p_mat_mixedOpt_5std']
yaw_yawOpt_5std = data['yaw_yawOpt_5std']
helix_amp_yawOpt_5std = data['helix_amp_yawOpt_5std']
p_mat_yawOpt_5std = data['p_mat_yawOpt_5std']
yaw_helixOpt_5std = data['yaw_helixOpt_5std']
helix_amp_helixOpt_5std = data['helix_amp_helixOpt_5std']
p_mat_helixOpt_5std = data['p_mat_helixOpt_5std']
p_mat_baseline_5std = data['p_mat_baseline_5std']




#%% PLOT POWER GAINS for different wind directions


# define wind conditions
wd_array = np.arange(0,360,1)
ws_array = np.arange(3,26,1)

savefig = False
name_path = r'figures\LUT_HKN\\'
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']



# calculate power gain - sigma=0 =====================================================================================================

p_baseline_0std_lk = np.sum(simres_baseline_0std.Power.values,axis=(0))

p_mixedOpt_0std_lk = np.sum(simres_mixedOpt_0std.Power.values,axis=(0))
p_yawOpt_0std_lk = np.sum(simres_yawOpt_0std.Power.values,axis=(0))
p_helixOpt_0std_lk = np.sum(simres_helixOpt_0std.Power.values,axis=(0))

p_gain_mixedOpt_0std_lk = 100*(p_mixedOpt_0std_lk-p_baseline_0std_lk)/p_baseline_0std_lk
p_gain_yawOpt_0std_lk = 100*(p_yawOpt_0std_lk-p_baseline_0std_lk)/p_baseline_0std_lk
p_gain_helixOpt_0std_lk = 100*(p_helixOpt_0std_lk-p_baseline_0std_lk)/p_baseline_0std_lk


# calculate power gain - sigma=2.5 =====================================================================================================

p_baseline_25std_lk = np.sum(p_mat_baseline_25std,axis=(0))

p_mixedOpt_25std_lk = np.sum(p_mat_mixedOpt_25std,axis=(0))
p_yawOpt_25std_lk = np.sum(p_mat_yawOpt_25std,axis=(0))
p_helixOpt_25std_lk = np.sum(p_mat_helixOpt_25std,axis=(0))

p_gain_mixedOpt_25std_lk = 100*(p_mixedOpt_25std_lk-p_baseline_25std_lk)/p_baseline_25std_lk
p_gain_yawOpt_25std_lk = 100*(p_yawOpt_25std_lk-p_baseline_25std_lk)/p_baseline_25std_lk
p_gain_helixOpt_25std_lk = 100*(p_helixOpt_25std_lk-p_baseline_25std_lk)/p_baseline_25std_lk


# calculate power gain - sigma=5 =====================================================================================================

p_baseline_5std_lk = np.sum(p_mat_baseline_5std,axis=(0))

p_mixedOpt_5std_lk = np.sum(p_mat_mixedOpt_5std,axis=(0))
p_yawOpt_5std_lk = np.sum(p_mat_yawOpt_5std,axis=(0))
p_helixOpt_5std_lk = np.sum(p_mat_helixOpt_5std,axis=(0))

p_gain_mixedOpt_5std_lk = 100*(p_mixedOpt_5std_lk-p_baseline_5std_lk)/p_baseline_5std_lk
p_gain_yawOpt_5std_lk = 100*(p_yawOpt_5std_lk-p_baseline_5std_lk)/p_baseline_5std_lk
p_gain_helixOpt_5std_lk = 100*(p_helixOpt_5std_lk-p_baseline_5std_lk)/p_baseline_5std_lk


# plot ===============================================

ws_ind = 5

fig,axs = plt.subplots(3,figsize=(15, 8), sharex=True)

#axs[0].set_title(f'No uncertainty - Wind speed: {ws_array[ws_ind]} m/s')
axs[0].set_title(r'Wind direction uncertainty: $\sigma_{\theta}=0^\circ$')
axs[0].plot(wd_array,p_gain_mixedOpt_0std_lk[:,ws_ind],label='Combined',c=colors[0])
axs[0].plot(wd_array,p_gain_yawOpt_0std_lk[:,ws_ind],label='Wake steering',c=colors[2])
axs[0].plot(wd_array,p_gain_helixOpt_0std_lk[:,ws_ind],label='Helix',c=colors[4])
axs[0].legend()
axs[0].set_ylabel('Power gain [%]')
axs[0].set_xlim([0,359])

#axs[1].set_title(f'Uncertainty (sigma=2.5) - Wind speed: {ws_array[ws_ind]} m/s')
axs[1].set_title(r'Wind direction uncertainty: $\sigma_{\theta}=2.5^\circ$')
axs[1].plot(wd_array,p_gain_mixedOpt_25std_lk[:,ws_ind],label='Combined',c=colors[0])
axs[1].plot(wd_array,p_gain_yawOpt_25std_lk[:,ws_ind],label='Wake steering',c=colors[2])
axs[1].plot(wd_array,p_gain_helixOpt_25std_lk[:,ws_ind],label='Helix',c=colors[4])
axs[1].legend()
axs[1].set_ylabel('Power gain [%]')
axs[1].set_xlim([0,359])

#axs[2].set_title(f'Uncertainty (sigma=5) - Wind speed: {ws_array[ws_ind]} m/s')
axs[2].set_title(r'Wind direction uncertainty: $\sigma_{\theta}=5^\circ$')
axs[2].plot(wd_array,p_gain_mixedOpt_5std_lk[:,ws_ind],label='Combined',c=colors[0])
axs[2].plot(wd_array,p_gain_yawOpt_5std_lk[:,ws_ind],label='Wake steering',c=colors[2])
axs[2].plot(wd_array,p_gain_helixOpt_5std_lk[:,ws_ind],label='Helix',c=colors[4])
axs[2].legend()
axs[2].set_ylabel('Power gain [%]')
axs[2].set_xlim([0,359])

axs[2].set_xlabel('Wind direction [deg]')
if savefig: plt.savefig(name_path+'power_gain_wd_v3.pdf',format='pdf',bbox_inches='tight')
plt.show()




#%% PLOT POWER GAINS for different wind speeds


# define wind conditions
wd_array = np.arange(0,360,1)
ws_array = np.arange(3,26,1)

savefig = False
name_path = r'figures\LUT_HKN\\'
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']


# calculate power gain - sigma=0 =====================================================================================================

p_baseline_0std_k = np.sum(p_baseline_0std_lk,axis=(0))

p_mixedOpt_0std_k = np.sum(p_mixedOpt_0std_lk,axis=(0))
p_yawOpt_0std_k = np.sum(p_yawOpt_0std_lk,axis=(0))
p_helixOpt_0std_k = np.sum(p_helixOpt_0std_lk,axis=(0))

p_gain_mixedOpt_0std_k = 100*(p_mixedOpt_0std_k-p_baseline_0std_k)/p_baseline_0std_k
p_gain_yawOpt_0std_k = 100*(p_yawOpt_0std_k-p_baseline_0std_k)/p_baseline_0std_k
p_gain_helixOpt_0std_k = 100*(p_helixOpt_0std_k-p_baseline_0std_k)/p_baseline_0std_k


# calculate power gain - sigma=2.5 =====================================================================================================

p_baseline_25std_k = np.sum(p_baseline_25std_lk,axis=(0))

p_mixedOpt_25std_k = np.sum(p_mixedOpt_25std_lk,axis=(0))
p_yawOpt_25std_k = np.sum(p_yawOpt_25std_lk,axis=(0))
p_helixOpt_25std_k = np.sum(p_helixOpt_25std_lk,axis=(0))

p_gain_mixedOpt_25std_k = 100*(p_mixedOpt_25std_k-p_baseline_25std_k)/p_baseline_25std_k
p_gain_yawOpt_25std_k = 100*(p_yawOpt_25std_k-p_baseline_25std_k)/p_baseline_25std_k
p_gain_helixOpt_25std_k = 100*(p_helixOpt_25std_k-p_baseline_25std_k)/p_baseline_25std_k


# calculate power gain - sigma=5 =====================================================================================================

p_baseline_5std_k = np.sum(p_baseline_5std_lk,axis=(0))

p_mixedOpt_5std_k = np.sum(p_mixedOpt_5std_lk,axis=(0))
p_yawOpt_5std_k = np.sum(p_yawOpt_5std_lk,axis=(0))
p_helixOpt_5std_k = np.sum(p_helixOpt_5std_lk,axis=(0))

p_gain_mixedOpt_5std_k = 100*(p_mixedOpt_5std_k-p_baseline_5std_k)/p_baseline_5std_k
p_gain_yawOpt_5std_k = 100*(p_yawOpt_5std_k-p_baseline_5std_k)/p_baseline_5std_k
p_gain_helixOpt_5std_k = 100*(p_helixOpt_5std_k-p_baseline_5std_k)/p_baseline_5std_k


fig,axs = plt.subplots(3,figsize=(6, 9), sharex=True)

axs[0].set_title(r'Wind direction uncertainty: $\sigma_{\theta}=0^\circ$')
axs[0].plot(ws_array,p_gain_mixedOpt_0std_k,label='Combined',c=colors[0])
axs[0].plot(ws_array,p_gain_yawOpt_0std_k,label='Wake steering',c=colors[2])
axs[0].plot(ws_array,p_gain_helixOpt_0std_k,label='Helix',c=colors[4])
axs[0].legend()
axs[0].set_ylabel('Power gain [%]')

axs[1].set_title(r'Wind direction uncertainty: $\sigma_{\theta}=2.5^\circ$')
axs[1].plot(ws_array,p_gain_mixedOpt_25std_k,label='Combined',c=colors[0])
axs[1].plot(ws_array,p_gain_yawOpt_25std_k,label='Wake steering',c=colors[2])
axs[1].plot(ws_array,p_gain_helixOpt_25std_k,label='Helix',c=colors[4])
axs[1].legend()
axs[1].set_ylabel('Power gain [%]')

axs[2].set_title(r'Wind direction uncertainty: $\sigma_{\theta}=5^\circ$')
axs[2].plot(ws_array,p_gain_mixedOpt_5std_k,label='Combined',c=colors[0])
axs[2].plot(ws_array,p_gain_yawOpt_5std_k,label='Wake steering',c=colors[2])
axs[2].plot(ws_array,p_gain_helixOpt_5std_k,label='Helix',c=colors[4])
axs[2].legend()
axs[2].set_ylabel('Power gain [%]')

axs[2].set_xlabel('Wind speed [m/s]')
if savefig: plt.savefig(name_path+'power_gain_ws.pdf',format='pdf')
plt.show()





#%% CALCULATE AEP GAINS

# extract porbability of flow cases (per turbine)
#prob_mat = simres_baseline_0std.P.values

## calculate AEP gain - sigma=0 =====================================================================================================
#
#aep_baseline_0std = np.sum(simres_baseline_0std.aep().values)
#
#aep_mixedOpt_0std = np.sum(simres_mixedOpt_0std.aep().values)
#aep_yawOpt_0std = np.sum(simres_yawOpt_0std.aep().values)
#aep_helixOpt_0std = np.sum(simres_helixOpt_0std.aep().values)
#
#aep_gain_mixedOpt_0std = 100*(aep_mixedOpt_0std-aep_baseline_0std)/aep_baseline_0std
#aep_gain_yawOpt_0std = 100*(aep_yawOpt_0std-aep_baseline_0std)/aep_baseline_0std
#aep_gain_helixOpt_0std = 100*(aep_helixOpt_0std-aep_baseline_0std)/aep_baseline_0std

simres_baseline_0std = wfm(x,y,wd=wd_array,ws=ws_array,yaw=np.zeros((len(x),len(wd_array),len(ws_array))),helix_amp=np.zeros((len(x),len(wd_array),len(ws_array))),tilt=0)
prob_mat = simres_baseline_0std.P.values

# calculate AEP gain - sigma=0 =====================================================================================================

aep_baseline_0std = 8760*np.sum(p_mat_baseline_0std*prob_mat)/1e9

aep_mixedOpt_0std = 8760*np.sum(p_mat_mixedOpt_0std*prob_mat)/1e9
aep_yawOpt_0std = 8760*np.sum(p_mat_yawOpt_0std*prob_mat)/1e9
aep_helixOpt_0std = 8760*np.sum(p_mat_helixOpt_0std*prob_mat)/1e9

aep_gain_mixedOpt_0std = 100*(aep_mixedOpt_0std-aep_baseline_0std)/aep_baseline_0std
aep_gain_yawOpt_0std = 100*(aep_yawOpt_0std-aep_baseline_0std)/aep_baseline_0std
aep_gain_helixOpt_0std = 100*(aep_helixOpt_0std-aep_baseline_0std)/aep_baseline_0std

# calculate AEP gain - sigma=2.5 =====================================================================================================

aep_baseline_25std = 8760*np.sum(p_mat_baseline_25std*prob_mat)/1e9

aep_mixedOpt_25std = 8760*np.sum(p_mat_mixedOpt_25std*prob_mat)/1e9
aep_yawOpt_25std = 8760*np.sum(p_mat_yawOpt_25std*prob_mat)/1e9
aep_helixOpt_25std = 8760*np.sum(p_mat_helixOpt_25std*prob_mat)/1e9

aep_gain_mixedOpt_25std = 100*(aep_mixedOpt_25std-aep_baseline_25std)/aep_baseline_25std
aep_gain_yawOpt_25std = 100*(aep_yawOpt_25std-aep_baseline_25std)/aep_baseline_25std
aep_gain_helixOpt_25std = 100*(aep_helixOpt_25std-aep_baseline_25std)/aep_baseline_25std


# calculate AEP gain - sigma=5 =====================================================================================================

aep_baseline_5std = 8760*np.sum(p_mat_baseline_5std*prob_mat)/1e9

aep_mixedOpt_5std = 8760*np.sum(p_mat_mixedOpt_5std*prob_mat)/1e9
aep_yawOpt_5std = 8760*np.sum(p_mat_yawOpt_5std*prob_mat)/1e9
aep_helixOpt_5std = 8760*np.sum(p_mat_helixOpt_5std*prob_mat)/1e9

aep_gain_mixedOpt_5std = 100*(aep_mixedOpt_5std-aep_baseline_5std)/aep_baseline_5std
aep_gain_yawOpt_5std = 100*(aep_yawOpt_5std-aep_baseline_5std)/aep_baseline_5std
aep_gain_helixOpt_5std = 100*(aep_helixOpt_5std-aep_baseline_5std)/aep_baseline_5std


#%% PLOT AEP GAINS

# plot ==========================================

savefig = False
#name_path = r'figures\LUT_HKN\\'
name_path = r'figures\WES_review\\'
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']

xlabel_list = [r'$\sigma_{\theta}=0^\circ$',r'$\sigma_{\theta}=2.5^\circ$',r'$\sigma_{\theta}=5^\circ$']
aep_gain_mixedOpt_array = np.array([aep_gain_mixedOpt_0std,aep_gain_mixedOpt_25std,aep_gain_mixedOpt_5std])
aep_gain_yawOpt_array = np.array([aep_gain_yawOpt_0std,aep_gain_yawOpt_25std,aep_gain_yawOpt_5std])
aep_gain_helixOpt_array = np.array([aep_gain_helixOpt_0std,aep_gain_helixOpt_25std,aep_gain_helixOpt_5std])

bar_width = 0.2
x_plot = np.arange(len(xlabel_list))
offsets = np.array([-bar_width, 0, bar_width])

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(x_plot + offsets[0], aep_gain_mixedOpt_array, width=bar_width, color=colors[0], label='Combined')
ax.bar(x_plot + offsets[1], aep_gain_yawOpt_array, width=bar_width, color=colors[2], label='Wake steering')
ax.bar(x_plot + offsets[2], aep_gain_helixOpt_array, width=bar_width, color=colors[4], label='Helix')

ax.set_xticks(x_plot)
ax.set_xticklabels(xlabel_list)
ax.set_ylabel('AEP gain [%]')
ax.legend()

if savefig: plt.savefig(name_path+'aep_gains_v2.pdf',format='pdf')
plt.show()


print(aep_gain_mixedOpt_array)
print(aep_gain_yawOpt_array)
print(aep_gain_helixOpt_array)




#%% WES review - robustness of uncertainty knowledge - create data (1.5 h)

# evaluation with sigma=0 ---------------

t = time.time()
sigma = 0.
n_values = 1

# mixed
p_mat_mixedOpt_0std_eval0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_0std,helix_amp_mixedOpt_0std,sigma=sigma,n=n_values)
p_mat_mixedOpt_25std_eval0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_25std,helix_amp_mixedOpt_25std,sigma=sigma,n=n_values)
p_mat_mixedOpt_5std_eval0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_5std,helix_amp_mixedOpt_5std,sigma=sigma,n=n_values)
aep_gain_mixedOpt_0std_eval0std = 100*((8760*np.sum(p_mat_mixedOpt_0std_eval0std*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std
aep_gain_mixedOpt_25std_eval0std = 100*((8760*np.sum(p_mat_mixedOpt_25std_eval0std*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std
aep_gain_mixedOpt_5std_eval0std = 100*((8760*np.sum(p_mat_mixedOpt_5std_eval0std*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std

# wake steering
p_mat_yawOpt_0std_eval0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_0std,helix_amp_yawOpt_0std,sigma=sigma,n=n_values)
p_mat_yawOpt_25std_eval0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_25std,helix_amp_yawOpt_25std,sigma=sigma,n=n_values)
p_mat_yawOpt_5std_eval0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_5std,helix_amp_yawOpt_5std,sigma=sigma,n=n_values)
aep_gain_yawOpt_0std_eval0std = 100*((8760*np.sum(p_mat_yawOpt_0std_eval0std*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std
aep_gain_yawOpt_25std_eval0std = 100*((8760*np.sum(p_mat_yawOpt_25std_eval0std*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std
aep_gain_yawOpt_5std_eval0std = 100*((8760*np.sum(p_mat_yawOpt_5std_eval0std*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std

# helix
p_mat_helixOpt_0std_eval0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_0std,helix_amp_helixOpt_0std,sigma=sigma,n=n_values)
p_mat_helixOpt_25std_eval0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_25std,helix_amp_helixOpt_25std,sigma=sigma,n=n_values)
p_mat_helixOpt_5std_eval0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_5std,helix_amp_helixOpt_5std,sigma=sigma,n=n_values)
aep_gain_helixOpt_0std_eval0std = 100*((8760*np.sum(p_mat_helixOpt_0std_eval0std*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std
aep_gain_helixOpt_25std_eval0std = 100*((8760*np.sum(p_mat_helixOpt_25std_eval0std*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std
aep_gain_helixOpt_5std_eval0std = 100*((8760*np.sum(p_mat_helixOpt_5std_eval0std*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std

print(f'Sigma=0 completed - time: {time.time()-t}')


# evaluation with sigma=2.5 ---------------

t = time.time()
sigma = 2.5
n_values = 9

# mixed
p_mat_mixedOpt_0std_eval25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_0std,helix_amp_mixedOpt_0std,sigma=sigma,n=n_values)
p_mat_mixedOpt_25std_eval25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_25std,helix_amp_mixedOpt_25std,sigma=sigma,n=n_values)
p_mat_mixedOpt_5std_eval25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_5std,helix_amp_mixedOpt_5std,sigma=sigma,n=n_values)
aep_gain_mixedOpt_0std_eval25std = 100*((8760*np.sum(p_mat_mixedOpt_0std_eval25std*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std
aep_gain_mixedOpt_25std_eval25std = 100*((8760*np.sum(p_mat_mixedOpt_25std_eval25std*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std
aep_gain_mixedOpt_5std_eval25std = 100*((8760*np.sum(p_mat_mixedOpt_5std_eval25std*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std

# wake steering
p_mat_yawOpt_0std_eval25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_0std,helix_amp_yawOpt_0std,sigma=sigma,n=n_values)
p_mat_yawOpt_25std_eval25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_25std,helix_amp_yawOpt_25std,sigma=sigma,n=n_values)
p_mat_yawOpt_5std_eval25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_5std,helix_amp_yawOpt_5std,sigma=sigma,n=n_values)
aep_gain_yawOpt_0std_eval25std = 100*((8760*np.sum(p_mat_yawOpt_0std_eval25std*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std
aep_gain_yawOpt_25std_eval25std = 100*((8760*np.sum(p_mat_yawOpt_25std_eval25std*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std
aep_gain_yawOpt_5std_eval25std = 100*((8760*np.sum(p_mat_yawOpt_5std_eval25std*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std

# helix
p_mat_helixOpt_0std_eval25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_0std,helix_amp_helixOpt_0std,sigma=sigma,n=n_values)
p_mat_helixOpt_25std_eval25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_25std,helix_amp_helixOpt_25std,sigma=sigma,n=n_values)
p_mat_helixOpt_5std_eval25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_5std,helix_amp_helixOpt_5std,sigma=sigma,n=n_values)
aep_gain_helixOpt_0std_eval25std = 100*((8760*np.sum(p_mat_helixOpt_0std_eval25std*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std
aep_gain_helixOpt_25std_eval25std = 100*((8760*np.sum(p_mat_helixOpt_25std_eval25std*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std
aep_gain_helixOpt_5std_eval25std = 100*((8760*np.sum(p_mat_helixOpt_5std_eval25std*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std

print(f'Sigma=2.5 completed - time: {time.time()-t}')


# evaluation with sigma=5 ---------------

t = time.time()
sigma = 5.
n_values = 17

# mixed
p_mat_mixedOpt_0std_eval5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_0std,helix_amp_mixedOpt_0std,sigma=sigma,n=n_values)
p_mat_mixedOpt_25std_eval5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_25std,helix_amp_mixedOpt_25std,sigma=sigma,n=n_values)
p_mat_mixedOpt_5std_eval5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_5std,helix_amp_mixedOpt_5std,sigma=sigma,n=n_values)
aep_gain_mixedOpt_0std_eval5std = 100*((8760*np.sum(p_mat_mixedOpt_0std_eval5std*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std
aep_gain_mixedOpt_25std_eval5std = 100*((8760*np.sum(p_mat_mixedOpt_25std_eval5std*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std
aep_gain_mixedOpt_5std_eval5std = 100*((8760*np.sum(p_mat_mixedOpt_5std_eval5std*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std

# wake steering
p_mat_yawOpt_0std_eval5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_0std,helix_amp_yawOpt_0std,sigma=sigma,n=n_values)
p_mat_yawOpt_25std_eval5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_25std,helix_amp_yawOpt_25std,sigma=sigma,n=n_values)
p_mat_yawOpt_5std_eval5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_yawOpt_5std,helix_amp_yawOpt_5std,sigma=sigma,n=n_values)
aep_gain_yawOpt_0std_eval5std = 100*((8760*np.sum(p_mat_yawOpt_0std_eval5std*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std
aep_gain_yawOpt_25std_eval5std = 100*((8760*np.sum(p_mat_yawOpt_25std_eval5std*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std
aep_gain_yawOpt_5std_eval5std = 100*((8760*np.sum(p_mat_yawOpt_5std_eval5std*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std

# helix
p_mat_helixOpt_0std_eval5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_0std,helix_amp_helixOpt_0std,sigma=sigma,n=n_values)
p_mat_helixOpt_25std_eval5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_25std,helix_amp_helixOpt_25std,sigma=sigma,n=n_values)
p_mat_helixOpt_5std_eval5std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_5std,helix_amp_helixOpt_5std,sigma=sigma,n=n_values)
aep_gain_helixOpt_0std_eval5std = 100*((8760*np.sum(p_mat_helixOpt_0std_eval5std*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std
aep_gain_helixOpt_25std_eval5std = 100*((8760*np.sum(p_mat_helixOpt_25std_eval5std*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std
aep_gain_helixOpt_5std_eval5std = 100*((8760*np.sum(p_mat_helixOpt_5std_eval5std*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std

print(f'Sigma=5 completed - time: {time.time()-t}')


# save data ---------------

sigma_eval_array = np.array([0,2.5,5])
sigma_opt_array = np.array([0,2.5,5])

aep_gain_mat_mixedOpt = np.zeros((len(sigma_eval_array),len(sigma_opt_array)))
aep_gain_mat_mixedOpt[0,0] = aep_gain_mixedOpt_0std_eval0std
aep_gain_mat_mixedOpt[0,1] = aep_gain_mixedOpt_25std_eval0std
aep_gain_mat_mixedOpt[0,2] = aep_gain_mixedOpt_5std_eval0std
aep_gain_mat_mixedOpt[1,0] = aep_gain_mixedOpt_0std_eval25std
aep_gain_mat_mixedOpt[1,1] = aep_gain_mixedOpt_25std_eval25std
aep_gain_mat_mixedOpt[1,2] = aep_gain_mixedOpt_5std_eval25std
aep_gain_mat_mixedOpt[2,0] = aep_gain_mixedOpt_0std_eval5std
aep_gain_mat_mixedOpt[2,1] = aep_gain_mixedOpt_25std_eval5std
aep_gain_mat_mixedOpt[2,2] = aep_gain_mixedOpt_5std_eval5std

aep_gain_mat_yawOpt = np.zeros((len(sigma_eval_array),len(sigma_opt_array)))
aep_gain_mat_yawOpt[0,0] = aep_gain_yawOpt_0std_eval0std
aep_gain_mat_yawOpt[0,1] = aep_gain_yawOpt_25std_eval0std
aep_gain_mat_yawOpt[0,2] = aep_gain_yawOpt_5std_eval0std
aep_gain_mat_yawOpt[1,0] = aep_gain_yawOpt_0std_eval25std
aep_gain_mat_yawOpt[1,1] = aep_gain_yawOpt_25std_eval25std
aep_gain_mat_yawOpt[1,2] = aep_gain_yawOpt_5std_eval25std
aep_gain_mat_yawOpt[2,0] = aep_gain_yawOpt_0std_eval5std
aep_gain_mat_yawOpt[2,1] = aep_gain_yawOpt_25std_eval5std
aep_gain_mat_yawOpt[2,2] = aep_gain_yawOpt_5std_eval5std

aep_gain_mat_helixOpt = np.zeros((len(sigma_eval_array),len(sigma_opt_array)))
aep_gain_mat_helixOpt[0,0] = aep_gain_helixOpt_0std_eval0std
aep_gain_mat_helixOpt[0,1] = aep_gain_helixOpt_25std_eval0std
aep_gain_mat_helixOpt[0,2] = aep_gain_helixOpt_5std_eval0std
aep_gain_mat_helixOpt[1,0] = aep_gain_helixOpt_0std_eval25std
aep_gain_mat_helixOpt[1,1] = aep_gain_helixOpt_25std_eval25std
aep_gain_mat_helixOpt[1,2] = aep_gain_helixOpt_5std_eval25std
aep_gain_mat_helixOpt[2,0] = aep_gain_helixOpt_0std_eval5std
aep_gain_mat_helixOpt[2,1] = aep_gain_helixOpt_25std_eval5std
aep_gain_mat_helixOpt[2,2] = aep_gain_helixOpt_5std_eval5std

with open(f'AEPgains_robustness_v2.pkl', 'wb') as f:
    pickle.dump({'sigma_eval_array' : sigma_eval_array,
                 'sigma_opt_array' : sigma_opt_array,
                 'aep_gain_mat_mixedOpt' : aep_gain_mat_mixedOpt,
                 'aep_gain_mat_yawOpt' : aep_gain_mat_yawOpt,
                 'aep_gain_mat_helixOpt' : aep_gain_mat_helixOpt,
                 }, f)




#%% WES review - robustness of uncertainty knowledge - plot data

savefig = False
name_path = r'figures\WES_review\\'
format_fig = 'svg'

# extract data
with open(f'AEPgains_robustness_v2.pkl', 'rb') as f:
    data = pickle.load(f)

sigma_eval_array = data['sigma_eval_array']
sigma_opt_array = data['sigma_opt_array']
aep_gain_mat_mixedOpt = data['aep_gain_mat_mixedOpt']
aep_gain_mat_yawOpt = data['aep_gain_mat_yawOpt']
aep_gain_mat_helixOpt = data['aep_gain_mat_helixOpt']


from matplotlib.colors import TwoSlopeNorm
fig, axs = plt.subplots(figsize=(10,3), nrows=1, ncols=3, sharey=True)
mats = [
    (aep_gain_mat_yawOpt, "Wake steering"),
    (aep_gain_mat_helixOpt, "Helix"),
    (aep_gain_mat_mixedOpt, "Combined")
]
max_val = np.maximum(-min(mat.min() for mat, _ in mats),max(mat.max() for mat, _ in mats))
norm = TwoSlopeNorm(vmin=-max_val*1.5,vcenter=0,vmax=max_val*1.5)
for ax, (mat, title) in zip(axs, mats):
    im = ax.imshow(mat, cmap='PuOr', norm=norm)
    ax.set_xticks(np.arange(len(sigma_opt_array)))
    ax.set_yticks(np.arange(len(sigma_eval_array)))
    ax.set_xticklabels(sigma_opt_array)
    ax.set_yticklabels(sigma_eval_array)
    ax.set_xlabel(r'Optimization $\sigma_\theta$')
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",color="black")
axs[0].set_ylabel(r'Evaluation $\sigma_\theta$')
cbar = fig.colorbar(im,ax=axs,location='right',shrink=0.9)
cbar.set_label("AEP gain [%]")
if savefig: plt.savefig(name_path+'aep_gains_robustness_v2'+'.'+format_fig,format=format_fig,bbox_inches='tight')
plt.show()




name_list = ['aep_gains_robustness_sigmaEval0',
             'aep_gains_robustness_sigmaEval25',
             'aep_gains_robustness_sigmaEval5',]

xlabel_list = [r'$\sigma_{\theta,\mathrm{opt}}=0^\circ$',r'$\sigma_{\theta,\mathrm{opt}}=2.5^\circ$',r'$\sigma_{\theta,\mathrm{opt}}=5^\circ$']
bar_width = 0.2
x_plot = np.arange(len(xlabel_list))
offsets = np.array([-bar_width, 0, bar_width])

for i in np.arange(len(sigma_eval_array)):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x_plot+offsets[0],aep_gain_mat_mixedOpt[i,:],width=bar_width,color=colors[0],label='Combined')
    ax.bar(x_plot+offsets[1],aep_gain_mat_yawOpt[i,:],width=bar_width,color=colors[2],label='Wake steering')
    ax.bar(x_plot+offsets[2],aep_gain_mat_helixOpt[i,:],width=bar_width,color=colors[4],label='Helix')
    ax.plot([np.min(x_plot)-0.5,np.max(x_plot)+0.5],[0.,0.],c='k',linewidth=0.5)
    ax.set_xlim([-0.5,len(x_plot)-0.5])
    ax.set_xticks(x_plot)
    ax.set_xticklabels(xlabel_list)
    ax.set_ylabel('AEP gain [%]')
    ax.legend()
    if savefig: plt.savefig(name_path+name_list[i]+'.'+format_fig,format=format_fig)
    plt.show()






#%% WES review - avoid need synchronization

#%%
# functions

def calculate_dxdy(x,y,wd,diameter):
    x_mat_1 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
    x_mat_2 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
    y_mat_1 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
    y_mat_2 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))
    d = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)
    theta = np.arctan2(y_mat_2-y_mat_1,x_mat_2-x_mat_1)
    gamma = wd*(np.pi/180)-(3/2)*np.pi+theta
    dx = d*np.cos(gamma)/diameter
    dy = d*np.sin(gamma)/diameter
    eps = 1e-3
    dx[dx<eps] = np.inf
    return dx,dy

def filter_avoid_helixSynch(x,y,wd_array,diameter,k,helix_mat_unfiltered_ilk):
    i_dim,l_dim,k_dim = helix_mat_unfiltered_ilk.shape
    fil_avoid_synch_ilk = np.zeros((i_dim,l_dim,k_dim),dtype=bool)
    for wd_ind in np.arange(len(wd_array)):
        wd = wd_array[wd_ind]
        helix_on_ik = helix_mat_unfiltered_ilk[:,wd_ind,:]>0
        dx_ii,dy_ii = calculate_dxdy(x,y,wd,diameter)
        fil_wake_ii = ((dy_ii<=(0.5+k*dx_ii))&(dy_ii>=(-0.5-k*dx_ii)))&(dx_ii<np.inf)
        fil_wake_iik = np.tile(fil_wake_ii[:,:,na],(1,1,k_dim))
        helix_on_iik = np.tile(helix_on_ik[:,na,:],(1,i_dim,1))
        fil_avoid_synch_iik = fil_wake_iik & helix_on_iik
        fil_avoid_synch_ik = np.any(fil_avoid_synch_iik,axis=0)
        fil_avoid_synch_ilk[:,wd_ind,:] = fil_avoid_synch_ik
    helix_mat_filtered_ilk = helix_mat_unfiltered_ilk.copy()
    helix_mat_filtered_ilk[fil_avoid_synch_ilk] = 0
    return helix_mat_filtered_ilk

#%%
# calculate values (10 min)

t = time.time()

k = 0.01

helix_amp_helixOpt_0std_filNoSynch = filter_avoid_helixSynch(x,y,wd_array,diameter,k,helix_amp_helixOpt_0std)
helix_amp_helixOpt_25std_filNoSynch = filter_avoid_helixSynch(x,y,wd_array,diameter,k,helix_amp_helixOpt_25std)
helix_amp_helixOpt_5std_filNoSynch = filter_avoid_helixSynch(x,y,wd_array,diameter,k,helix_amp_helixOpt_5std)

p_mat_helixOpt_0std_filNoSynch = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_0std,helix_amp_helixOpt_0std_filNoSynch,sigma=0.,n=1)
p_mat_helixOpt_25std_filNoSynch = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_25std,helix_amp_helixOpt_25std_filNoSynch,sigma=2.5,n=9)
p_mat_helixOpt_5std_filNoSynch = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_5std,helix_amp_helixOpt_5std_filNoSynch,sigma=5.,n=17)
aep_gain_helixOpt_0std_filNoSynch = 100*((8760*np.sum(p_mat_helixOpt_0std_filNoSynch*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std
aep_gain_helixOpt_25std_filNoSynch = 100*((8760*np.sum(p_mat_helixOpt_25std_filNoSynch*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std
aep_gain_helixOpt_5std_filNoSynch = 100*((8760*np.sum(p_mat_helixOpt_5std_filNoSynch*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std


with open(f'AEPgains_helixNoSynch_v2.pkl', 'wb') as f:
    pickle.dump({'aep_gain_helixOpt_0std_filNoSynch' : aep_gain_helixOpt_0std_filNoSynch,
                 'aep_gain_helixOpt_25std_filNoSynch' : aep_gain_helixOpt_25std_filNoSynch,
                 'aep_gain_helixOpt_5std_filNoSynch' : aep_gain_helixOpt_5std_filNoSynch,
                 }, f)
    
print(f'Calculation completed - Time: {time.time()-t}')

#%%
# plot results

# extract data
with open(f'AEPgains_helixNoSynch_v2.pkl', 'rb') as f:
    data = pickle.load(f)

aep_gain_helixOpt_0std_filNoSynch = data['aep_gain_helixOpt_0std_filNoSynch']
aep_gain_helixOpt_25std_filNoSynch = data['aep_gain_helixOpt_25std_filNoSynch']
aep_gain_helixOpt_5std_filNoSynch = data['aep_gain_helixOpt_5std_filNoSynch']

k = 0.01
helix_amp_helixOpt_0std_filNoSynch = filter_avoid_helixSynch(x,y,wd_array,diameter,k,helix_amp_helixOpt_0std)
helix_amp_helixOpt_25std_filNoSynch = filter_avoid_helixSynch(x,y,wd_array,diameter,k,helix_amp_helixOpt_25std)
helix_amp_helixOpt_5std_filNoSynch = filter_avoid_helixSynch(x,y,wd_array,diameter,k,helix_amp_helixOpt_5std)


savefig = False
name_path = r'figures\WES_review\\'

wd_ind = 201
ws_ind = 5
fig,axs = plt.subplots(figsize=(10,4),nrows=1,ncols=2)
sc1 = axs[0].scatter(x,y,c=helix_amp_helixOpt_0std[:,wd_ind,ws_ind],cmap='Greens',vmin=0.,vmax=5.,edgecolors='black',linewidths=0.5)
axs[0].set_aspect('equal')
axs[0].axis('off')
axs[0].set_title('Helix (unfiltered)')
sc2 = axs[1].scatter(x,y,c=helix_amp_helixOpt_0std_filNoSynch[:,wd_ind,ws_ind],cmap='Greens',vmin=0.,vmax=5.,edgecolors='black',linewidths=0.5)
axs[1].set_aspect('equal')
axs[1].axis('off')
axs[1].set_title('Helix (filtered)')
cbar = fig.colorbar(sc1, ax=axs, location='right')
cbar.set_label("Helix amp [deg]")
if savefig: plt.savefig(name_path+'noSynch_HKNmap_example_wd201_ws8_0std.pdf',format='pdf',bbox_inches='tight')
plt.show()


colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']
xlabel_list = [r'$\sigma_{\theta}=0^\circ$',r'$\sigma_{\theta}=2.5^\circ$',r'$\sigma_{\theta}=5^\circ$']
aep_gain_helixOpt_array = np.array([aep_gain_helixOpt_0std,aep_gain_helixOpt_25std,aep_gain_helixOpt_5std])
aep_gain_helixOpt_filNoSynch_array = np.array([aep_gain_helixOpt_0std_filNoSynch,aep_gain_helixOpt_25std_filNoSynch,aep_gain_helixOpt_5std_filNoSynch])
bar_width = 0.2
x_plot = np.arange(len(xlabel_list))
offsets = np.array([-bar_width/2,bar_width/2])
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x_plot + offsets[0], aep_gain_helixOpt_array, width=bar_width, color=colors[4], label='Helix (unfiltered)')
ax.bar(x_plot + offsets[1], aep_gain_helixOpt_filNoSynch_array, width=bar_width, color=colors[4], label='Helix (filtered)',hatch='////',edgecolor='white')
ax.set_xticks(x_plot)
ax.set_xticklabels(xlabel_list)
ax.set_ylabel('AEP gain [%]')
ax.legend()
if savefig: plt.savefig(name_path+'noSynch_aep_gains_v2.pdf',format='pdf')
plt.show()




#%% WES review - helix only 5deg

#%%
# functions

def filter_onlyMaxAmp(helix_amp_ilk,helix_amp_max=5,helix_amp_mid=2.5):
    helix_amp_ilk_filtered = np.zeros_like(helix_amp_ilk)
    fil_helix_amp_mid = helix_amp_ilk>helix_amp_mid
    helix_amp_ilk_filtered[fil_helix_amp_mid] = helix_amp_max
    return helix_amp_ilk_filtered


#%%
# calculate values (10 min)

t = time.time()

helix_amp_helixOpt_0std_fil5deg = filter_onlyMaxAmp(helix_amp_helixOpt_0std)
helix_amp_helixOpt_25std_fil5deg = filter_onlyMaxAmp(helix_amp_helixOpt_25std)
helix_amp_helixOpt_5std_fil5deg = filter_onlyMaxAmp(helix_amp_helixOpt_5std)

p_mat_helixOpt_0std_fil5deg = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_0std,helix_amp_helixOpt_0std_fil5deg,sigma=0.,n=1)
p_mat_helixOpt_25std_fil5deg = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_25std,helix_amp_helixOpt_25std_fil5deg,sigma=2.5,n=9)
p_mat_helixOpt_5std_fil5deg = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_5std,helix_amp_helixOpt_5std_fil5deg,sigma=5.,n=17)
aep_gain_helixOpt_0std_fil5deg = 100*((8760*np.sum(p_mat_helixOpt_0std_fil5deg*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std
aep_gain_helixOpt_25std_fil5deg = 100*((8760*np.sum(p_mat_helixOpt_25std_fil5deg*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std
aep_gain_helixOpt_5std_fil5deg = 100*((8760*np.sum(p_mat_helixOpt_5std_fil5deg*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std


with open(f'AEPgains_helixOnly5deg_v2.pkl', 'wb') as f:
    pickle.dump({'aep_gain_helixOpt_0std_fil5deg' : aep_gain_helixOpt_0std_fil5deg,
                 'aep_gain_helixOpt_25std_fil5deg' : aep_gain_helixOpt_25std_fil5deg,
                 'aep_gain_helixOpt_5std_fil5deg' : aep_gain_helixOpt_5std_fil5deg,
                 }, f)
    
print(f'Calculation completed - Time: {time.time()-t}')


#%%
# plot results

# extract data
with open(f'AEPgains_helixOnly5deg_v2.pkl', 'rb') as f:
    data = pickle.load(f)

aep_gain_helixOpt_0std_fil5deg = data['aep_gain_helixOpt_0std_fil5deg']
aep_gain_helixOpt_25std_fil5deg = data['aep_gain_helixOpt_25std_fil5deg']
aep_gain_helixOpt_5std_fil5deg = data['aep_gain_helixOpt_5std_fil5deg']

savefig = False
name_path = r'figures\WES_review\\'

colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']
xlabel_list = [r'$\sigma_{\theta}=0^\circ$',r'$\sigma_{\theta}=2.5^\circ$',r'$\sigma_{\theta}=5^\circ$']
aep_gain_helixOpt_array = np.array([aep_gain_helixOpt_0std,aep_gain_helixOpt_25std,aep_gain_helixOpt_5std])
aep_gain_helixOpt_fil5deg_array = np.array([aep_gain_helixOpt_0std_fil5deg,aep_gain_helixOpt_25std_fil5deg,aep_gain_helixOpt_5std_fil5deg])
bar_width = 0.2
x_plot = np.arange(len(xlabel_list))
offsets = np.array([-bar_width/2,bar_width/2])
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x_plot + offsets[0], aep_gain_helixOpt_array, width=bar_width, color=colors[4], label='Helix')
ax.bar(x_plot + offsets[1], aep_gain_helixOpt_fil5deg_array, width=bar_width, color=colors[4], label='Helix (only 5 deg)',hatch='\\\\',edgecolor='white')
ax.set_xticks(x_plot)
ax.set_xticklabels(xlabel_list)
ax.set_ylabel('AEP gain [%]')
ax.legend()
if savefig: plt.savefig(name_path+'only5deg_aep_gains_v2.svg',format='svg')
plt.show()





#%% WES review - plot AEP gains per turbine


aep_delta_mixedOpt_0std_i = 8760*(np.sum(prob_mat*p_mat_mixedOpt_0std,axis=(1,2))-np.sum(prob_mat*p_mat_baseline_0std,axis=(1,2)))/1e9
aep_delta_mixedOpt_25std_i = 8760*(np.sum(prob_mat*p_mat_mixedOpt_25std,axis=(1,2))-np.sum(prob_mat*p_mat_baseline_25std,axis=(1,2)))/1e9
aep_delta_mixedOpt_5std_i = 8760*(np.sum(prob_mat*p_mat_mixedOpt_5std,axis=(1,2))-np.sum(prob_mat*p_mat_baseline_5std,axis=(1,2)))/1e9

aep_delta_yawOpt_0std_i = 8760*(np.sum(prob_mat*p_mat_yawOpt_0std,axis=(1,2))-np.sum(prob_mat*p_mat_baseline_0std,axis=(1,2)))/1e9
aep_delta_yawOpt_25std_i = 8760*(np.sum(prob_mat*p_mat_yawOpt_25std,axis=(1,2))-np.sum(prob_mat*p_mat_baseline_25std,axis=(1,2)))/1e9
aep_delta_yawOpt_5std_i = 8760*(np.sum(prob_mat*p_mat_yawOpt_5std,axis=(1,2))-np.sum(prob_mat*p_mat_baseline_5std,axis=(1,2)))/1e9

aep_delta_helixOpt_0std_i = 8760*(np.sum(prob_mat*p_mat_helixOpt_0std,axis=(1,2))-np.sum(prob_mat*p_mat_baseline_0std,axis=(1,2)))/1e9
aep_delta_helixOpt_25std_i = 8760*(np.sum(prob_mat*p_mat_helixOpt_25std,axis=(1,2))-np.sum(prob_mat*p_mat_baseline_25std,axis=(1,2)))/1e9
aep_delta_helixOpt_5std_i = 8760*(np.sum(prob_mat*p_mat_helixOpt_5std,axis=(1,2))-np.sum(prob_mat*p_mat_baseline_5std,axis=(1,2)))/1e9


savefig = False
name_path = r'figures\WES_review\\'


from matplotlib.colors import TwoSlopeNorm

fig,axs = plt.subplots(figsize=(12,10),nrows=3,ncols=3)

max_val = np.maximum(np.maximum(np.maximum(-np.min(aep_delta_yawOpt_0std_i),np.max(aep_delta_yawOpt_0std_i)),np.maximum(-np.min(aep_delta_helixOpt_0std_i),np.max(aep_delta_helixOpt_0std_i))),np.maximum(-np.min(aep_delta_mixedOpt_0std_i),np.max(aep_delta_mixedOpt_0std_i)))
norm = TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

axs[0,0].set_title(r'Wake steering ($\sigma_\theta=0^\circ$)')
sc00 = axs[0,0].scatter(x,y,c=aep_delta_yawOpt_0std_i,cmap='PuOr',norm=norm,edgecolors='black',linewidths=0.5)
axs[0,0].set_aspect('equal')
axs[0,0].axis('off')

axs[0,1].set_title(r'Helix ($\sigma_\theta=0^\circ$)')
sc01 = axs[0,1].scatter(x,y,c=aep_delta_helixOpt_0std_i,cmap='PuOr',norm=norm,edgecolors='black',linewidths=0.5)
axs[0,1].set_aspect('equal')
axs[0,1].axis('off')

axs[0,2].set_title(r'Combined ($\sigma_\theta=0^\circ$)')
sc02 = axs[0,2].scatter(x,y,c=aep_delta_mixedOpt_0std_i,cmap='PuOr',norm=norm,edgecolors='black',linewidths=0.5)
axs[0,2].set_aspect('equal')
axs[0,2].axis('off')

cbar = fig.colorbar(sc00, ax=axs[0,:], location='right')
cbar.set_label("Delta AEP [GWh]")

max_val = np.maximum(np.maximum(np.maximum(-np.min(aep_delta_yawOpt_25std_i),np.max(aep_delta_yawOpt_25std_i)),np.maximum(-np.min(aep_delta_helixOpt_25std_i),np.max(aep_delta_helixOpt_25std_i))),np.maximum(-np.min(aep_delta_mixedOpt_25std_i),np.max(aep_delta_mixedOpt_25std_i)))
norm = TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

axs[1,0].set_title(r'Wake steering ($\sigma_\theta=2.5^\circ$)')
sc10 = axs[1,0].scatter(x,y,c=aep_delta_yawOpt_25std_i,cmap='PuOr',norm=norm,edgecolors='black',linewidths=0.5)
axs[1,0].set_aspect('equal')
axs[1,0].axis('off')

axs[1,1].set_title(r'Helix ($\sigma_\theta=2.5^\circ$)')
sc11 = axs[1,1].scatter(x,y,c=aep_delta_helixOpt_25std_i,cmap='PuOr',norm=norm,edgecolors='black',linewidths=0.5)
axs[1,1].set_aspect('equal')
axs[1,1].axis('off')

axs[1,2].set_title(r'Combined ($\sigma_\theta=2.5^\circ$)')
sc12 = axs[1,2].scatter(x,y,c=aep_delta_mixedOpt_25std_i,cmap='PuOr',norm=norm,edgecolors='black',linewidths=0.5)
axs[1,2].set_aspect('equal')
axs[1,2].axis('off')

cbar = fig.colorbar(sc10, ax=axs[1,:], location='right')
cbar.set_label("Delta AEP [GWh]")

max_val = np.maximum(np.maximum(np.maximum(-np.min(aep_delta_yawOpt_5std_i),np.max(aep_delta_yawOpt_5std_i)),np.maximum(-np.min(aep_delta_helixOpt_5std_i),np.max(aep_delta_helixOpt_5std_i))),np.maximum(-np.min(aep_delta_mixedOpt_5std_i),np.max(aep_delta_mixedOpt_5std_i)))
norm = TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

axs[2,0].set_title(r'Wake steering ($\sigma_\theta=5^\circ$)')
sc20 = axs[2,0].scatter(x,y,c=aep_delta_yawOpt_5std_i,cmap='PuOr',norm=norm,edgecolors='black',linewidths=0.5)
axs[2,0].set_aspect('equal')
axs[2,0].axis('off')

axs[2,1].set_title(r'Helix ($\sigma_\theta=5^\circ$)')
sc21 = axs[2,1].scatter(x,y,c=aep_delta_helixOpt_5std_i,cmap='PuOr',norm=norm,edgecolors='black',linewidths=0.5)
axs[2,1].set_aspect('equal')
axs[2,1].axis('off')

axs[2,2].set_title(r'Combined ($\sigma_\theta=5^\circ$)')
sc22 = axs[2,2].scatter(x,y,c=aep_delta_mixedOpt_5std_i,cmap='PuOr',norm=norm,edgecolors='black',linewidths=0.5)
axs[2,2].set_aspect('equal')
axs[2,2].axis('off')

cbar = fig.colorbar(sc20, ax=axs[2,:], location='right')
cbar.set_label("Delta AEP [GWh]")

if savefig: plt.savefig(name_path+'delta_aep_per_turbine_v2.svg',format='svg',bbox_inches='tight')
plt.show()


#%%
# AEP with pp=1.88 (only evaluation) - (40 min)
# ASSUMPTION: in PyWake pp=2.8 (not really true because it is based on the power curve)

from py_wake_helix.py_wake_helix_tools import calculatePmat_withUncertainty_pp

t = time.time()
p_mat_baseline_0std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,np.zeros((len(x),len(wd_array),len(ws_array))),np.zeros((len(x),len(wd_array),len(ws_array))),sigma=0.,n=1)
p_mat_mixedOpt_0std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_0std,helix_amp_mixedOpt_0std,sigma=0.,n=1)
p_mat_yawOpt_0std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,yaw_yawOpt_0std,helix_amp_yawOpt_0std,sigma=0.,n=1)
p_mat_helixOpt_0std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,yaw_helixOpt_0std,helix_amp_helixOpt_0std,sigma=0.,n=1)
print(f'Sigma=0 completed - Time: {time.time()-t}')

t = time.time()
p_mat_baseline_25std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,np.zeros((len(x),len(wd_array),len(ws_array))),np.zeros((len(x),len(wd_array),len(ws_array))),sigma=2.5,n=9)
p_mat_mixedOpt_25std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_25std,helix_amp_mixedOpt_25std,sigma=2.5,n=9)
p_mat_yawOpt_25std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,yaw_yawOpt_25std,helix_amp_yawOpt_25std,sigma=2.5,n=9)
p_mat_helixOpt_25std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,yaw_helixOpt_25std,helix_amp_helixOpt_25std,sigma=2.5,n=9)
print(f'Sigma=2.5 completed - Time: {time.time()-t}')

t = time.time()
p_mat_baseline_5std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,np.zeros((len(x),len(wd_array),len(ws_array))),np.zeros((len(x),len(wd_array),len(ws_array))),sigma=5.,n=17)
p_mat_mixedOpt_5std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,yaw_mixedOpt_5std,helix_amp_mixedOpt_5std,sigma=5.,n=17)
p_mat_yawOpt_5std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,yaw_yawOpt_5std,helix_amp_yawOpt_5std,sigma=5.,n=17)
p_mat_helixOpt_5std_pp188 = calculatePmat_withUncertainty_pp(wfm,x,y,wd_array,ws_array,yaw_helixOpt_5std,helix_amp_helixOpt_5std,sigma=5.,n=17)
print(f'Sigma=5 completed - Time: {time.time()-t}')


aep_baseline_0std_pp188 = 8760*np.sum(p_mat_baseline_0std_pp188*prob_mat)/1e9
aep_gain_mixedOpt_0std_pp188 = 100*((8760*np.sum(p_mat_mixedOpt_0std_pp188*prob_mat)/1e9)-aep_baseline_0std_pp188)/aep_baseline_0std_pp188
aep_gain_yawOpt_0std_pp188 = 100*((8760*np.sum(p_mat_yawOpt_0std_pp188*prob_mat)/1e9)-aep_baseline_0std_pp188)/aep_baseline_0std_pp188
aep_gain_helixOpt_0std_pp188 = 100*((8760*np.sum(p_mat_helixOpt_0std_pp188*prob_mat)/1e9)-aep_baseline_0std_pp188)/aep_baseline_0std_pp188

aep_baseline_25std_pp188 = 8760*np.sum(p_mat_baseline_25std_pp188*prob_mat)/1e9
aep_gain_mixedOpt_25std_pp188 = 100*((8760*np.sum(p_mat_mixedOpt_25std_pp188*prob_mat)/1e9)-aep_baseline_25std_pp188)/aep_baseline_25std_pp188
aep_gain_yawOpt_25std_pp188 = 100*((8760*np.sum(p_mat_yawOpt_25std_pp188*prob_mat)/1e9)-aep_baseline_25std_pp188)/aep_baseline_25std_pp188
aep_gain_helixOpt_25std_pp188 = 100*((8760*np.sum(p_mat_helixOpt_25std_pp188*prob_mat)/1e9)-aep_baseline_25std_pp188)/aep_baseline_25std_pp188

aep_baseline_5std_pp188 = 8760*np.sum(p_mat_baseline_5std_pp188*prob_mat)/1e9
aep_gain_mixedOpt_5std_pp188 = 100*((8760*np.sum(p_mat_mixedOpt_5std_pp188*prob_mat)/1e9)-aep_baseline_5std_pp188)/aep_baseline_5std_pp188
aep_gain_yawOpt_5std_pp188 = 100*((8760*np.sum(p_mat_yawOpt_5std_pp188*prob_mat)/1e9)-aep_baseline_5std_pp188)/aep_baseline_5std_pp188
aep_gain_helixOpt_5std_pp188 = 100*((8760*np.sum(p_mat_helixOpt_5std_pp188*prob_mat)/1e9)-aep_baseline_5std_pp188)/aep_baseline_5std_pp188

with open(f'AEPgains_EvalPp188.pkl', 'wb') as f:
    pickle.dump({'aep_gain_mixedOpt_0std_pp188' : aep_gain_mixedOpt_0std_pp188,
                 'aep_gain_yawOpt_0std_pp188' : aep_gain_yawOpt_0std_pp188,
                 'aep_gain_helixOpt_0std_pp188' : aep_gain_helixOpt_0std_pp188,
                 'aep_gain_mixedOpt_25std_pp188' : aep_gain_mixedOpt_25std_pp188,
                 'aep_gain_yawOpt_25std_pp188' : aep_gain_yawOpt_25std_pp188,
                 'aep_gain_helixOpt_25std_pp188' : aep_gain_helixOpt_25std_pp188,
                 'aep_gain_mixedOpt_5std_pp188' : aep_gain_mixedOpt_5std_pp188,
                 'aep_gain_yawOpt_5std_pp188' : aep_gain_yawOpt_5std_pp188,
                 'aep_gain_helixOpt_5std_pp188' : aep_gain_helixOpt_5std_pp188,
                 }, f)


#%%
# AEP with pp=1.88 (only evaluation) - plot
# ASSUMPTION: in PyWake pp=2.8 (not really true because it is based on the power curve)

# extract data
with open(f'AEPgains_EvalPp188.pkl', 'rb') as f:
    data = pickle.load(f)

aep_gain_mixedOpt_0std_pp188 = data['aep_gain_mixedOpt_0std_pp188']
aep_gain_yawOpt_0std_pp188 = data['aep_gain_yawOpt_0std_pp188']
aep_gain_helixOpt_0std_pp188 = data['aep_gain_helixOpt_0std_pp188']

aep_gain_mixedOpt_25std_pp188 = data['aep_gain_mixedOpt_25std_pp188']
aep_gain_yawOpt_25std_pp188 = data['aep_gain_yawOpt_25std_pp188']
aep_gain_helixOpt_25std_pp188 = data['aep_gain_helixOpt_25std_pp188']

aep_gain_mixedOpt_5std_pp188 = data['aep_gain_mixedOpt_5std_pp188']
aep_gain_yawOpt_5std_pp188 = data['aep_gain_yawOpt_5std_pp188']
aep_gain_helixOpt_5std_pp188 = data['aep_gain_helixOpt_5std_pp188']


savefig = False
name_path = r'figures\WES_review\\'
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']

xlabel_list = [r'$\sigma_{\theta}=0^\circ$',r'$\sigma_{\theta}=2.5^\circ$',r'$\sigma_{\theta}=5^\circ$']
aep_gain_mixedOpt_pp188_array = np.array([aep_gain_mixedOpt_0std_pp188,aep_gain_mixedOpt_25std_pp188,aep_gain_mixedOpt_5std_pp188])
aep_gain_yawOpt_pp188_array = np.array([aep_gain_yawOpt_0std_pp188,aep_gain_yawOpt_25std_pp188,aep_gain_yawOpt_5std_pp188])
aep_gain_helixOpt_pp188_array = np.array([aep_gain_helixOpt_0std_pp188,aep_gain_helixOpt_25std_pp188,aep_gain_helixOpt_5std_pp188])

bar_width = 0.2
x_plot = np.arange(len(xlabel_list))
offsets = np.array([-bar_width, 0, bar_width])

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(x_plot + offsets[0], aep_gain_mixedOpt_pp188_array, width=bar_width, color=colors[0], label='Combined')
ax.bar(x_plot + offsets[1], aep_gain_yawOpt_pp188_array, width=bar_width, color=colors[2], label='Wake steering')
ax.bar(x_plot + offsets[2], aep_gain_helixOpt_pp188_array, width=bar_width, color=colors[4], label='Helix')

ax.set_xticks(x_plot)
ax.set_xticklabels(xlabel_list)
ax.set_ylabel('AEP gain [%]')
ax.legend()

if savefig: plt.savefig(name_path+'aep_gains_pp188.pdf',format='pdf')
plt.show()


print(aep_gain_mixedOpt_pp188_array)
print(aep_gain_yawOpt_pp188_array)
print(aep_gain_helixOpt_pp188_array)




    
#%% FILTER LUT (avoid control when there is no gain)


# filter LUT - sigma=0 =====================================================================================================

yaw_mixedOpt_0std_filtered = np.zeros_like(yaw_mixedOpt_0std)
yaw_yawOpt_0std_filtered = np.zeros_like(yaw_yawOpt_0std)
yaw_helixOpt_0std_filtered = np.zeros_like(yaw_helixOpt_0std)

yaw_mixedOpt_0std_filtered[np.tile(p_gain_mixedOpt_0std_lk[na,:,:]>0,(len(x),1,1))] = yaw_mixedOpt_0std[np.tile(p_gain_mixedOpt_0std_lk[na,:,:]>0,(len(x),1,1))]
yaw_yawOpt_0std_filtered[np.tile(p_gain_yawOpt_0std_lk[na,:,:]>0,(len(x),1,1))] = yaw_yawOpt_0std[np.tile(p_gain_yawOpt_0std_lk[na,:,:]>0,(len(x),1,1))]
yaw_helixOpt_0std_filtered[np.tile(p_gain_helixOpt_0std_lk[na,:,:]>0,(len(x),1,1))] = yaw_helixOpt_0std[np.tile(p_gain_helixOpt_0std_lk[na,:,:]>0,(len(x),1,1))]

helix_amp_mixedOpt_0std_filtered = np.zeros_like(helix_amp_mixedOpt_0std)
helix_amp_yawOpt_0std_filtered = np.zeros_like(helix_amp_yawOpt_0std)
helix_amp_helixOpt_0std_filtered = np.zeros_like(helix_amp_helixOpt_0std)

helix_amp_mixedOpt_0std_filtered[np.tile(p_gain_mixedOpt_0std_lk[na,:,:]>0,(len(x),1,1))] = helix_amp_mixedOpt_0std[np.tile(p_gain_mixedOpt_0std_lk[na,:,:]>0,(len(x),1,1))]
helix_amp_yawOpt_0std_filtered[np.tile(p_gain_yawOpt_0std_lk[na,:,:]>0,(len(x),1,1))] = helix_amp_yawOpt_0std[np.tile(p_gain_yawOpt_0std_lk[na,:,:]>0,(len(x),1,1))]
helix_amp_helixOpt_0std_filtered[np.tile(p_gain_helixOpt_0std_lk[na,:,:]>0,(len(x),1,1))] = helix_amp_helixOpt_0std[np.tile(p_gain_helixOpt_0std_lk[na,:,:]>0,(len(x),1,1))]


# filter LUT - sigma=2.5 =====================================================================================================

yaw_mixedOpt_25std_filtered = np.zeros_like(yaw_mixedOpt_25std)
yaw_yawOpt_25std_filtered = np.zeros_like(yaw_yawOpt_25std)
yaw_helixOpt_25std_filtered = np.zeros_like(yaw_helixOpt_25std)

yaw_mixedOpt_25std_filtered[np.tile(p_gain_mixedOpt_25std_lk[na,:,:]>0,(len(x),1,1))] = yaw_mixedOpt_25std[np.tile(p_gain_mixedOpt_25std_lk[na,:,:]>0,(len(x),1,1))]
yaw_yawOpt_25std_filtered[np.tile(p_gain_yawOpt_25std_lk[na,:,:]>0,(len(x),1,1))] = yaw_yawOpt_25std[np.tile(p_gain_yawOpt_25std_lk[na,:,:]>0,(len(x),1,1))]
yaw_helixOpt_25std_filtered[np.tile(p_gain_helixOpt_25std_lk[na,:,:]>0,(len(x),1,1))] = yaw_helixOpt_25std[np.tile(p_gain_helixOpt_25std_lk[na,:,:]>0,(len(x),1,1))]

helix_amp_mixedOpt_25std_filtered = np.zeros_like(helix_amp_mixedOpt_25std)
helix_amp_yawOpt_25std_filtered = np.zeros_like(helix_amp_yawOpt_25std)
helix_amp_helixOpt_25std_filtered = np.zeros_like(helix_amp_helixOpt_25std)

helix_amp_mixedOpt_25std_filtered[np.tile(p_gain_mixedOpt_25std_lk[na,:,:]>0,(len(x),1,1))] = helix_amp_mixedOpt_25std[np.tile(p_gain_mixedOpt_25std_lk[na,:,:]>0,(len(x),1,1))]
helix_amp_yawOpt_25std_filtered[np.tile(p_gain_yawOpt_25std_lk[na,:,:]>0,(len(x),1,1))] = helix_amp_yawOpt_25std[np.tile(p_gain_yawOpt_25std_lk[na,:,:]>0,(len(x),1,1))]
helix_amp_helixOpt_25std_filtered[np.tile(p_gain_helixOpt_25std_lk[na,:,:]>0,(len(x),1,1))] = helix_amp_helixOpt_25std[np.tile(p_gain_helixOpt_25std_lk[na,:,:]>0,(len(x),1,1))]


# filter LUT - sigma=5 =====================================================================================================

yaw_mixedOpt_5std_filtered = np.zeros_like(yaw_mixedOpt_5std)
yaw_yawOpt_5std_filtered = np.zeros_like(yaw_yawOpt_5std)
yaw_helixOpt_5std_filtered = np.zeros_like(yaw_helixOpt_5std)

yaw_mixedOpt_5std_filtered[np.tile(p_gain_mixedOpt_5std_lk[na,:,:]>0,(len(x),1,1))] = yaw_mixedOpt_5std[np.tile(p_gain_mixedOpt_5std_lk[na,:,:]>0,(len(x),1,1))]
yaw_yawOpt_5std_filtered[np.tile(p_gain_yawOpt_5std_lk[na,:,:]>0,(len(x),1,1))] = yaw_yawOpt_5std[np.tile(p_gain_yawOpt_5std_lk[na,:,:]>0,(len(x),1,1))]
yaw_helixOpt_5std_filtered[np.tile(p_gain_helixOpt_5std_lk[na,:,:]>0,(len(x),1,1))] = yaw_helixOpt_5std[np.tile(p_gain_helixOpt_5std_lk[na,:,:]>0,(len(x),1,1))]

helix_amp_mixedOpt_5std_filtered = np.zeros_like(helix_amp_mixedOpt_5std)
helix_amp_yawOpt_5std_filtered = np.zeros_like(helix_amp_yawOpt_5std)
helix_amp_helixOpt_5std_filtered = np.zeros_like(helix_amp_helixOpt_5std)

helix_amp_mixedOpt_5std_filtered[np.tile(p_gain_mixedOpt_5std_lk[na,:,:]>0,(len(x),1,1))] = helix_amp_mixedOpt_5std[np.tile(p_gain_mixedOpt_5std_lk[na,:,:]>0,(len(x),1,1))]
helix_amp_yawOpt_5std_filtered[np.tile(p_gain_yawOpt_5std_lk[na,:,:]>0,(len(x),1,1))] = helix_amp_yawOpt_5std[np.tile(p_gain_yawOpt_5std_lk[na,:,:]>0,(len(x),1,1))]
helix_amp_helixOpt_5std_filtered[np.tile(p_gain_helixOpt_5std_lk[na,:,:]>0,(len(x),1,1))] = helix_amp_helixOpt_5std[np.tile(p_gain_helixOpt_5std_lk[na,:,:]>0,(len(x),1,1))]



#%%
# calculate impact of using helix in region III

t_init = time.time()
ws_filter = simres_helixOpt_0std.WS_eff_ilk<=11.

# sigma = 0
t = time.time()
helix_amp_helixOpt_0std_filterNoIII = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_helixOpt_0std_filterNoIII[ws_filter] = helix_amp_helixOpt_0std_filtered[ws_filter]
p_mat_helixOpt_0std_filterNoIII = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_0std,helix_amp_helixOpt_0std_filterNoIII,sigma=0.,n=1)
aep_gain_helixOpt_0std_filterNoIII = 100*((8760*np.sum(p_mat_helixOpt_0std_filterNoIII*prob_mat)/1e9)-aep_baseline_0std)/aep_baseline_0std
print(f'AEP gain: {aep_gain_helixOpt_0std_filterNoIII:.3f} - Time: {time.time()-t}')

# sigma = 2.5
t = time.time()
helix_amp_helixOpt_25std_filterNoIII = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_helixOpt_25std_filterNoIII[ws_filter] = helix_amp_helixOpt_25std_filtered[ws_filter]
p_mat_helixOpt_25std_filterNoIII = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_25std,helix_amp_helixOpt_25std_filterNoIII,sigma=2.5,n=9)
aep_gain_helixOpt_25std_filterNoIII = 100*((8760*np.sum(p_mat_helixOpt_25std_filterNoIII*prob_mat)/1e9)-aep_baseline_25std)/aep_baseline_25std
print(f'AEP gain: {aep_gain_helixOpt_25std_filterNoIII:.3f} - Time: {time.time()-t}')

# sigma = 5
t = time.time()
helix_amp_helixOpt_5std_filterNoIII = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp_helixOpt_5std_filterNoIII[ws_filter] = helix_amp_helixOpt_5std_filtered[ws_filter]
p_mat_helixOpt_5std_filterNoIII = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_helixOpt_5std,helix_amp_helixOpt_5std_filterNoIII,sigma=5.,n=17)
aep_gain_helixOpt_5std_filterNoIII = 100*((8760*np.sum(p_mat_helixOpt_5std_filterNoIII*prob_mat)/1e9)-aep_baseline_5std)/aep_baseline_5std
print(f'AEP gain: {aep_gain_helixOpt_5std_filterNoIII:.3f} - Time: {time.time()-t}')


with open(f'AEPgains_filterNoIII.pkl', 'wb') as f:
    pickle.dump({'aep_gain_helixOpt_0std_filterNoIII' : aep_gain_helixOpt_0std_filterNoIII,
                 'aep_gain_helixOpt_25std_filterNoIII' : aep_gain_helixOpt_25std_filterNoIII,
                 'aep_gain_helixOpt_5std_filterNoIII' : aep_gain_helixOpt_5std_filterNoIII,
                 }, f)
    
print(f'Calculation completed - Time: {time.time()-t_init}')


#%%
# plot results

# extract data
with open(f'AEPgains_filterNoIII.pkl', 'rb') as f:
    data = pickle.load(f)

aep_gain_helixOpt_0std_filterNoIII = data['aep_gain_helixOpt_0std_filterNoIII']
aep_gain_helixOpt_25std_filterNoIII = data['aep_gain_helixOpt_25std_filterNoIII']
aep_gain_helixOpt_5std_filterNoIII = data['aep_gain_helixOpt_5std_filterNoIII']

savefig = False
name_path = r'figures\WES_review\\'

colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']
xlabel_list = [r'$\sigma_{\theta}=0^\circ$',r'$\sigma_{\theta}=2.5^\circ$',r'$\sigma_{\theta}=5^\circ$']
aep_gain_helixOpt_array = np.array([aep_gain_helixOpt_0std,aep_gain_helixOpt_25std,aep_gain_helixOpt_5std])
aep_gain_helixOpt_filterNoIII_array = np.array([aep_gain_helixOpt_0std_filterNoIII,aep_gain_helixOpt_25std_filterNoIII,aep_gain_helixOpt_5std_filterNoIII])
bar_width = 0.2
x_plot = np.arange(len(xlabel_list))
offsets = np.array([-bar_width/2,bar_width/2])
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x_plot + offsets[0], aep_gain_helixOpt_array, width=bar_width, color=colors[4], label='Helix')
ax.bar(x_plot + offsets[1], aep_gain_helixOpt_filterNoIII_array, width=bar_width, color=colors[4], label='Helix (only region II)',hatch='\\\\',edgecolor='white')
ax.set_xticks(x_plot)
ax.set_xticklabels(xlabel_list)
ax.set_ylabel('AEP gain [%]')
ax.legend()
if savefig: plt.savefig(name_path+'filterNoIII_aep_gains.pdf',format='pdf')
plt.show()





#%% BAR PLOT: HOW OFTEN HELIX AND YAW CONTROL ARE ACTIVATED

# calculate probability of occurence for each turbine position ================================================================================================================

# wd_site = hkn_site_scaled.ds['wd'].values
# sector_frequency_per_turbine = np.zeros((len(wd_site),len(x)))
# weibull_A_per_turbine = np.zeros((len(wd_site),len(x)))
# weibull_k_per_turbine = np.zeros((len(wd_site),len(x)))

# for i in np.arange(len(wd_site)):
    
#     interp_func = RegularGridInterpolator((hkn_site_scaled.ds['x'].values,hkn_site_scaled.ds['y'].values),hkn_site_scaled.ds['Sector_frequency'].values[:,:,i])
#     sector_frequency_per_turbine[i,:] = interp_func(np.column_stack((x,y)))

#     interp_func = RegularGridInterpolator((hkn_site_scaled.ds['x'].values,hkn_site_scaled.ds['y'].values),hkn_site_scaled.ds['Weibull_A'].values[:,:,i])
#     weibull_A_per_turbine[i,:] = interp_func(np.column_stack((x,y)))

#     interp_func = RegularGridInterpolator((hkn_site_scaled.ds['x'].values,hkn_site_scaled.ds['y'].values),hkn_site_scaled.ds['Weibull_k'].values[:,:,i])
#     weibull_k_per_turbine[i,:] = interp_func(np.column_stack((x,y)))

# wd_array = np.arange(0,360,1)
# ws_array = np.arange(3,26,1)


# # ws probability per turbine (based on the Weibull distribution at every turbine location and wind direction sector)
# p_ws_per_turbine_temp = np.zeros((len(x),len(wd_site),len(ws_array)))
# for i in np.arange(len(ws_array)):
#     p_ws_per_turbine_temp[:,:,i] = np.exp(-(((ws_array[i]-0.5)/weibull_A_per_turbine.T)**weibull_k_per_turbine.T))-np.exp(-(((ws_array[i]+0.5)/weibull_A_per_turbine.T)**weibull_k_per_turbine.T))
# interp_func = interp1d(wd_site, p_ws_per_turbine_temp,axis=1)
# p_ws_per_turbine = interp_func(wd_array)

# # wd probability per turbine
# interp_func = interp1d(wd_site,sector_frequency_per_turbine.T,axis=1)
# p_wd_per_turbine_temp = interp_func(wd_array)
# p_wd_per_turbine = p_wd_per_turbine_temp*(np.sum(sector_frequency_per_turbine,axis=(0))/np.sum(p_wd_per_turbine_temp,axis=(1)))[:,na]

# # consider both contributions
# p_mat_per_turbine = p_ws_per_turbine*p_wd_per_turbine[:,:,na]

# extract porbability of flow cases (per turbine)
#prob_mat = simres_baseline_0std.P.values


# sigma=0 - calculate fraction of actuation of each control strategy ================================================================================

yaw_binary_mixedOpt_0std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_binary_yawOpt_0std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_binary_helixOpt_0std = np.zeros((len(x),len(wd_array),len(ws_array)))

helix_binary_mixedOpt_0std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_binary_yawOpt_0std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_binary_helixOpt_0std = np.zeros((len(x),len(wd_array),len(ws_array)))

yaw_binary_mixedOpt_0std[np.abs(yaw_mixedOpt_0std_filtered)>0] = 1
yaw_binary_yawOpt_0std[np.abs(yaw_yawOpt_0std_filtered)>0] = 1
yaw_binary_helixOpt_0std[np.abs(yaw_helixOpt_0std_filtered)>0] = 1

helix_binary_mixedOpt_0std[np.abs(helix_amp_mixedOpt_0std_filtered)>0] = 1
helix_binary_yawOpt_0std[np.abs(helix_amp_yawOpt_0std_filtered)>0] = 1
helix_binary_helixOpt_0std[np.abs(helix_amp_helixOpt_0std_filtered)>0] = 1

probability_yaw_mixedOpt_0std = np.sum(yaw_binary_mixedOpt_0std*prob_mat,axis=(1,2))
probability_yaw_yawOpt_0std = np.sum(yaw_binary_yawOpt_0std*prob_mat,axis=(1,2))
probability_yaw_helixOpt_0std = np.sum(yaw_binary_helixOpt_0std*prob_mat,axis=(1,2))

probability_helix_mixedOpt_0std = np.sum(helix_binary_mixedOpt_0std*prob_mat,axis=(1,2))
probability_helix_yawOpt_0std = np.sum(helix_binary_yawOpt_0std*prob_mat,axis=(1,2))
probability_helix_helixOpt_0std = np.sum(helix_binary_helixOpt_0std*prob_mat,axis=(1,2))


# sigma=2.5 - calculate fraction of actuation of each control strategy ================================================================================

yaw_binary_mixedOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_binary_yawOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_binary_helixOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))

helix_binary_mixedOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_binary_yawOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_binary_helixOpt_25std = np.zeros((len(x),len(wd_array),len(ws_array)))

yaw_binary_mixedOpt_25std[np.abs(yaw_mixedOpt_25std_filtered)>0] = 1
yaw_binary_yawOpt_25std[np.abs(yaw_yawOpt_25std_filtered)>0] = 1
yaw_binary_helixOpt_25std[np.abs(yaw_helixOpt_25std_filtered)>0] = 1

helix_binary_mixedOpt_25std[np.abs(helix_amp_mixedOpt_25std_filtered)>0] = 1
helix_binary_yawOpt_25std[np.abs(helix_amp_yawOpt_25std_filtered)>0] = 1
helix_binary_helixOpt_25std[np.abs(helix_amp_helixOpt_25std_filtered)>0] = 1

probability_yaw_mixedOpt_25std = np.sum(yaw_binary_mixedOpt_25std*prob_mat,axis=(1,2))
probability_yaw_yawOpt_25std = np.sum(yaw_binary_yawOpt_25std*prob_mat,axis=(1,2))
probability_yaw_helixOpt_25std = np.sum(yaw_binary_helixOpt_25std*prob_mat,axis=(1,2))

probability_helix_mixedOpt_25std = np.sum(helix_binary_mixedOpt_25std*prob_mat,axis=(1,2))
probability_helix_yawOpt_25std = np.sum(helix_binary_yawOpt_25std*prob_mat,axis=(1,2))
probability_helix_helixOpt_25std = np.sum(helix_binary_helixOpt_25std*prob_mat,axis=(1,2))


# sigma=5 - calculate fraction of actuation of each control strategy ================================================================================

yaw_binary_mixedOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_binary_yawOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
yaw_binary_helixOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))

helix_binary_mixedOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_binary_yawOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_binary_helixOpt_5std = np.zeros((len(x),len(wd_array),len(ws_array)))

yaw_binary_mixedOpt_5std[np.abs(yaw_mixedOpt_5std_filtered)>0] = 1
yaw_binary_yawOpt_5std[np.abs(yaw_yawOpt_5std_filtered)>0] = 1
yaw_binary_helixOpt_5std[np.abs(yaw_helixOpt_5std_filtered)>0] = 1

helix_binary_mixedOpt_5std[np.abs(helix_amp_mixedOpt_5std_filtered)>0] = 1
helix_binary_yawOpt_5std[np.abs(helix_amp_yawOpt_5std_filtered)>0] = 1
helix_binary_helixOpt_5std[np.abs(helix_amp_helixOpt_5std_filtered)>0] = 1

probability_yaw_mixedOpt_5std = np.sum(yaw_binary_mixedOpt_5std*prob_mat,axis=(1,2))
probability_yaw_yawOpt_5std = np.sum(yaw_binary_yawOpt_5std*prob_mat,axis=(1,2))
probability_yaw_helixOpt_5std = np.sum(yaw_binary_helixOpt_5std*prob_mat,axis=(1,2))

probability_helix_mixedOpt_5std = np.sum(helix_binary_mixedOpt_5std*prob_mat,axis=(1,2))
probability_helix_yawOpt_5std = np.sum(helix_binary_yawOpt_5std*prob_mat,axis=(1,2))
probability_helix_helixOpt_5std = np.sum(helix_binary_helixOpt_5std*prob_mat,axis=(1,2))



# plot (all turbines) =========================

savefig = False
#name_path = r'figures\LUT_HKN\\'
name_path = r'figures\WES_review\\'
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']


fig, axs = plt.subplots(3,figsize=(15, 9),sharex=True)

x_plot = np.arange(len(x))+0.4
bar_width = 0.25

axs[0].set_title('No uncertainty')
axs[0].bar(x_plot-bar_width, 100*probability_yaw_mixedOpt_0std, width=bar_width, color=colors[2], label='Mixed (Wake steering)',edgecolor='k',hatch='////')
axs[0].bar(x_plot-bar_width, 100*probability_helix_mixedOpt_0std, width=bar_width, bottom=100*probability_yaw_mixedOpt_0std,color=colors[4], label='Mixed (Helix)',edgecolor='k',hatch='////')
axs[0].bar(x_plot, 100*probability_yaw_yawOpt_0std, width=bar_width, color=colors[2], label='Wake steering',edgecolor='k')
axs[0].bar(x_plot+bar_width, 100*probability_helix_helixOpt_0std, width=bar_width,color=colors[4], label='Helix',edgecolor='k')
axs[0].set_xlim([-4*bar_width,len(x)-1+4*bar_width])
axs[0].set_ylabel('Percentage of operation [%]')
axs[0].legend()

axs[1].set_title('Uncertainty (sigma=2.5)')
axs[1].bar(x_plot-bar_width, 100*probability_yaw_mixedOpt_25std, width=bar_width, color=colors[2], label='Mixed (Wake steering)',edgecolor='k',hatch='////')
axs[1].bar(x_plot-bar_width, 100*probability_helix_mixedOpt_25std, width=bar_width, bottom=100*probability_yaw_mixedOpt_25std,color=colors[4], label='Mixed (Helix)',edgecolor='k',hatch='////')
axs[1].bar(x_plot, 100*probability_yaw_yawOpt_25std, width=bar_width, color=colors[2], label='Wake steering',edgecolor='k')
axs[1].bar(x_plot+bar_width, 100*probability_helix_helixOpt_25std, width=bar_width,color=colors[4], label='Helix',edgecolor='k')
axs[1].set_xlim([-4*bar_width,len(x)-1+4*bar_width])
axs[1].set_ylabel('Percentage of operation [%]')
axs[1].legend()

axs[2].set_title('Uncertainty (sigma=5)')
axs[2].bar(x_plot-bar_width, 100*probability_yaw_mixedOpt_5std, width=bar_width, color=colors[2], label='Mixed (Wake steering)',edgecolor='k',hatch='////')
axs[2].bar(x_plot-bar_width, 100*probability_helix_mixedOpt_5std, width=bar_width, bottom=100*probability_yaw_mixedOpt_5std,color=colors[4], label='Mixed (Helix)',edgecolor='k',hatch='////')
axs[2].bar(x_plot, 100*probability_yaw_yawOpt_5std, width=bar_width, color=colors[2], label='Wake steering',edgecolor='k')
axs[2].bar(x_plot+bar_width, 100*probability_helix_helixOpt_5std, width=bar_width,color=colors[4], label='Helix',edgecolor='k')
axs[2].set_xlim([-4*bar_width,len(x)-1+4*bar_width])
axs[2].set_ylabel('Percentage of operation [%]')
axs[2].legend()


axs[2].set_xlabel('Turbine')
axs[2].set_xticks(x_plot)
axs[2].set_xticklabels(np.arange(len(x)))

plt.tight_layout()
if savefig: plt.savefig(name_path+'perc_operation_barplot.svg',format='svg')
plt.show()


#%%
# plot (median-min-max of all turbines) =========================

savefig = False
#name_path = r'figures/LUT_HKN/'
name_path = r'figures\WES_review\\'
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']


xlabel_list = [r'$\sigma_{\theta}=0^\circ$',r'$\sigma_{\theta}=2.5^\circ$',r'$\sigma_{\theta}=5^\circ$']

prob_mixedOpt_list = [100*(probability_yaw_mixedOpt_0std+probability_helix_mixedOpt_0std),100*(probability_yaw_mixedOpt_25std+probability_helix_mixedOpt_25std),100*(probability_yaw_mixedOpt_5std+probability_helix_mixedOpt_5std)]
prob_yawOpt_list = [100*probability_yaw_yawOpt_0std,100*probability_yaw_yawOpt_25std,100*probability_yaw_yawOpt_5std]
prob_helixOpt_list = [100*probability_helix_helixOpt_0std,100*probability_helix_helixOpt_25std,100*probability_helix_helixOpt_5std]

bar_width = 0.15
x_plot = np.arange(len(xlabel_list))
offsets = np.array([-bar_width*1.15, 0, bar_width*1.15])

def set_box_color(bp,color):
    for element in ['boxes','whiskers','caps','medians']:
        plt.setp(bp[element], color=color)
    for patch in bp['boxes']:
        patch.set(facecolor=color,alpha=1)

fig, ax = plt.subplots(figsize=(6, 4))
box1 = ax.boxplot(prob_mixedOpt_list, positions=x_plot + offsets[0], patch_artist=True, widths=bar_width)
box2 = ax.boxplot(prob_yawOpt_list, positions=x_plot + offsets[1], patch_artist=True, widths=bar_width)
box3 = ax.boxplot(prob_helixOpt_list, positions=x_plot + offsets[2], patch_artist=True, widths=bar_width)
set_box_color(box1, colors[0])
set_box_color(box2, colors[2])
set_box_color(box3, colors[4])
ax.scatter([],[], c=colors[0],marker='s', label='Combined')
ax.scatter([],[], c=colors[2],marker='s', label='Wake steering')
ax.scatter([],[], c=colors[4],marker='s', label='Helix')
ax.set_xticks(x_plot)
ax.set_xticklabels(xlabel_list)
ax.set_ylabel(r'${\mathrm{COT}_i}$ [%]')
ax.legend()
#if savefig: plt.savefig(name_path+'perc_operation_boxplot.pdf',format='pdf')
if savefig: plt.savefig(name_path+'cot_boxplot_v2.pdf',format='pdf')
plt.show()


#%%
# plot (layout + percentage of operation) - mixed

savefig = False
name_path = r'figures/LUT_HKN/'

fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(10,4),gridspec_kw={'width_ratios':[1,1,1,0.05],'wspace':0.2})

axs[0].set_title(r'$\sigma_{\theta}=0^\circ$')
c_values = 100*(probability_yaw_mixedOpt_0std+probability_helix_mixedOpt_0std)
sc = axs[0].scatter(x,y,c=c_values,cmap='Purples',vmin=0,vmax=60,edgecolors='black',linewidths=0.25)
axs[0].set_aspect('equal')
axs[0].axis('off')

axs[1].set_title(r'$\sigma_{\theta}=2.5^\circ$')
c_values = 100*(probability_yaw_mixedOpt_25std+probability_helix_mixedOpt_25std)
sc = axs[1].scatter(x,y,c=c_values,cmap='Purples',vmin=0,vmax=60,edgecolors='black',linewidths=0.25)
axs[1].set_aspect('equal')
axs[1].axis('off')

axs[2].set_title(r'$\sigma_{\theta}=5^\circ$')
c_values = 100*(probability_yaw_mixedOpt_5std+probability_helix_mixedOpt_5std)
sc = axs[2].scatter(x,y,c=c_values,cmap='Purples',vmin=0,vmax=60,edgecolors='black',linewidths=0.25)
axs[2].set_aspect('equal')
axs[2].axis('off')

fig.colorbar(sc, cax=axs[3], label=r'$p_{\mathrm{LT},i}$ [%]')

if savefig: plt.savefig(name_path+'perc_operation_layout_mixedOpt.svg',format='svg')

plt.show()


#%%

#def plot_optControlRose(ind_turbine,
#                        x,
#                        y,
#                        yaw_ilk,
#                        helix_amp_ilk,
#                        x_boundaries,
#                        y_boundaries,
#                        savefig = False,
#                        name_path = None,
#                        name_fig = None,
#                        format_fig = None,
#                        ws_min = 3,
#                        ws_max = 14):
#    
#    # extract yaw and helix_amp values
#    yaw_mat = yaw_ilk[ind_turbine,:,:]
#    helix_mat = helix_amp_ilk[ind_turbine,:,:]
#
#    # define axis of the polar plot
#    ws_array_plot = ws_array[int(np.where(ws_array==ws_min)[0]):int(np.where(ws_array==ws_max)[0]+1)]
#    angles = np.linspace(0, 2 * np.pi, len(wd_array), endpoint=False)
#
#    # define colormaps
#    cmap1 = cm.bwr
#    cmap2 = cm.Greens
#    norm1 = mcolors.Normalize(vmin=-30., vmax=30)
#    norm2 = mcolors.Normalize(vmin=0., vmax=5.)
#
#    # initialize figure
#    fig = plt.figure(figsize=(8, 6))
#    ax = fig.add_subplot(111, projection='polar')
#    ax.set_theta_direction(-1)
#    ax.set_theta_offset(np.pi/2.0)
#
#    # define height of each bar
#    bar_height = 1
#
#    for ws_ind in np.arange(len(ws_array_plot)):
#        for wd_ind in np.arange(len(wd_array)):
#            
#            # calculate color based on the value of the control variable
#            if np.abs(yaw_mat[wd_ind, ws_ind]) > 0:
#                color = cmap1(norm1(yaw_mat[wd_ind, ws_ind]))
#            elif helix_mat[wd_ind, ws_ind] > 0:
#                color = cmap2(norm2(helix_mat[wd_ind, ws_ind]))
#            else:
#                color = [1, 1, 1, 1]
#
#            # plot bar
#            ax.bar(angles[wd_ind],height=bar_height,bottom=ws_ind*bar_height,color=color,width=2 *np.pi/len(wd_array),edgecolor='none',alpha=1.)
#
#    # add ticks correspondent to the wind speed and remove ticks for wind direction
#    ax.set_yticks(np.arange(len(ws_array_plot)))
#    ax.set_yticklabels(ws_array_plot)
#    ax.set_xticklabels([])
#
#    # place the main graph [left, bottom, width, height]
#    cbar_ax1 = fig.add_axes([0.80, 0.5, 0.02, 0.4])
#    cbar_ax2 = fig.add_axes([0.87, 0.5, 0.02, 0.4])  
#
#    # create colorbar for yaw angles
#    sm1 = cm.ScalarMappable(cmap=cmap1, norm=norm1)
#    sm1.set_array([])
#    cbar1 = plt.colorbar(sm1, cax=cbar_ax1)
#    cbar1.set_label('Yaw angles [deg]')
#    
#    # create colorbar for helix amp
#    sm2 = cm.ScalarMappable(cmap=cmap2, norm=norm2)
#    sm2.set_array([])
#    cbar2 = plt.colorbar(sm2, cax=cbar_ax2)
#    cbar2.set_label('Helix amp [deg]')
#
#    # create scatter plot to show the correspondent turbine in the layout
#    scatter_ax = fig.add_axes([0.73, 0.12, 0.18, 0.35])  
#    scatter_ax.scatter(x,y,c='k',alpha=0.6)
#    scatter_ax.scatter(x[ind_turbine],y[ind_turbine],c='r',alpha=1.)
#    scatter_ax.plot(x_boundaries,y_boundaries,c='k', alpha=0.6)
#    scatter_ax.axis('equal')
#    scatter_ax.set_xticks([])
#    scatter_ax.set_yticks([])
#    scatter_ax.set_frame_on(False)
#    
#    # save figure
#    if savefig: plt.savefig(name_path+name_fig,format=format_fig,bbox_inches='tight')
#
#    plt.show()


#%% PLOT: value of the control variable for each flow case (per turbine)

import matplotlib.cm as cm
import matplotlib.colors as mcolors



def plot_optControlRose(ind_turbine,
                        x,
                        y,
                        yaw_ilk,
                        helix_amp_ilk,
                        x_boundaries,
                        y_boundaries,
                        savefig = False,
                        name_path = None,
                        name_fig = None,
                        format_fig = None,
                        ws_min = 3,
                        ws_max = 14):
    
    # extract yaw and helix_amp values
    yaw_mat = yaw_ilk[ind_turbine,:,:]
    helix_mat = helix_amp_ilk[ind_turbine,:,:]

    # define axis of the polar plot
    ws_array_plot = ws_array[int(np.where(ws_array==ws_min)[0]):int(np.where(ws_array==ws_max)[0]+1)]
    angles = np.linspace(0, 2 * np.pi, len(wd_array), endpoint=False)

    # define colormaps
    cmap1 = cm.bwr
    cmap2 = cm.Greens
    norm1 = mcolors.Normalize(vmin=-30., vmax=30)
    norm2 = mcolors.Normalize(vmin=0., vmax=5.)

    # initialize figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)

    # define height of each bar
    bar_height = 1

    for ws_ind in np.arange(len(ws_array_plot)):
        for wd_ind in np.arange(len(wd_array)):
            
            # calculate color based on the value of the control variable
            if np.abs(yaw_mat[wd_ind, ws_ind]) > 0:
                color = cmap1(norm1(yaw_mat[wd_ind, ws_ind]))
            elif helix_mat[wd_ind, ws_ind] > 0:
                color = cmap2(norm2(helix_mat[wd_ind, ws_ind]))
            else:
                color = [1, 1, 1, 1]

            # plot bar
            ax.bar(angles[wd_ind],height=bar_height,bottom=ws_ind*bar_height,color=color,width=2 *np.pi/len(wd_array),edgecolor='none',alpha=1.)

    # add ticks correspondent to the wind speed and remove ticks for wind direction
    ax.set_yticks(np.arange(len(ws_array_plot)))
    ax.set_yticklabels(ws_array_plot)
    ax.set_xticklabels([])

    # place the main graph [left, bottom, width, height]
    cbar_ax1 = fig.add_axes([0.82, 0.5, 0.02, 0.4])
    cbar_ax2 = fig.add_axes([0.95, 0.5, 0.02, 0.4])  

    # create colorbar for yaw angles
    sm1 = cm.ScalarMappable(cmap=cmap1, norm=norm1)
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, cax=cbar_ax1)
    cbar1.set_label('Yaw angles [deg]')
    
    # create colorbar for helix amp
    sm2 = cm.ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, cax=cbar_ax2)
    cbar2.set_label('Helix amp [deg]')

    # create scatter plot to show the correspondent turbine in the layout
    scatter_ax = fig.add_axes([0.78, 0.12, 0.22, 0.35])  
    scatter_ax.scatter(x,y,c='k',alpha=0.6,s=10)
    scatter_ax.scatter(x[ind_turbine],y[ind_turbine],c='r',alpha=1.,s=80,marker='*')
    #scatter_ax.plot(x_boundaries,y_boundaries,c='k', alpha=0.6)
    scatter_ax.axis('equal')
    scatter_ax.set_xticks([])
    scatter_ax.set_yticks([])
    scatter_ax.set_frame_on(False)
    
    # save figure
    if savefig: plt.savefig(name_path+name_fig,format=format_fig,bbox_inches='tight')

    plt.show()



    


x = (hkn_wt_x-x_sub)*(diameter/diameter_hkn)
y = (hkn_wt_y-y_sub)*(diameter/diameter_hkn)
x_boundaries = (hkn_boundaries_x-x_sub)*(diameter/diameter_hkn)
y_boundaries = (hkn_boundaries_y-y_sub)*(diameter/diameter_hkn)

savefig = False
name_path = r'figures/LUT_HKN/'
#name_path = r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\LUT_HKN\\'
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']

#%%
# plot turbine 0 (mixed operation) ============================================

#plot_optControlRose(ind_turbine = 0,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_mixedOpt_0std_filtered,
#                    helix_amp_ilk = helix_amp_mixedOpt_0std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'mixedControlRose_wt0_0std.svg',
#                    format_fig = 'svg')

plot_optControlRose(ind_turbine = 0,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_mixedOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_mixedOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'mixedControlRose_wt0_25std.svg',
                    format_fig = 'svg')

#plot_optControlRose(ind_turbine = 0,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_mixedOpt_5std_filtered,
#                    helix_amp_ilk = helix_amp_mixedOpt_5std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'mixedControlRose_wt0_5std.svg',
#                    format_fig = 'svg')


# plot turbine 0 (yaw operation) ============================================

#plot_optControlRose(ind_turbine = 0,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_yawOpt_0std_filtered,
#                    helix_amp_ilk = helix_amp_yawOpt_0std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'yawControlRose_wt0_0std.svg',
#                    format_fig = 'svg')

plot_optControlRose(ind_turbine = 0,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_yawOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_yawOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'yawControlRose_wt0_25std.svg',
                    format_fig = 'svg')

#plot_optControlRose(ind_turbine = 0,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_yawOpt_5std_filtered,
#                    helix_amp_ilk = helix_amp_yawOpt_5std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'yawControlRose_wt0_5std.svg',
#                    format_fig = 'svg')
#

# plot turbine 0 (helix operation) ============================================

#plot_optControlRose(ind_turbine = 0,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_helixOpt_0std_filtered,
#                    helix_amp_ilk = helix_amp_helixOpt_0std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'helixControlRose_wt0_0std.svg',
#                    format_fig = 'svg')

plot_optControlRose(ind_turbine = 0,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_helixOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_helixOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'helixControlRose_wt0_25std.svg',
                    format_fig = 'svg')

#plot_optControlRose(ind_turbine = 0,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_helixOpt_5std_filtered,
#                    helix_amp_ilk = helix_amp_helixOpt_5std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'helixControlRose_wt0_5std.svg',
#                    format_fig = 'svg')



# plot turbine 30 (mixed operation) ============================================

#plot_optControlRose(ind_turbine = 30,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_mixedOpt_0std_filtered,
#                    helix_amp_ilk = helix_amp_mixedOpt_0std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'mixedControlRose_wt30_0std.svg',
#                    format_fig = 'svg')

plot_optControlRose(ind_turbine = 30,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_mixedOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_mixedOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'mixedControlRose_wt30_25std.svg',
                    format_fig = 'svg')

#plot_optControlRose(ind_turbine = 30,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_mixedOpt_5std_filtered,
#                    helix_amp_ilk = helix_amp_mixedOpt_5std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'mixedControlRose_wt30_5std.svg',
#                    format_fig = 'svg')


# plot turbine 0 (yaw operation) ============================================

#plot_optControlRose(ind_turbine = 30,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_yawOpt_0std_filtered,
#                    helix_amp_ilk = helix_amp_yawOpt_0std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'yawControlRose_wt30_0std.svg',
#                    format_fig = 'svg')

plot_optControlRose(ind_turbine = 30,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_yawOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_yawOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'yawControlRose_wt30_25std.svg',
                    format_fig = 'svg')

#plot_optControlRose(ind_turbine = 30,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_yawOpt_5std_filtered,
#                    helix_amp_ilk = helix_amp_yawOpt_5std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'yawControlRose_wt30_5std.svg',
#                    format_fig = 'svg')


# plot turbine 0 (helix operation) ============================================

#plot_optControlRose(ind_turbine = 30,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_helixOpt_0std_filtered,
#                    helix_amp_ilk = helix_amp_helixOpt_0std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'helixControlRose_wt30_0std.svg',
#                    format_fig = 'svg')

plot_optControlRose(ind_turbine = 30,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_helixOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_helixOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'helixControlRose_wt30_25std.svg',
                    format_fig = 'svg')

#plot_optControlRose(ind_turbine = 30,
#                    x = x,
#                    y = y,
#                    yaw_ilk = yaw_helixOpt_5std_filtered,
#                    helix_amp_ilk = helix_amp_helixOpt_5std_filtered,
#                    x_boundaries = x_boundaries,
#                    y_boundaries = y_boundaries,
#                    savefig = savefig,
#                    name_path = name_path,
#                    name_fig = 'helixControlRose_wt30_5std.svg',
#                    format_fig = 'svg')

#%%

plot_optControlRose(ind_turbine = 0,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_mixedOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_mixedOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'mixedControlRose_wt0_25std_v2.png',
                    format_fig = 'png')

plot_optControlRose(ind_turbine = 30,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_mixedOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_mixedOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'mixedControlRose_wt30_25std_v2.png',
                    format_fig = 'png')


plot_optControlRose(ind_turbine = 0,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_yawOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_yawOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'yawControlRose_wt0_25std_v2.png',
                    format_fig = 'png')

plot_optControlRose(ind_turbine = 30,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_yawOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_yawOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'yawControlRose_wt30_25std_v2.png',
                    format_fig = 'png')

plot_optControlRose(ind_turbine = 0,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_helixOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_helixOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'helixControlRose_wt0_25std_v2.png',
                    format_fig = 'png')

plot_optControlRose(ind_turbine = 30,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_helixOpt_25std_filtered,
                    helix_amp_ilk = helix_amp_helixOpt_25std_filtered,
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'helixControlRose_wt30_25std_v2.png',
                    format_fig = 'png')



#%%
# PLOT WIND RESOURCE MAP

savefig = False
#name_path = r'../figures_WES_paper/'
name_path = r'figures/WES_review/'

# scale parameters
hkn_site_x_grid_scaled = (hkn_site_x_grid-x_sub)*(diameter/diameter_hkn)
hkn_site_y_grid_scaled = (hkn_site_y_grid-y_sub)*(diameter/diameter_hkn)
hkn_boundaries_x_scaled = (hkn_boundaries_x-x_sub)*(diameter/diameter_hkn)
hkn_boundaries_y_scaled = (hkn_boundaries_y-y_sub)*(diameter/diameter_hkn)

# ws mean
ws_mean = hkn_ws_mean*((170./115.)**0.1)

# plot mean wind speed
fig,ax = plt.subplots(figsize=(6,5))
vmin = np.min(ws_mean)
vmax = np.max(ws_mean)
alpha=0.6
contour = ax.contourf(hkn_site_x_grid_scaled,hkn_site_y_grid_scaled,ws_mean,cmap='Blues',alpha=alpha,vmin=vmin,vmax=vmax)
cbar = plt.colorbar(contour,label=r'Mean wind speed [$\mathrm{m\,s^{-1}}$]')
ax.scatter(x,y,c='k')
#ax.plot(hkn_boundaries_x_scaled,hkn_boundaries_y_scaled,c='k')
plt.axis('equal')
ax.axis('off')
if savefig: plt.savefig(name_path+'HKN_ws_map.pdf',format='pdf')
plt.show()





#%%
# PLOT WEIBULL DISTRIBUTION

savefig = False
#name_path = r'../figures_WES_paper/'
name_path = r'figures/WES_review/'


def weibull_pdf(x, k, a):
    return (k / a) * (x / a)**(k - 1) * np.exp(-(x / a)**k)

k = ds_hkn_scaled['Weibull_k'].sel(x=x_sub, y=y_sub, wd=0, method="nearest").values
a = ds_hkn_scaled['Weibull_A'].sel(x=x_sub, y=y_sub, wd=0, method="nearest").values

x = np.linspace(0, 30, 1000)
y = weibull_pdf(x, k, a)

# plot mean Weibull
fig,ax = plt.subplots(figsize=(6,5))
ax.plot(x,y,c='b')
ax.set_ylabel(r'Density [-]')
ax.set_xlabel(r'Wind speed [$\mathrm{m\,s^{-1}}$]')
if savefig: plt.savefig(name_path+'HKN_Weibull.pdf',format='pdf')
plt.show()


#%%
# PLOT WIND ROSE (plot)

savefig = False
#name_path = r'../figures_WES_paper/'
name_path = r'figures/WES_review/'

#fig, ax = plt.subplots(figsize=(6,5))
#hkn_site_scaled.plot_wd_distribution(n_wd=12, ws_bins=[0,5,10,15,20,25],ax=ax)
#if savefig: plt.savefig(name_path+'wind_rose.pdf',format='pdf')
#plt.show()

fig, ax = plt.subplots(figsize=(4,4))

n_wd = 12
ws_bins = [0,5,10,15,20,25]
site = hkn_site_scaled

x = np.array([0.]) # substation
y = np.array([0.]) # substation
h = 170.


wd_bin_size = 360 // n_wd
wd = np.arange(0, 360, wd_bin_size)
theta = wd / 180 * np.pi
if not ax.__class__.__name__ == 'PolarAxesSubplot':
    if hasattr(ax, 'subplot'):
        ax.clf()
        ax = ax.subplot(111, projection='polar')
    else:
        ax.figure.clf()
        ax = ax.figure.add_subplot(111, projection='polar')
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2.0)

ws_bins = np.asarray(ws_bins)
ws = ((ws_bins[1:] + ws_bins[:-1]) / 2)

lw = site.local_wind(x=x, y=y, h=h, wd=np.arange(360), ws=ws, wd_bin_size=1)
P = lw.P

P.coords['sector'] = ('wd', site._sector(wd))
P = P.groupby('sector').sum()

cmap = plt.cm.Blues # set colormap
colors = cmap(np.linspace(0.1, 1., len(ws_bins),endpoint=True))

if ws_bins is None or 'ws' not in P.dims:
    ax.bar(theta, P.values, width=np.deg2rad(wd_bin_size), bottom=0.0)
else:
    P = P.T
    start_P = np.vstack([np.zeros_like(P[:1]), P.cumsum('ws')[:-1]])
    for ws1, ws2, p_ws0, p_ws, color in zip(lw.ws_lower[0, 0], lw.ws_upper[0, 0], start_P, P, colors):
        ax.bar(theta, p_ws, width=np.deg2rad(wd_bin_size), bottom=p_ws0,
               color = color,
                label=r"%0.0f-%0.0f $\mathrm{ms^{-1}}$" % (ws1, ws2))
    ax.legend(bbox_to_anchor=(1.15, 1.1),fontsize=11)

ax.set_yticks([0.03,0.06,0.09,0.12,0.15])
ax.set_yticklabels([])
ax.set_rlabel_position(-22.5)
ax.grid(True)

ax.tick_params(axis='x', labelsize=12)

if savefig: plt.savefig(name_path+'wind_rose_v4.pdf',format='pdf')
plt.show()



#%%
# PLOT WIND ROSE (plot) - black

savefig = False
#name_path = r'../figures_WES_paper/'
name_path = r'figures/LUT_HKN/'

#fig, ax = plt.subplots(figsize=(6,5))
#hkn_site_scaled.plot_wd_distribution(n_wd=12, ws_bins=[0,5,10,15,20,25],ax=ax)
#if savefig: plt.savefig(name_path+'wind_rose.pdf',format='pdf')
#plt.show()

fig, ax = plt.subplots(figsize=(4,4))

n_wd = 12
ws_bins = [0,5,10,15,20,25]
site = hkn_site_scaled

x = np.array([0.]) # substation
y = np.array([0.]) # substation
h = 170.


wd_bin_size = 360 // n_wd
wd = np.arange(0, 360, wd_bin_size)
theta = wd / 180 * np.pi
if not ax.__class__.__name__ == 'PolarAxesSubplot':
    if hasattr(ax, 'subplot'):
        ax.clf()
        ax = ax.subplot(111, projection='polar')
    else:
        ax.figure.clf()
        ax = ax.figure.add_subplot(111, projection='polar')
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2.0)

ws_bins = np.asarray(ws_bins)
ws = ((ws_bins[1:] + ws_bins[:-1]) / 2)

lw = site.local_wind(x=x, y=y, h=h, wd=np.arange(360), ws=ws, wd_bin_size=1)
P = lw.P

P.coords['sector'] = ('wd', site._sector(wd))
P = P.groupby('sector').sum()

cmap = plt.cm.Greys # set colormap
colors = cmap(np.linspace(0.1, 1., len(ws_bins),endpoint=True))

if ws_bins is None or 'ws' not in P.dims:
    ax.bar(theta, P.values, width=np.deg2rad(wd_bin_size), bottom=0.0)
else:
    P = P.T
    start_P = np.vstack([np.zeros_like(P[:1]), P.cumsum('ws')[:-1]])
    for ws1, ws2, p_ws0, p_ws, color in zip(lw.ws_lower[0, 0], lw.ws_upper[0, 0], start_P, P, colors):
        ax.bar(theta, p_ws, width=np.deg2rad(wd_bin_size), bottom=p_ws0,
               color = color,
                label=r"%0.0f-%0.0f $\mathrm{ms^{-1}}$" % (ws1, ws2))
    #ax.legend(bbox_to_anchor=(1.15, 1.1),fontsize=11)

#ax.set_yticks([0.03,0.06,0.09,0.12,0.15])
ax.set_yticklabels([])
ax.set_rlabel_position(-22.5)
ax.grid(True)

ax.tick_params(axis='x', labelsize=12)
#ax.set_xticklabels([])
#ax.axis('off')

if savefig: plt.savefig(name_path+'wind_rose_black_v2.svg',format='svg',bbox_inches='tight')
plt.show()





#%%
# CALCULATE MIN DISTANCE AND POWER DENSITY


x_mat_1 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
x_mat_2 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
y_mat_1 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
y_mat_2 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))
d = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)

d[d<1e-10] = np.inf
min_distance = np.min(d)
print(f'Min distance: {min_distance/diameter} D')

from shapely.geometry import Polygon

coords = list(zip(hkn_boundaries_x_scaled[:-1],hkn_boundaries_y_scaled[:-1]))
polygon = Polygon(coords)
area = polygon.area # [m2]
power_tot = 22*len(x)*1e6
print(f'Power density: {power_tot/area} W/m2')


#%% STUDY THE EFFECT OF SAMPLING (e.g. 5deg instead of 1deg)

# wd_array_2deg = np.arange(0,360,2)
# wd_array_5deg = np.arange(0,360,5)

# # sigma = 0 ============================================================================================

# t = time.time()

# # baseline ---------------------------------------------------------------
# simres_baseline_0std_1deg = simres_baseline_0std
# simres_baseline_0std_2deg = wfm(x,y,wd=wd_array_2deg,ws=ws_array,yaw=np.zeros((len(x),len(wd_array_2deg),len(ws_array))),tilt=0,helix_amp=np.zeros((len(x),len(wd_array_2deg),len(ws_array))))
# simres_baseline_0std_5deg = wfm(x,y,wd=wd_array_5deg,ws=ws_array,yaw=np.zeros((len(x),len(wd_array_5deg),len(ws_array))),tilt=0,helix_amp=np.zeros((len(x),len(wd_array_5deg),len(ws_array))))

# # mixed operation ---------------------------------------------------------
# yaw_mixedOpt_0std_2deg = yaw_mixedOpt_0std[:,np.arange(0,360,2),:]
# yaw_mixedOpt_0std_5deg = yaw_mixedOpt_0std[:,np.arange(0,360,5),:]
# helix_amp_mixedOpt_0std_2deg = helix_amp_mixedOpt_0std[:,np.arange(0,360,2),:]
# helix_amp_mixedOpt_0std_5deg = helix_amp_mixedOpt_0std[:,np.arange(0,360,5),:]
# simres_mixedOpt_0std_1deg = simres_mixedOpt_0std
# simres_mixedOpt_0std_2deg = wfm(x,y,wd=wd_array_2deg,ws=ws_array,yaw=yaw_mixedOpt_0std_2deg,tilt=0,helix_amp=helix_amp_mixedOpt_0std_2deg)
# simres_mixedOpt_0std_5deg = wfm(x,y,wd=wd_array_5deg,ws=ws_array,yaw=yaw_mixedOpt_0std_5deg,tilt=0,helix_amp=helix_amp_mixedOpt_0std_5deg)

# # yaw operation ---------------------------------------------------------
# yaw_yawOpt_0std_2deg = yaw_yawOpt_0std[:,np.arange(0,360,2),:]
# yaw_yawOpt_0std_5deg = yaw_yawOpt_0std[:,np.arange(0,360,5),:]
# helix_amp_yawOpt_0std_2deg = helix_amp_yawOpt_0std[:,np.arange(0,360,2),:]
# helix_amp_yawOpt_0std_5deg = helix_amp_yawOpt_0std[:,np.arange(0,360,5),:]
# simres_yawOpt_0std_1deg = simres_yawOpt_0std
# simres_yawOpt_0std_2deg = wfm(x,y,wd=wd_array_2deg,ws=ws_array,yaw=yaw_yawOpt_0std_2deg,tilt=0,helix_amp=helix_amp_yawOpt_0std_2deg)
# simres_yawOpt_0std_5deg = wfm(x,y,wd=wd_array_5deg,ws=ws_array,yaw=yaw_yawOpt_0std_5deg,tilt=0,helix_amp=helix_amp_yawOpt_0std_5deg)

# # helix operation ---------------------------------------------------------
# yaw_helixOpt_0std_2deg = yaw_helixOpt_0std[:,np.arange(0,360,2),:]
# yaw_helixOpt_0std_5deg = yaw_helixOpt_0std[:,np.arange(0,360,5),:]
# helix_amp_helixOpt_0std_2deg = helix_amp_helixOpt_0std[:,np.arange(0,360,2),:]
# helix_amp_helixOpt_0std_5deg = helix_amp_helixOpt_0std[:,np.arange(0,360,5),:]
# simres_helixOpt_0std_1deg = simres_helixOpt_0std
# simres_helixOpt_0std_2deg = wfm(x,y,wd=wd_array_2deg,ws=ws_array,yaw=yaw_helixOpt_0std_2deg,tilt=0,helix_amp=helix_amp_helixOpt_0std_2deg)
# simres_helixOpt_0std_5deg = wfm(x,y,wd=wd_array_5deg,ws=ws_array,yaw=yaw_helixOpt_0std_5deg,tilt=0,helix_amp=helix_amp_helixOpt_0std_5deg)

# print(f'Time: {time.time()-t}')

# #%%


# # calculate AEP gain - sigma=0 =====================================================================================================

# aep_baseline_0std_1deg = np.sum(simres_baseline_0std_1deg.aep().values)
# aep_baseline_0std_2deg = np.sum(simres_baseline_0std_2deg.aep().values)
# aep_baseline_0std_5deg = np.sum(simres_baseline_0std_5deg.aep().values)

# aep_mixedOpt_0std_1deg = np.sum(simres_mixedOpt_0std_1deg.aep().values)
# aep_mixedOpt_0std_2deg = np.sum(simres_mixedOpt_0std_2deg.aep().values)
# aep_mixedOpt_0std_5deg = np.sum(simres_mixedOpt_0std_5deg.aep().values)

# aep_yawOpt_0std_1deg = np.sum(simres_yawOpt_0std_1deg.aep().values)
# aep_yawOpt_0std_2deg = np.sum(simres_yawOpt_0std_2deg.aep().values)
# aep_yawOpt_0std_5deg = np.sum(simres_yawOpt_0std_5deg.aep().values)

# aep_helixOpt_0std_1deg = np.sum(simres_helixOpt_0std_1deg.aep().values)
# aep_helixOpt_0std_2deg = np.sum(simres_helixOpt_0std_2deg.aep().values)
# aep_helixOpt_0std_5deg = np.sum(simres_helixOpt_0std_5deg.aep().values)

# print('Sigma = 0 -----------------------------------------------------')
# print('Baseline ------------------------------------') 
# print(f'1 deg: AEP = {aep_baseline_0std_1deg}')
# print(f'2 deg: AEP = {aep_baseline_0std_2deg}')
# print(f'5 deg: AEP = {aep_baseline_0std_5deg}')
# print('Mixed control ------------------------------------') 
# print(f'1 deg: AEP = {aep_mixedOpt_0std_1deg}')
# print(f'2 deg: AEP = {aep_mixedOpt_0std_2deg}')
# print(f'5 deg: AEP = {aep_mixedOpt_0std_5deg}')
# print('Yaw control ------------------------------------')
# print(f'1 deg: AEP = {aep_yawOpt_0std_1deg}')
# print(f'2 deg: AEP = {aep_yawOpt_0std_2deg}')
# print(f'5 deg: AEP = {aep_yawOpt_0std_5deg}')
# print('Helix control ------------------------------------')
# print(f'1 deg: AEP = {aep_helixOpt_0std_1deg}')
# print(f'2 deg: AEP = {aep_helixOpt_0std_2deg}')
# print(f'5 deg: AEP = {aep_helixOpt_0std_5deg}')
# print('---------------------------------------------------------------')


# aep_gain_mixedOpt_0std_1deg = 100*(aep_mixedOpt_0std_1deg-aep_baseline_0std_1deg)/aep_baseline_0std_1deg
# aep_gain_mixedOpt_0std_2deg = 100*(aep_mixedOpt_0std_2deg-aep_baseline_0std_2deg)/aep_baseline_0std_2deg
# aep_gain_mixedOpt_0std_5deg = 100*(aep_mixedOpt_0std_5deg-aep_baseline_0std_5deg)/aep_baseline_0std_5deg

# aep_gain_yawOpt_0std_1deg = 100*(aep_yawOpt_0std_1deg-aep_baseline_0std_1deg)/aep_baseline_0std_1deg
# aep_gain_yawOpt_0std_2deg = 100*(aep_yawOpt_0std_2deg-aep_baseline_0std_2deg)/aep_baseline_0std_2deg
# aep_gain_yawOpt_0std_5deg = 100*(aep_yawOpt_0std_5deg-aep_baseline_0std_5deg)/aep_baseline_0std_5deg

# aep_gain_helixOpt_0std_1deg = 100*(aep_helixOpt_0std_1deg-aep_baseline_0std_1deg)/aep_baseline_0std_1deg
# aep_gain_helixOpt_0std_2deg = 100*(aep_helixOpt_0std_2deg-aep_baseline_0std_2deg)/aep_baseline_0std_2deg
# aep_gain_helixOpt_0std_5deg = 100*(aep_helixOpt_0std_5deg-aep_baseline_0std_5deg)/aep_baseline_0std_5deg

# print('Sigma = 0 -----------------------------------------------------')
# print('Mixed control ------------------------------------') 
# print(f'1 deg: AEP gain = {aep_gain_mixedOpt_0std_1deg}')
# print(f'2 deg: AEP gain = {aep_gain_mixedOpt_0std_2deg}')
# print(f'5 deg: AEP gain = {aep_gain_mixedOpt_0std_5deg}')
# print('Yaw control ------------------------------------')
# print(f'1 deg: AEP gain = {aep_gain_yawOpt_0std_1deg}')
# print(f'2 deg: AEP gain = {aep_gain_yawOpt_0std_2deg}')
# print(f'5 deg: AEP gain = {aep_gain_yawOpt_0std_5deg}')
# print('Helix control ------------------------------------')
# print(f'1 deg: AEP gain = {aep_gain_helixOpt_0std_1deg}')
# print(f'2 deg: AEP gain = {aep_gain_helixOpt_0std_2deg}')
# print(f'5 deg: AEP gain = {aep_gain_helixOpt_0std_5deg}')
# print('---------------------------------------------------------------')


# #%%

# savefig = False
# # name_path = r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\HKN_corner\\'
# colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']


# # calculate power gain - sigma=0 =====================================================================================================

# p_baseline_0std_mat_1deg = np.sum(simres_baseline_0std_1deg.Power.values,axis=(0))
# p_baseline_0std_mat_2deg = np.sum(simres_baseline_0std_2deg.Power.values,axis=(0))
# p_baseline_0std_mat_5deg = np.sum(simres_baseline_0std_5deg.Power.values,axis=(0))

# p_mixedOpt_0std_mat_1deg = np.sum(simres_mixedOpt_0std_1deg.Power.values,axis=(0))
# p_mixedOpt_0std_mat_2deg = np.sum(simres_mixedOpt_0std_2deg.Power.values,axis=(0))
# p_mixedOpt_0std_mat_5deg = np.sum(simres_mixedOpt_0std_5deg.Power.values,axis=(0))

# p_yawOpt_0std_mat_1deg = np.sum(simres_yawOpt_0std_1deg.Power.values,axis=(0))
# p_yawOpt_0std_mat_2deg = np.sum(simres_yawOpt_0std_2deg.Power.values,axis=(0))
# p_yawOpt_0std_mat_5deg = np.sum(simres_yawOpt_0std_5deg.Power.values,axis=(0))

# p_helixOpt_0std_mat_1deg = np.sum(simres_helixOpt_0std_1deg.Power.values,axis=(0))
# p_helixOpt_0std_mat_2deg = np.sum(simres_helixOpt_0std_2deg.Power.values,axis=(0))
# p_helixOpt_0std_mat_5deg = np.sum(simres_helixOpt_0std_5deg.Power.values,axis=(0))

# p_gain_mixedOpt_0std_mat_1deg = 100*(p_mixedOpt_0std_mat_1deg-p_baseline_0std_mat_1deg)/p_baseline_0std_mat_1deg
# p_gain_mixedOpt_0std_mat_2deg = 100*(p_mixedOpt_0std_mat_2deg-p_baseline_0std_mat_2deg)/p_baseline_0std_mat_2deg
# p_gain_mixedOpt_0std_mat_5deg = 100*(p_mixedOpt_0std_mat_5deg-p_baseline_0std_mat_5deg)/p_baseline_0std_mat_5deg

# p_gain_yawOpt_0std_mat_1deg = 100*(p_yawOpt_0std_mat_1deg-p_baseline_0std_mat_1deg)/p_baseline_0std_mat_1deg
# p_gain_yawOpt_0std_mat_2deg = 100*(p_yawOpt_0std_mat_2deg-p_baseline_0std_mat_2deg)/p_baseline_0std_mat_2deg
# p_gain_yawOpt_0std_mat_5deg = 100*(p_yawOpt_0std_mat_5deg-p_baseline_0std_mat_5deg)/p_baseline_0std_mat_5deg

# p_gain_helixOpt_0std_mat_1deg = 100*(p_helixOpt_0std_mat_1deg-p_baseline_0std_mat_1deg)/p_baseline_0std_mat_1deg
# p_gain_helixOpt_0std_mat_2deg = 100*(p_helixOpt_0std_mat_2deg-p_baseline_0std_mat_2deg)/p_baseline_0std_mat_2deg
# p_gain_helixOpt_0std_mat_5deg = 100*(p_helixOpt_0std_mat_5deg-p_baseline_0std_mat_5deg)/p_baseline_0std_mat_5deg



# ws_ind_array = np.array([7])
# for i in np.arange(len(ws_ind_array)):
#     ws_ind = ws_ind_array[i]
#     plt.figure()
#     plt.title(f'Mixed control (std=0deg) - Wind speed: {ws_array[ws_ind]} m/s')
#     plt.plot(wd_array,p_gain_mixedOpt_0std_mat_1deg[:,ws_ind],label='1 deg',c=colors[0])
#     plt.plot(wd_array_2deg,p_gain_mixedOpt_0std_mat_2deg[:,ws_ind],label='2 deg',c=colors[2])
#     plt.plot(wd_array_5deg,p_gain_mixedOpt_0std_mat_5deg[:,ws_ind],label='5 deg',c=colors[4])
#     plt.legend()
#     plt.xlabel('Wind direction [deg]')
#     plt.ylabel('Power gain [%]')
#     # if savefig: plt.savefig(name_path+'HKNcorner_power_gain_0std.svg',format='svg')
#     plt.show()

# ws_ind_array = np.array([7])
# for i in np.arange(len(ws_ind_array)):
#     ws_ind = ws_ind_array[i]
#     plt.figure()
#     plt.title(f'Yaw control (std=0deg) - Wind speed: {ws_array[ws_ind]} m/s')
#     plt.plot(wd_array,p_gain_yawOpt_0std_mat_1deg[:,ws_ind],label='1 deg',c=colors[0])
#     plt.plot(wd_array_2deg,p_gain_yawOpt_0std_mat_2deg[:,ws_ind],label='2 deg',c=colors[2])
#     plt.plot(wd_array_5deg,p_gain_yawOpt_0std_mat_5deg[:,ws_ind],label='5 deg',c=colors[4])
#     plt.legend()
#     plt.xlabel('Wind direction [deg]')
#     plt.ylabel('Power gain [%]')
#     # if savefig: plt.savefig(name_path+'HKNcorner_power_gain_0std.svg',format='svg')
#     plt.show()

# ws_ind_array = np.array([7])
# for i in np.arange(len(ws_ind_array)):
#     ws_ind = ws_ind_array[i]
#     plt.figure()
#     plt.title(f'Helix control (std=0deg) - Wind speed: {ws_array[ws_ind]} m/s')
#     plt.plot(wd_array,p_gain_helixOpt_0std_mat_1deg[:,ws_ind],label='1 deg',c=colors[0])
#     plt.plot(wd_array_2deg,p_gain_helixOpt_0std_mat_2deg[:,ws_ind],label='2 deg',c=colors[2])
#     plt.plot(wd_array_5deg,p_gain_helixOpt_0std_mat_5deg[:,ws_ind],label='5 deg',c=colors[4])
#     plt.legend()
#     plt.xlabel('Wind direction [deg]')
#     plt.ylabel('Power gain [%]')
#     # if savefig: plt.savefig(name_path+'HKNcorner_power_gain_0std.svg',format='svg')
#     plt.show()





#%% ANALYZE EFFECT OF SAMPLING FOR UNCERTAINTY ======================================================================
# ===================================================================================================


# def optimalControl_HKNwt0(wd_t_array,
#                           ws_t_array,
#                           n_values,
#                           sigma,
#                           optimize_yaw,
#                           optimize_helix):

#     yaw_opt_wt0 = np.zeros(len(wd_t_array))
#     helix_amp_opt_wt0 = np.zeros(len(ws_t_array))
    
    
#     for i in np.arange(len(wd_t_array)):
        
#         def power_HKN(yaw_wt0,helix_amp_wt0):
#             wd_t = np.array([wd_t_array[i]])
#             ws_t = np.array([ws_t_array[i]])
#             yaw_t = np.zeros(len(x))[:,na,na]
#             yaw_t[0,0,0] = yaw_wt0
#             helix_amp_t = np.zeros(len(x))[:,na,na]
#             helix_amp_t[0,0,0] = helix_amp_wt0
#             p = calculatePower_withUncertainty(wfm,x,y,wd_t,ws_t,yaw_t,helix_amp_t,sigma=sigma,n=n_values)
#             return p
            
#         optimizer = WFFC_Optimizer_SR(x = np.array([x[0]]),
#                                       y = np.array([y[0]]),
#                                       wd = wd_t_array[i],
#                                       f_obj = power_HKN,
#                                       yaw_max = 30.,
#                                       helix_amp_max = 5.,
#                                       n_step = 3,
#                                       n_values = 5,
#                                       optimize_yaw = optimize_yaw,
#                                       optimize_helix_amp = optimize_helix)
#         optimizer.optimize()
#         yaw_opt_wt0[i] = optimizer.yaw_opt
#         helix_amp_opt_wt0[i] = optimizer.helix_amp_opt

#     return yaw_opt_wt0,helix_amp_opt_wt0



# wd_t_array = np.arange(170,230,1)
# ws_t_array = np.ones(len(wd_t_array))*8.



# sigma = 2.5

# # n = 5 -------------------------------------------------------------------------

# n_values = 5

# # mixed operation (around 7.5 min)
# t = time.time()
# yaw_mixedOpt_sigma25_n5,helix_amp_mixedOpt_sigma25_n5 = optimalControl_HKNwt0(wd_t_array,ws_t_array,n_values,sigma,optimize_yaw=True,optimize_helix=True)
# print(f'Optimization completed (mixed, s={sigma}, n={n_values}) - Time: {time.time()-t}')

# # yaw operation
# t = time.time()
# yaw_yawOpt_sigma25_n5,helix_amp_yawOpt_sigma25_n5 = optimalControl_HKNwt0(wd_t_array,ws_t_array,n_values,sigma,optimize_yaw=True,optimize_helix=False)
# print(f'Optimization completed (mixed, s={sigma}, n={n_values}) - Time: {time.time()-t}')

# # helix operation
# t = time.time()
# yaw_helixOpt_sigma25_n5,helix_amp_helixOpt_sigma25_n5 = optimalControl_HKNwt0(wd_t_array,ws_t_array,n_values,sigma,optimize_yaw=False,optimize_helix=True)
# print(f'Optimization completed (mixed, s={sigma}, n={n_values}) - Time: {time.time()-t}')


# # n = 7 -------------------------------------------------------------------------

# n_values = 7

# # mixed operation
# t = time.time()
# yaw_mixedOpt_sigma25_n7,helix_amp_mixedOpt_sigma25_n7 = optimalControl_HKNwt0(wd_t_array,ws_t_array,n_values,sigma,optimize_yaw=True,optimize_helix=True)
# print(f'Optimization completed (mixed, s={sigma}, n={n_values}) - Time: {time.time()-t}')

# # yaw operation
# t = time.time()
# yaw_yawOpt_sigma25_n7,helix_amp_yawOpt_sigma25_n7 = optimalControl_HKNwt0(wd_t_array,ws_t_array,n_values,sigma,optimize_yaw=True,optimize_helix=False)
# print(f'Optimization completed (mixed, s={sigma}, n={n_values}) - Time: {time.time()-t}')

# # helix operation
# t = time.time()
# yaw_helixOpt_sigma25_n7,helix_amp_helixOpt_sigma25_n7 = optimalControl_HKNwt0(wd_t_array,ws_t_array,n_values,sigma,optimize_yaw=False,optimize_helix=True)
# print(f'Optimization completed (mixed, s={sigma}, n={n_values}) - Time: {time.time()-t}')


# # n = 9 -------------------------------------------------------------------------

# n_values = 9

# # mixed operation
# t = time.time()
# yaw_mixedOpt_sigma25_n9,helix_amp_mixedOpt_sigma25_n9 = optimalControl_HKNwt0(wd_t_array,ws_t_array,n_values,sigma,optimize_yaw=True,optimize_helix=True)
# print(f'Optimization completed (mixed, s={sigma}, n={n_values}) - Time: {time.time()-t}')

# # yaw operation
# t = time.time()
# yaw_yawOpt_sigma25_n9,helix_amp_yawOpt_sigma25_n9 = optimalControl_HKNwt0(wd_t_array,ws_t_array,n_values,sigma,optimize_yaw=True,optimize_helix=False)
# print(f'Optimization completed (mixed, s={sigma}, n={n_values}) - Time: {time.time()-t}')

# # helix operation
# t = time.time()
# yaw_helixOpt_sigma25_n9,helix_amp_helixOpt_sigma25_n9 = optimalControl_HKNwt0(wd_t_array,ws_t_array,n_values,sigma,optimize_yaw=False,optimize_helix=True)
# print(f'Optimization completed (mixed, s={sigma}, n={n_values}) - Time: {time.time()-t}')


# #%% make plots


# plt.figure(figsize=(12,8))
# plt.title('Mixed operation - yaw')
# plt.plot(wd_t_array,yaw_mixedOpt_sigma25_n5,label='n=5',marker='.',c='r',alpha=0.5)
# plt.plot(wd_t_array,yaw_mixedOpt_sigma25_n7,label='n=7',marker='.',c='g',alpha=0.5)
# plt.plot(wd_t_array,yaw_mixedOpt_sigma25_n9,label='n=9',marker='.',c='b',alpha=0.5)
# plt.legend()
# plt.ylabel('Yaw angle turbine 0 [deg]')
# plt.xlabel('Wind direction [deg]')
# plt.plot()

# plt.figure(figsize=(12,8))
# plt.title('Mixed operation - helix')
# plt.plot(wd_t_array,helix_amp_mixedOpt_sigma25_n5,label='n=5',marker='.',c='r',alpha=0.5)
# plt.plot(wd_t_array,helix_amp_mixedOpt_sigma25_n7,label='n=7',marker='.',c='g',alpha=0.5)
# plt.plot(wd_t_array,helix_amp_mixedOpt_sigma25_n9,label='n=9',marker='.',c='b',alpha=0.5)
# plt.legend()
# plt.ylabel('Helix amplitude turbine 0 [deg]')
# plt.xlabel('Wind direction [deg]')
# plt.plot()

# plt.figure(figsize=(12,8))
# plt.title('Yaw operation')
# plt.plot(wd_t_array,yaw_yawOpt_sigma25_n5,label='n=5',marker='.',c='r',alpha=0.5)
# plt.plot(wd_t_array,yaw_yawOpt_sigma25_n7,label='n=7',marker='.',c='g',alpha=0.5)
# plt.plot(wd_t_array,yaw_yawOpt_sigma25_n9,label='n=9',marker='.',c='b',alpha=0.5)
# plt.legend()
# plt.ylabel('Yaw angle turbine 0 [deg]')
# plt.xlabel('Wind direction [deg]')
# plt.plot()

# plt.figure(figsize=(12,8))
# plt.title('Helix operation')
# plt.plot(wd_t_array,helix_amp_helixOpt_sigma25_n5,label='n=5',marker='.',c='r',alpha=0.5)
# plt.plot(wd_t_array,helix_amp_helixOpt_sigma25_n7,label='n=7',marker='.',c='g',alpha=0.5)
# plt.plot(wd_t_array,helix_amp_helixOpt_sigma25_n9,label='n=9',marker='.',c='b',alpha=0.5)
# plt.legend()
# plt.ylabel('Helix amplitude turbine 0 [deg]')
# plt.xlabel('Wind direction [deg]')
# plt.plot()


#%%
# test wake losses model

x = (hkn_wt_x-x_sub)*(diameter/diameter_hkn)
y = (hkn_wt_y-y_sub)*(diameter/diameter_hkn)
wd_array = np.arange(0,360,1)
ws_array = np.arange(3,26,1)
prob_mat = simres_baseline_0std.P.values
ws_rated = 11.


def calculate_aep(wfm,x,y,wd_array,ws_array,yaw_ilk):

    # baseline
    simres =  wfm(x,y,wd=wd_array,ws=ws_array,yaw=0,tilt=0,helix_amp=np.zeros((len(x),len(wd_array),len(ws_array))))
    aep = simres.aep().sum()
    aep0 = simres.aep(with_wake_loss=False).sum()

    # yaw
    simres_yaw =  wfm(x,y,wd=wd_array,ws=ws_array,yaw=yaw_ilk,tilt=0,helix_amp=np.zeros((len(x),len(wd_array),len(ws_array))))
    aep_yaw = simres_yaw.aep().sum()

    return aep,aep0,aep_yaw


def calculate_dxdy(x,y,k,wd,diameter):
    
    # extend dimensions of x and y
    x_mat_1 = np.tile(np.reshape(x,(len(x),1)),(1,len(x)))
    x_mat_2 = np.tile(np.reshape(x,(1,len(x))),(len(x),1))
    y_mat_1 = np.tile(np.reshape(y,(len(y),1)),(1,len(y)))
    y_mat_2 = np.tile(np.reshape(y,(1,len(y))),(len(y),1))

    # calculate dx and dy
    d = np.sqrt((x_mat_1-x_mat_2)**2+(y_mat_1-y_mat_2)**2)
    theta = np.arctan2(y_mat_2-y_mat_1,x_mat_2-x_mat_1)
    gamma = wd*(np.pi/180)-(3/2)*np.pi+theta
    dx = d*np.cos(gamma)
    dy = d*np.sin(gamma)
    
    # identify waked turbines
    #condition_waked = np.logical_and(dx>0, np.abs(dy)<=(diameter/2+k*dx))
    condition_waked = np.logical_and(dx>0, np.abs(dy)-diameter/2<=(diameter/2+k*dx))

    # calculate number of waked turbine
    n_t_waked = np.sum(condition_waked,1)
    
    # assign +inf to dx and dy of unwaked turbines
    dx_waked = np.ones(dx.shape)*np.inf
    dy_waked = np.ones(dy.shape)*np.inf
    dx_waked[condition_waked] = dx[condition_waked]
    dy_waked[condition_waked] = dy[condition_waked]
    
    
    # extarct dx,dy of the closest turbine (filter neighbour)
    d_waked = np.sqrt(dx_waked**2+dy_waked**2)
    wt_neigh = np.argmin(d_waked,1)
    wt_neigh_mat = np.tile(np.reshape(wt_neigh,(len(wt_neigh),1)),(1,len(wt_neigh)))
    wt_count_mat = np.tile(np.reshape(np.arange(0,len(wt_neigh)),(1,len(wt_neigh))),(len(wt_neigh),1))
    fil_neigh = wt_neigh_mat == wt_count_mat
    dx_neigh = dx[fil_neigh]
    dy_neigh = dy[fil_neigh]


    # filter values (only turbines whose wake affects other turbines)
    fil_wake = n_t_waked>0
    n_t_waked_fil = n_t_waked[fil_wake]
    dx_neigh_fil = dx_neigh[fil_wake]
    dy_neigh_fil = dy_neigh[fil_wake]
    dx_all_fil = np.reshape(dx_waked[np.tile(np.reshape(fil_wake,(len(fil_wake),1)),(1,len(x)))],(len(dx_neigh_fil),len(x)))
    dy_all_fil = np.reshape(dy_waked[np.tile(np.reshape(fil_wake,(len(fil_wake),1)),(1,len(y)))],(len(dy_neigh_fil),len(y)))
    
    #return dx,dy,condition_waked
    return fil_wake,dx_neigh_fil,dy_neigh_fil,n_t_waked_fil,dx_all_fil,dy_all_fil






def calculate_geomYaw_ExpCorr(x,
                              y,
                              wd,
                              ws,
                              ws_rated,
                              wind_turbine,
                              wfm,
                              yaw_max = 21.846,             # coefficient tuned for IEA22MW
                              p_x = 4.889,                  # coefficient tuned for IEA22MW
                              p_y = 9.594,                  # coefficient tuned for IEA22MW
                              q_x = 5.820,                  # coefficient tuned for IEA22MW
                              q_y = 0.380,                  # coefficient tuned for IEA22MW
                              alpha_f_ws_eff = 0.150,       # coefficient tuned for IEA22MW
                              w_corr = 0.456                # coefficient tuned for IEA22MW
                              ):


    # extract diameter from the turbine
    diameter = wind_turbine.diameter()
    
    # set entraintment constant
    k = 0.1
    
    # initialize geometric yaw array
    yaw_array = np.zeros((len(x),len(wd),len(ws)))
    
    # calculate effective wind speed for all turbines, wind directions and wind speeds
    ws_eff_ilk = wfm(x,y,wd=wd,ws=ws,tilt=0,yaw=0).WS_eff_ilk
    
    # iterate for each wind direction
    for wd_ind in np.arange(0,len(wd)):
    
        # calculate dx and dy for the nearest waked turbine for one wind direction
        fil_wake,dx_neigh,dy_neigh,_,dx_all,dy_all = calculate_dxdy(x,y,k,wd[wd_ind],diameter) 
        dx_all_ext = np.tile(np.reshape(dx_all,(len(dx_neigh),len(x),1)),(1,1,len(ws)))
        dy_all_ext = np.tile(np.reshape(dy_all,(len(dy_neigh),len(y),1)),(1,1,len(ws)))
        
        # associate the effective wind speed of the nearest waked turbine each turbine
        ws_eff = ws_eff_ilk[:,wd_ind,:]
        ws_eff_mat = np.tile(np.reshape(ws_eff,(1,len(x),len(ws))),(len(x),1,1))
        ws_eff_mat_fil = ws_eff_mat[fil_wake,:,:]
        fil_ws_eff = np.ones((len(dx_neigh),len(x),len(ws)))
        
        # filter out turbines such that: ws_eff>ws_rated
        fil_ws_eff[ws_eff_mat_fil>ws_rated] = 0
        
        # filter out turbines such that: ws_eff<<ws_cut_in
        ws_cut_in = 4
        delta_cut_in = 2 # NEED PROPER TUNING
        fil_ws_eff[ws_eff_mat_fil<ws_cut_in-delta_cut_in] = 0
        
        # find the nearest turbine in the wake (after filtering)
        dx_all_ext[fil_ws_eff<1] = np.inf
        dy_all_ext[fil_ws_eff<1] = np.inf
        
        # calculate f_ws_eff
        #alpha_f_ws_eff = 0.3 # NEED PROPER TUNING
        f_ws_eff = 1-alpha_f_ws_eff*np.exp(np.reshape(ws_eff[fil_wake,:],(len(dx_neigh),1,len(ws)))-np.tile(np.reshape(ws,(1,1,len(ws))),(len(dx_neigh),1,1)))
        
        
        # initialize geometric yaw angles for one wind direction
        yaw_temp = np.zeros((len(x),1,len(ws)))
        eps = 1e-10
        
        
        # calculate the first approx of geometric angle
        
        # distinguish dy>0 and dy<0
        dy_all_ext_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_2 = -np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dx_all_ext_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dx_all_ext_2 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_1[dy_all_ext>=0] = dy_all_ext[dy_all_ext>=0]
        dy_all_ext_2[dy_all_ext<0] = dy_all_ext[dy_all_ext<0]
        dx_all_ext_1[dy_all_ext>=0] = dx_all_ext[dy_all_ext>=0]
        dx_all_ext_2[dy_all_ext<0] = dx_all_ext[dy_all_ext<0]
        
        # calculate first approx for both dy>0 and dy<0
        yaw_geom_mat_1 = np.sign(dy_all_ext_1+eps)*yaw_max*((1+p_x)/(p_x+np.e**(dx_all_ext/(q_x*diameter))))*((1+p_y)/(p_y+np.e**(np.abs(dy_all_ext_1)/(q_y*diameter))))
        yaw_geom_mat_2 = np.sign(dy_all_ext_2+eps)*yaw_max*((1+p_x)/(p_x+np.e**(dx_all_ext/(q_x*diameter))))*((1+p_y)/(p_y+np.e**(np.abs(dy_all_ext_2)/(q_y*diameter))))
        yaw_geom_first_approx_1 = np.amax(yaw_geom_mat_1,axis=1)
        yaw_geom_first_approx_2 = np.amin(yaw_geom_mat_2,axis=1)
        
        # choose dominant influence (dy>0 or dy<0) and select the first approximation
        yaw_geom_first_approx = yaw_geom_first_approx_1
        yaw_geom_first_approx[yaw_geom_first_approx_1<np.abs(yaw_geom_first_approx_2)] = yaw_geom_first_approx_2[yaw_geom_first_approx_1<np.abs(yaw_geom_first_approx_2)]
        
        
        # extract dx and dy relevant for both dy>0 and dy<0
        
        ind_1 = np.argmax(yaw_geom_mat_1,axis=1,keepdims=True)
        ind_2 = np.argmin(yaw_geom_mat_2,axis=1,keepdims=True)
        fil_ind_1 = np.tile(ind_1,(1,len(x),1))==np.tile(np.reshape(np.arange(len(x)),(1,len(x),1)),(len(dx_neigh),1,len(ws)))
        fil_ind_2 = np.tile(ind_2,(1,len(x),1))==np.tile(np.reshape(np.arange(len(x)),(1,len(x),1)),(len(dx_neigh),1,len(ws)))
        
        dx_all_ext_relevant_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dx_all_ext_relevant_2 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_relevant_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_relevant_2 = -np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        
        dx_all_ext_relevant_1[fil_ind_1] = dx_all_ext_1[fil_ind_1]
        dx_all_ext_relevant_2[fil_ind_2] = dx_all_ext_2[fil_ind_2]
        dy_all_ext_relevant_1[fil_ind_1] = dy_all_ext_1[fil_ind_1]
        dy_all_ext_relevant_2[fil_ind_2] = dy_all_ext_2[fil_ind_2]
        
        dx_ext_relevant_1 = np.amin(dx_all_ext_relevant_1,axis=1)
        dx_ext_relevant_2 = np.amin(dx_all_ext_relevant_2,axis=1)
        dy_ext_relevant_1 = np.amin(dy_all_ext_relevant_1,axis=1)
        dy_ext_relevant_2 = -np.amin(np.abs(dy_all_ext_relevant_2),axis=1)
        
        dx_ext_correction = dx_ext_relevant_1.copy()
        dx_ext_correction[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)] = dx_ext_relevant_2[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)]
        dy_ext_correction = dy_ext_relevant_1.copy()
        dy_ext_correction[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)] = dy_ext_relevant_2[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)]
        
        
        # apply correction considering next waked turbines
        c_t = wind_turbine.ct(np.tile(np.reshape(ws,(1,len(ws))),(len(dx_neigh),1)))
        delta_wd = -((c_t/2)*(np.sin(np.pi*yaw_geom_first_approx/180))*(np.cos(np.pi*yaw_geom_first_approx/180))**2)/((1+k*(dx_ext_correction/diameter))**2)    
            
        fil_correction = (np.abs(dx_ext_correction)!=np.inf) & (np.abs(dy_ext_correction)!=np.inf)
        dx_ext_correction_new = np.inf*np.ones((len(dx_neigh),len(ws)))
        dy_ext_correction_new = np.inf*np.ones((len(dx_neigh),len(ws)))
        dx_ext_correction_new[fil_correction] = dx_ext_correction[fil_correction]*np.cos(delta_wd[fil_correction])+dy_ext_correction[fil_correction]*np.sin(delta_wd[fil_correction])
        dy_ext_correction_new[fil_correction] = -dx_ext_correction[fil_correction]*np.sin(delta_wd[fil_correction])+dy_ext_correction[fil_correction]*np.cos(delta_wd[fil_correction])
        
        yaw_geom_correction = np.sign(dy_ext_correction_new+eps)*yaw_max*((1+p_x)/(p_x+np.e**(dx_ext_correction_new/(q_x*diameter))))*((1+p_y)/(p_y+np.e**(np.abs(dy_ext_correction_new)/(q_y*diameter))))
        yaw_geom_temp = yaw_geom_first_approx+w_corr*yaw_geom_correction
        
        
        # apply correction effective wind speed and set limits
        yaw_geom = np.minimum(np.maximum(f_ws_eff*np.reshape(yaw_geom_temp,(len(dx_neigh),1,len(ws))),-yaw_max),yaw_max)
        
        # assign values
        yaw_temp[fil_wake,:,:] = yaw_geom
        yaw_array[:,wd_ind,:] = np.reshape(yaw_temp,(len(x),len(ws)))
    
    return yaw_array


def calculate_geomYaw_ExpCorr_EmpGauss(x,
                              y,
                              wd,
                              ws,
                              ws_rated,
                              wind_turbine,
                              wfm,
                              yaw_max = 21.846,             # coefficient tuned for IEA22MW
                              p_x = 4.889,                  # coefficient tuned for IEA22MW
                              p_y = 9.594,                  # coefficient tuned for IEA22MW
                              q_x = 5.820,                  # coefficient tuned for IEA22MW
                              q_y = 0.380,                  # coefficient tuned for IEA22MW
                              alpha_f_ws_eff = 0.150,       # coefficient tuned for IEA22MW
                              w_corr = 0.456                # coefficient tuned for IEA22MW
                              ):


    # extract diameter from the turbine
    diameter = wind_turbine.diameter()
    
    # set entraintment constant
    k = 0.1
    
    # initialize geometric yaw array
    yaw_array = np.zeros((len(x),len(wd),len(ws)))
    
    # calculate effective wind speed for all turbines, wind directions and wind speeds
    ws_eff_ilk = wfm(x,y,wd=wd,ws=ws,tilt=0,yaw=0,helix_amp=np.zeros((len(x),len(wd),len(ws)))).WS_eff_ilk
    
    # iterate for each wind direction
    for wd_ind in np.arange(0,len(wd)):
    
        # calculate dx and dy for the nearest waked turbine for one wind direction
        fil_wake,dx_neigh,dy_neigh,_,dx_all,dy_all = calculate_dxdy(x,y,k,wd[wd_ind],diameter) 
        dx_all_ext = np.tile(np.reshape(dx_all,(len(dx_neigh),len(x),1)),(1,1,len(ws)))
        dy_all_ext = np.tile(np.reshape(dy_all,(len(dy_neigh),len(y),1)),(1,1,len(ws)))
        
        # associate the effective wind speed of the nearest waked turbine each turbine
        ws_eff = ws_eff_ilk[:,wd_ind,:]
        ws_eff_mat = np.tile(np.reshape(ws_eff,(1,len(x),len(ws))),(len(x),1,1))
        ws_eff_mat_fil = ws_eff_mat[fil_wake,:,:]
        fil_ws_eff = np.ones((len(dx_neigh),len(x),len(ws)))
        
        # filter out turbines such that: ws_eff>ws_rated
        fil_ws_eff[ws_eff_mat_fil>ws_rated] = 0
        
        # filter out turbines such that: ws_eff<<ws_cut_in
        ws_cut_in = 4
        delta_cut_in = 2 # NEED PROPER TUNING
        fil_ws_eff[ws_eff_mat_fil<ws_cut_in-delta_cut_in] = 0
        
        # find the nearest turbine in the wake (after filtering)
        dx_all_ext[fil_ws_eff<1] = np.inf
        dy_all_ext[fil_ws_eff<1] = np.inf
        
        # calculate f_ws_eff
        #alpha_f_ws_eff = 0.3 # NEED PROPER TUNING
        f_ws_eff = 1-alpha_f_ws_eff*np.exp(np.reshape(ws_eff[fil_wake,:],(len(dx_neigh),1,len(ws)))-np.tile(np.reshape(ws,(1,1,len(ws))),(len(dx_neigh),1,1)))
        
        
        # initialize geometric yaw angles for one wind direction
        yaw_temp = np.zeros((len(x),1,len(ws)))
        eps = 1e-10
        
        
        # calculate the first approx of geometric angle
        
        # distinguish dy>0 and dy<0
        dy_all_ext_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_2 = -np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dx_all_ext_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dx_all_ext_2 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_1[dy_all_ext>=0] = dy_all_ext[dy_all_ext>=0]
        dy_all_ext_2[dy_all_ext<0] = dy_all_ext[dy_all_ext<0]
        dx_all_ext_1[dy_all_ext>=0] = dx_all_ext[dy_all_ext>=0]
        dx_all_ext_2[dy_all_ext<0] = dx_all_ext[dy_all_ext<0]
        
        # calculate first approx for both dy>0 and dy<0
        yaw_geom_mat_1 = np.sign(dy_all_ext_1+eps)*yaw_max*((1+p_x)/(p_x+np.e**(dx_all_ext/(q_x*diameter))))*((1+p_y)/(p_y+np.e**(np.abs(dy_all_ext_1)/(q_y*diameter))))
        yaw_geom_mat_2 = np.sign(dy_all_ext_2+eps)*yaw_max*((1+p_x)/(p_x+np.e**(dx_all_ext/(q_x*diameter))))*((1+p_y)/(p_y+np.e**(np.abs(dy_all_ext_2)/(q_y*diameter))))
        yaw_geom_first_approx_1 = np.amax(yaw_geom_mat_1,axis=1)
        yaw_geom_first_approx_2 = np.amin(yaw_geom_mat_2,axis=1)
        
        # choose dominant influence (dy>0 or dy<0) and select the first approximation
        yaw_geom_first_approx = yaw_geom_first_approx_1
        yaw_geom_first_approx[yaw_geom_first_approx_1<np.abs(yaw_geom_first_approx_2)] = yaw_geom_first_approx_2[yaw_geom_first_approx_1<np.abs(yaw_geom_first_approx_2)]
        
        
        # extract dx and dy relevant for both dy>0 and dy<0
        
        ind_1 = np.argmax(yaw_geom_mat_1,axis=1,keepdims=True)
        ind_2 = np.argmin(yaw_geom_mat_2,axis=1,keepdims=True)
        fil_ind_1 = np.tile(ind_1,(1,len(x),1))==np.tile(np.reshape(np.arange(len(x)),(1,len(x),1)),(len(dx_neigh),1,len(ws)))
        fil_ind_2 = np.tile(ind_2,(1,len(x),1))==np.tile(np.reshape(np.arange(len(x)),(1,len(x),1)),(len(dx_neigh),1,len(ws)))
        
        dx_all_ext_relevant_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dx_all_ext_relevant_2 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_relevant_1 = np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        dy_all_ext_relevant_2 = -np.inf*np.ones((len(dx_neigh),len(x),len(ws)))
        
        dx_all_ext_relevant_1[fil_ind_1] = dx_all_ext_1[fil_ind_1]
        dx_all_ext_relevant_2[fil_ind_2] = dx_all_ext_2[fil_ind_2]
        dy_all_ext_relevant_1[fil_ind_1] = dy_all_ext_1[fil_ind_1]
        dy_all_ext_relevant_2[fil_ind_2] = dy_all_ext_2[fil_ind_2]
        
        dx_ext_relevant_1 = np.amin(dx_all_ext_relevant_1,axis=1)
        dx_ext_relevant_2 = np.amin(dx_all_ext_relevant_2,axis=1)
        dy_ext_relevant_1 = np.amin(dy_all_ext_relevant_1,axis=1)
        dy_ext_relevant_2 = -np.amin(np.abs(dy_all_ext_relevant_2),axis=1)
        
        dx_ext_correction = dx_ext_relevant_1.copy()
        dx_ext_correction[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)] = dx_ext_relevant_2[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)]
        dy_ext_correction = dy_ext_relevant_1.copy()
        dy_ext_correction[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)] = dy_ext_relevant_2[yaw_geom_first_approx_1>=np.abs(yaw_geom_first_approx_2)]
        
        
        # apply correction considering next waked turbines
        c_t = wind_turbine.ct(np.tile(np.reshape(ws,(1,len(ws))),(len(dx_neigh),1)))
        delta_wd = -((c_t/2)*(np.sin(np.pi*yaw_geom_first_approx/180))*(np.cos(np.pi*yaw_geom_first_approx/180))**2)/((1+k*(dx_ext_correction/diameter))**2)    
            
        fil_correction = (np.abs(dx_ext_correction)!=np.inf) & (np.abs(dy_ext_correction)!=np.inf)
        dx_ext_correction_new = np.inf*np.ones((len(dx_neigh),len(ws)))
        dy_ext_correction_new = np.inf*np.ones((len(dx_neigh),len(ws)))
        dx_ext_correction_new[fil_correction] = dx_ext_correction[fil_correction]*np.cos(delta_wd[fil_correction])+dy_ext_correction[fil_correction]*np.sin(delta_wd[fil_correction])
        dy_ext_correction_new[fil_correction] = -dx_ext_correction[fil_correction]*np.sin(delta_wd[fil_correction])+dy_ext_correction[fil_correction]*np.cos(delta_wd[fil_correction])
        
        yaw_geom_correction = np.sign(dy_ext_correction_new+eps)*yaw_max*((1+p_x)/(p_x+np.e**(dx_ext_correction_new/(q_x*diameter))))*((1+p_y)/(p_y+np.e**(np.abs(dy_ext_correction_new)/(q_y*diameter))))
        yaw_geom_temp = yaw_geom_first_approx+w_corr*yaw_geom_correction
        
        
        # apply correction effective wind speed and set limits
        yaw_geom = np.minimum(np.maximum(f_ws_eff*np.reshape(yaw_geom_temp,(len(dx_neigh),1,len(ws))),-yaw_max),yaw_max)
        
        # assign values
        yaw_temp[fil_wake,:,:] = yaw_geom
        yaw_array[:,wd_ind,:] = np.reshape(yaw_temp,(len(x),len(ws)))
    
    return yaw_array




#%%
# calculate AEP EmpGauss - no uncertinaty
ws_rated = 11.
gyaw_ilk_Emp = calculate_geomYaw_ExpCorr(x,y,wd_array,ws_array,ws_rated,wind_turbine,wfm)
aep_Emp,aep0_Emp,aep_yaw_Emp = calculate_aep(wfm,x,y,wd_array,ws_array,gyaw_ilk_Emp)

print(f'AEP: {aep_Emp}')
print(f'AEP (no losses): {aep0_Emp}')
print(f'AEP (wake steering): {aep_yaw_Emp}')


#%%
# calculate AEP Bastankhah wuth geom yaw - with uncertainty [0, 2.5] deg

from py_wake.deficit_models import BastankhahGaussianDeficit
from py_wake.deflection_models import JimenezWakeDeflection
from py_wake.wind_farm_models import PropagateDownwind

from py_wake_helix.py_wake_helix_tools import calculatePmat_withUncertainty_Bast

wfm_Bast = PropagateDownwind(hkn_site_scaled, wind_turbine,
                                            wake_deficitModel=BastankhahGaussianDeficit(),
                                            superpositionModel=SquaredSum(),
                                            deflectionModel=JimenezWakeDeflection(),
                                            turbulenceModel=None,
                                            rotorAvgModel=GaussianOverlapAvgModel())

t = time.time()

gyaw_ilk_Bast = calculate_geomYaw_ExpCorr(x,y,wd_array,ws_array,ws_rated,wind_turbine,wfm_Bast)

p_ilk_Bast_baseline_0std = calculatePmat_withUncertainty_Bast(wfm_Bast,x,y,wd_array,ws_array,np.zeros_like(gyaw_ilk_Bast),sigma=0.,n=1)
p_ilk_Bast_gyaw_0std = calculatePmat_withUncertainty_Bast(wfm_Bast,x,y,wd_array,ws_array,gyaw_ilk_Bast,sigma=0.,n=1)

p_ilk_Bast_baseline_25std = calculatePmat_withUncertainty_Bast(wfm_Bast,x,y,wd_array,ws_array,np.zeros_like(gyaw_ilk_Bast),sigma=2.5,n=9)
p_ilk_Bast_gyaw_25std = calculatePmat_withUncertainty_Bast(wfm_Bast,x,y,wd_array,ws_array,gyaw_ilk_Bast,sigma=2.5,n=9)


aep_baseline_Bast_0std = 8760*np.sum(p_ilk_Bast_baseline_0std*prob_mat)/1e9
aep_gyaw_Bast_0std = 8760*np.sum(p_ilk_Bast_gyaw_0std*prob_mat)/1e9

aep_baseline_Bast_25std = 8760*np.sum(p_ilk_Bast_baseline_25std*prob_mat)/1e9
aep_gyaw_Bast_25std = 8760*np.sum(p_ilk_Bast_gyaw_25std*prob_mat)/1e9

aep_gain_gyaw_Bast_0std = 100*(aep_gyaw_Bast_0std-aep_baseline_Bast_0std)/aep_baseline_Bast_0std
aep_gain_gyaw_Bast_25std = 100*(aep_gyaw_Bast_25std-aep_baseline_Bast_25std)/aep_baseline_Bast_25std

print(f'Bastankhah completed - Time: {time.time()-t}')

#%%
# calculate AEP EmpGauss wuth geom yaw - with uncertainty [0, 2.5] deg

t = time.time()

gyaw_ilk_EmpGauss = calculate_geomYaw_ExpCorr_EmpGauss(x,y,wd_array,ws_array,ws_rated,wind_turbine,wfm)

p_ilk_EmpGauss_baseline_0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,np.zeros_like(gyaw_ilk_EmpGauss),np.zeros_like(gyaw_ilk_EmpGauss),sigma=0.,n=1)
p_ilk_EmpGauss_gyaw_0std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,gyaw_ilk_EmpGauss,np.zeros_like(gyaw_ilk_EmpGauss),sigma=0.,n=1)

p_ilk_EmpGauss_baseline_25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,np.zeros_like(gyaw_ilk_EmpGauss),np.zeros_like(gyaw_ilk_EmpGauss),sigma=2.5,n=9)
p_ilk_EmpGauss_gyaw_25std = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,gyaw_ilk_EmpGauss,np.zeros_like(gyaw_ilk_EmpGauss),sigma=2.5,n=9)


aep_baseline_EmpGauss_0std = 8760*np.sum(p_ilk_EmpGauss_baseline_0std*prob_mat)/1e9
aep_gyaw_EmpGauss_0std = 8760*np.sum(p_ilk_EmpGauss_gyaw_0std*prob_mat)/1e9

aep_baseline_EmpGauss_25std = 8760*np.sum(p_ilk_EmpGauss_baseline_25std*prob_mat)/1e9
aep_gyaw_EmpGauss_25std = 8760*np.sum(p_ilk_EmpGauss_gyaw_25std*prob_mat)/1e9

aep_gain_gyaw_EmpGauss_0std = 100*(aep_gyaw_EmpGauss_0std-aep_baseline_EmpGauss_0std)/aep_baseline_EmpGauss_0std
aep_gain_gyaw_EmpGauss_25std = 100*(aep_gyaw_EmpGauss_25std-aep_baseline_EmpGauss_25std)/aep_baseline_EmpGauss_25std

print(f'EmpGauss completed - Time: {time.time()-t}')


#%%
# plots

colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']
xlabel_list = [r'$\sigma_{\theta}=0^\circ$',r'$\sigma_{\theta}=2.5^\circ$',r'$\sigma_{\theta}=5^\circ$']

aep_gain_gyaw_array = np.array([aep_gain_gyaw_EmpGauss_0std,aep_gain_gyaw_EmpGauss_25std])
aep_gain_gyaw_Bast_array = np.array([aep_gain_gyaw_Bast_0std,aep_gain_gyaw_Bast_25std])

bar_width = 0.2
x_plot = np.arange(len(xlabel_list))
offsets = np.array([-bar_width/2,bar_width/2])
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x_plot[:2] + offsets[0], aep_gain_gyaw_array, width=bar_width, color=colors[2], label='Geom Yaw (EmpGauss)')
ax.bar(x_plot[:2] + offsets[1], aep_gain_gyaw_Bast_array, width=bar_width, color=colors[2], label='Geom Yaw (Bastankhah)',hatch='////',edgecolor='white')
ax.set_xticks(x_plot[:2])
ax.set_xticklabels(xlabel_list[:2])
ax.set_ylabel('AEP gain [%]')
ax.legend()
if savefig: plt.savefig(name_path+'wfm_comparison_gyaw.pdf',format='pdf')
plt.show()



# %%
# CHECK INPUT SITE



from py_wake.site.shear import PowerShear

ds_hkn_scaled_corrected = xr.Dataset(
    data_vars={
        'Sector_frequency':(['x','y','wd'],hkn_site.ds['Sector_frequency'].values),
        'Weibull_A':(['x','y','wd'],hkn_site.ds['Weibull_A'].values*((170./115.)**0.1)),
        'Weibull_k':(['x','y','wd'],hkn_site.ds['Weibull_k'].values),
        'TI':0.04    
        },
    coords={
        'x':(hkn_site.ds['x'].values-x_sub)*(diameter/diameter_hkn),
        'y':(hkn_site.ds['y'].values-y_sub)*(diameter/diameter_hkn),
        'wd':hkn_site.ds['wd'].values
        }
    )
hkn_site_scaled_corrected = XRSite(ds_hkn_scaled_corrected)#,shear=PowerShear(h_ref=115, alpha=.1))

# define wind farm model (EMPGAUSS - OPT COEFF.)
wfm_corrected = PropagateDownwind_helix(hkn_site_scaled_corrected, wind_turbine,
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

wd_array = np.arange(0,360,1)
ws_array = np.arange(3,26,1)
yaw = np.zeros((len(x),len(wd_array),len(ws_array)))
helix_amp = np.zeros((len(x),len(wd_array),len(ws_array)))

simres = wfm(x,y,wd=wd_array,ws=ws_array,yaw=yaw,helix_amp=helix_amp,tilt=0)
simres_corrected = wfm_corrected(x,y,wd=wd_array,ws=ws_array,yaw=yaw,helix_amp=helix_amp,tilt=0)
#%%
power_mat = simres.Power.values
prob_mat = simres.P.values

power_mat_corrected = simres_corrected.Power.values
prob_mat_corrected = simres_corrected.P.values

print(np.max(np.abs(power_mat-power_mat_corrected)))
print(np.max(np.abs(prob_mat-prob_mat_corrected)))
# %%
# plot presentation wind europe
savefig = False

plt.figure()
plt.scatter(x, y,c='k')
plt.axis('equal')
plt.axis('off')
plt.gca().set_facecolor('none') 
plt.gcf().patch.set_alpha(0)
if savefig: plt.savefig("xy_HKN.png", transparent=True,bbox_inches='tight')
plt.show()
