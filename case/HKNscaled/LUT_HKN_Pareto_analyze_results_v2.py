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
# DEFINE CASE STUDY

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

#%%
# FUNCTIONS

def extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array):
    yaw_mat = np.zeros((len(x),len(wd_array),len(ws_array)))
    helix_amp_mat = np.zeros((len(x),len(wd_array),len(ws_array)))
    for i in np.arange(len(data_list)):
        yaw_mat_temp = data_list[i]['yaw_opt']
        helix_amp_mat_temp = data_list[i]['helix_amp_opt']
        yaw_mat[:,:,ws_ind_temp_list[i]] = yaw_mat_temp
        helix_amp_mat[:,:,ws_ind_temp_list[i]] = helix_amp_mat_temp
    return yaw_mat,helix_amp_mat



#%%
# IMPORT DATA - helix_amp_max = 5 deg
# around 13 min per weight value

#weight_coefficient_array = ['0','0.25','0.5','0.75','1','2.5','5','7.5','10']
weight_coefficient_array = ['0000','0025','0050','0075','0100','0250','0500','0750','1000']

wd_array = np.arange(0,360,1)
ws_array = np.arange(3,26,1)

sigma = 2.5
n_values = 9


# baseline --------------------
p_mat_baseline = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,np.zeros((len(x),len(wd_array),len(ws_array))),np.zeros((len(x),len(wd_array),len(ws_array))),sigma=sigma,n=n_values)
simres_baseline_0std = wfm(x,y,wd=wd_array,ws=ws_array,yaw=np.zeros((len(x),len(wd_array),len(ws_array))),tilt=0,helix_amp=np.zeros((len(x),len(wd_array),len(ws_array))))


# yaw control ------------------

p_mat_yawControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
yaw_mat_yawControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
helix_amp_mat_yawControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))

t = time.time()

for w_ind in np.arange(len(weight_coefficient_array)):

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_yaw_s25_ws3to5_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_1 = pickle.load(f)
    ws_ind_temp_1 = np.array([0,1,2])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_yaw_s25_ws6to8_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_2 = pickle.load(f)
    ws_ind_temp_2 = np.array([3,4,5])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_yaw_s25_ws9to11_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_3 = pickle.load(f)
    ws_ind_temp_3 = np.array([6,7,8])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_yaw_s25_ws12to14_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_4 = pickle.load(f)
    ws_ind_temp_4 = np.array([9,10,11])

    data_list = [data_temp_1,data_temp_2,data_temp_3,data_temp_4]
    ws_ind_temp_list = [ws_ind_temp_1,ws_ind_temp_2,ws_ind_temp_3,ws_ind_temp_4]

    yaw_mat,helix_amp_mat = extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array)
    p_mat = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mat,helix_amp_mat,sigma=sigma,n=n_values)

    p_mat_yawControl_ampMax5_Pareto[w_ind,:,:,:] = p_mat
    yaw_mat_yawControl_ampMax5_Pareto[w_ind,:,:,:] = yaw_mat
    helix_amp_mat_yawControl_ampMax5_Pareto[w_ind,:,:,:] = helix_amp_mat

print(f'Time for extracting the data - Yaw control: \t {time.time()-t}')


# helix control ------------------

p_mat_helixControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
yaw_mat_helixControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
helix_amp_mat_helixControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))

t = time.time()

for w_ind in np.arange(len(weight_coefficient_array)):

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_helix_s25_ws3to5_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_1 = pickle.load(f)
    ws_ind_temp_1 = np.array([0,1,2])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_helix_s25_ws6to8_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_2 = pickle.load(f)
    ws_ind_temp_2 = np.array([3,4,5])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_helix_s25_ws9to11_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_3 = pickle.load(f)
    ws_ind_temp_3 = np.array([6,7,8])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_helix_s25_ws12to14_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_4 = pickle.load(f)
    ws_ind_temp_4 = np.array([9,10,11])

    data_list = [data_temp_1,data_temp_2,data_temp_3,data_temp_4]
    ws_ind_temp_list = [ws_ind_temp_1,ws_ind_temp_2,ws_ind_temp_3,ws_ind_temp_4]

    yaw_mat,helix_amp_mat = extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array)
    p_mat = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mat,helix_amp_mat,sigma=sigma,n=n_values)

    p_mat_helixControl_ampMax5_Pareto[w_ind,:,:,:] = p_mat
    yaw_mat_helixControl_ampMax5_Pareto[w_ind,:,:,:] = yaw_mat
    helix_amp_mat_helixControl_ampMax5_Pareto[w_ind,:,:,:] = helix_amp_mat

print(f'Time for extracting the data - Helix control: \t {time.time()-t}')


# mixed control ------------------

p_mat_mixedControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
yaw_mat_mixedControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
helix_amp_mat_mixedControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))

t = time.time()

for w_ind in np.arange(len(weight_coefficient_array)):

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_mixed_s25_ws3to5_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_1 = pickle.load(f)
    ws_ind_temp_1 = np.array([0,1,2])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_mixed_s25_ws6to8_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_2 = pickle.load(f)
    ws_ind_temp_2 = np.array([3,4,5])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_mixed_s25_ws9to11_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_3 = pickle.load(f)
    ws_ind_temp_3 = np.array([6,7,8])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_mixed_s25_ws12to14_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_4 = pickle.load(f)
    ws_ind_temp_4 = np.array([9,10,11])

    data_list = [data_temp_1,data_temp_2,data_temp_3,data_temp_4]
    ws_ind_temp_list = [ws_ind_temp_1,ws_ind_temp_2,ws_ind_temp_3,ws_ind_temp_4]

    yaw_mat,helix_amp_mat = extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array)
    p_mat = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mat,helix_amp_mat,sigma=sigma,n=n_values)

    p_mat_mixedControl_ampMax5_Pareto[w_ind,:,:,:] = p_mat
    yaw_mat_mixedControl_ampMax5_Pareto[w_ind,:,:,:] = yaw_mat
    helix_amp_mat_mixedControl_ampMax5_Pareto[w_ind,:,:,:] = helix_amp_mat

print(f'Time for extracting the data - Mixed control: \t {time.time()-t}')



# %%
# CALCULATE AEP GAINS - helix_amp_max = 5 deg

# extract porbability of flow cases (per turbine)
prob_mat = simres_baseline_0std.P.values

# calculate AEP baseline
aep_baseline = 8760*np.sum(p_mat_baseline*prob_mat)/1e9

# calculate AEP gains

aep_yawControl_ampMax5_Pareto = 8760*np.sum(p_mat_yawControl_ampMax5_Pareto*prob_mat[na,:,:,:],axis=(1,2,3))/1e9
aep_helixControl_ampMax5_Pareto = 8760*np.sum(p_mat_helixControl_ampMax5_Pareto*prob_mat[na,:,:,:],axis=(1,2,3))/1e9
aep_mixedControl_ampMax5_Pareto = 8760*np.sum(p_mat_mixedControl_ampMax5_Pareto*prob_mat[na,:,:,:],axis=(1,2,3))/1e9

aep_gain_yawControl_ampMax5_Pareto = 100*(aep_yawControl_ampMax5_Pareto-aep_baseline)/aep_baseline
aep_gain_helixControl_ampMax5_Pareto = 100*(aep_helixControl_ampMax5_Pareto-aep_baseline)/aep_baseline
aep_gain_mixedControl_ampMax5_Pareto = 100*(aep_mixedControl_ampMax5_Pareto-aep_baseline)/aep_baseline


# calculate percentage of operation

wffc_binary_mat_yawControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
wffc_binary_mat_helixControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
wffc_binary_mat_mixedControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))

wffc_binary_mat_yawControl_ampMax5_Pareto[(np.abs(yaw_mat_yawControl_ampMax5_Pareto)>0)|(helix_amp_mat_yawControl_ampMax5_Pareto>0)]=1
wffc_binary_mat_helixControl_ampMax5_Pareto[(np.abs(yaw_mat_helixControl_ampMax5_Pareto)>0)|(helix_amp_mat_helixControl_ampMax5_Pareto>0)]=1
wffc_binary_mat_mixedControl_ampMax5_Pareto[(np.abs(yaw_mat_mixedControl_ampMax5_Pareto)>0)|(helix_amp_mat_mixedControl_ampMax5_Pareto>0)]=1

perc_operation_i_yawControl_ampMax5_Pareto = 100*np.sum(wffc_binary_mat_yawControl_ampMax5_Pareto*prob_mat[na,:,:,:],axis=(2,3))
perc_operation_i_helixControl_ampMax5_Pareto = 100*np.sum(wffc_binary_mat_helixControl_ampMax5_Pareto*prob_mat[na,:,:,:],axis=(2,3))
perc_operation_i_mixedControl_ampMax5_Pareto = 100*np.sum(wffc_binary_mat_mixedControl_ampMax5_Pareto*prob_mat[na,:,:,:],axis=(2,3))

perc_operation_mean_yawControl_ampMax5_Pareto = np.mean(perc_operation_i_yawControl_ampMax5_Pareto,axis=(1))
perc_operation_mean_helixControl_ampMax5_Pareto = np.mean(perc_operation_i_helixControl_ampMax5_Pareto,axis=(1))
perc_operation_mean_mixedControl_ampMax5_Pareto = np.mean(perc_operation_i_mixedControl_ampMax5_Pareto,axis=(1))



#%%
# SAVE DATA TO PLOT PARETO - helix_amp_max = 5 deg

with open(f'data_HKN_Pareto_v3_5deg.pkl', 'wb') as f:
    pickle.dump({'aep_baseline' : aep_baseline,
                 'prob_mat' : prob_mat,
                 'aep_yawControl_ampMax5_Pareto' : aep_yawControl_ampMax5_Pareto,
                 'aep_helixControl_ampMax5_Pareto' : aep_helixControl_ampMax5_Pareto,
                 'aep_mixedControl_ampMax5_Pareto' : aep_mixedControl_ampMax5_Pareto,
                 'wffc_binary_mat_yawControl_ampMax5_Pareto' : wffc_binary_mat_yawControl_ampMax5_Pareto,
                 'wffc_binary_mat_helixControl_ampMax5_Pareto' : wffc_binary_mat_helixControl_ampMax5_Pareto,
                 'wffc_binary_mat_mixedControl_ampMax5_Pareto' : wffc_binary_mat_mixedControl_ampMax5_Pareto,
                 'perc_operation_i_yawControl_ampMax5_Pareto' : perc_operation_i_yawControl_ampMax5_Pareto,
                 'perc_operation_i_helixControl_ampMax5_Pareto' : perc_operation_i_helixControl_ampMax5_Pareto,
                 'perc_operation_i_mixedControl_ampMax5_Pareto' : perc_operation_i_mixedControl_ampMax5_Pareto,
                }, f)


#%%
# IMPORT DATA - helix_amp_max = 2.5 deg
# around 13 min per weight value

#weight_coefficient_array = ['0','0.25','0.5','0.75','1','2.5','5','7.5','10']
weight_coefficient_array = ['0000','0025','0050','0075','0100','0250','0500','0750','1000']

wd_array = np.arange(0,360,1)
ws_array = np.arange(3,26,1)

sigma = 2.5
n_values = 9


# baseline --------------------
p_mat_baseline = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,np.zeros((len(x),len(wd_array),len(ws_array))),np.zeros((len(x),len(wd_array),len(ws_array))),sigma=sigma,n=n_values)
simres_baseline_0std = wfm(x,y,wd=wd_array,ws=ws_array,yaw=np.zeros((len(x),len(wd_array),len(ws_array))),tilt=0,helix_amp=np.zeros((len(x),len(wd_array),len(ws_array))))


# yaw control ------------------

p_mat_yawControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
yaw_mat_yawControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
helix_amp_mat_yawControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))

t = time.time()

for w_ind in np.arange(len(weight_coefficient_array)):

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_yaw_s25_ws3to5_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_1 = pickle.load(f)
    ws_ind_temp_1 = np.array([0,1,2])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_yaw_s25_ws6to8_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_2 = pickle.load(f)
    ws_ind_temp_2 = np.array([3,4,5])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_yaw_s25_ws9to11_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_3 = pickle.load(f)
    ws_ind_temp_3 = np.array([6,7,8])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_yaw_s25_ws12to14_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_4 = pickle.load(f)
    ws_ind_temp_4 = np.array([9,10,11])

    data_list = [data_temp_1,data_temp_2,data_temp_3,data_temp_4]
    ws_ind_temp_list = [ws_ind_temp_1,ws_ind_temp_2,ws_ind_temp_3,ws_ind_temp_4]

    yaw_mat,helix_amp_mat = extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array)
    p_mat = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mat,helix_amp_mat,sigma=sigma,n=n_values)

    p_mat_yawControl_ampMax25_Pareto[w_ind,:,:,:] = p_mat
    yaw_mat_yawControl_ampMax25_Pareto[w_ind,:,:,:] = yaw_mat
    helix_amp_mat_yawControl_ampMax25_Pareto[w_ind,:,:,:] = helix_amp_mat

print(f'Time for extracting the data - Yaw control: \t {time.time()-t}')


# helix control ------------------

p_mat_helixControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
yaw_mat_helixControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
helix_amp_mat_helixControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))

t = time.time()

for w_ind in np.arange(len(weight_coefficient_array)):

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_helix_s25_ws3to5_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_1 = pickle.load(f)
    ws_ind_temp_1 = np.array([0,1,2])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_helix_s25_ws6to8_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_2 = pickle.load(f)
    ws_ind_temp_2 = np.array([3,4,5])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_helix_s25_ws9to11_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_3 = pickle.load(f)
    ws_ind_temp_3 = np.array([6,7,8])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_helix_s25_ws12to14_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_4 = pickle.load(f)
    ws_ind_temp_4 = np.array([9,10,11])

    data_list = [data_temp_1,data_temp_2,data_temp_3,data_temp_4]
    ws_ind_temp_list = [ws_ind_temp_1,ws_ind_temp_2,ws_ind_temp_3,ws_ind_temp_4]

    yaw_mat,helix_amp_mat = extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array)
    p_mat = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mat,helix_amp_mat,sigma=sigma,n=n_values)

    p_mat_helixControl_ampMax25_Pareto[w_ind,:,:,:] = p_mat
    yaw_mat_helixControl_ampMax25_Pareto[w_ind,:,:,:] = yaw_mat
    helix_amp_mat_helixControl_ampMax25_Pareto[w_ind,:,:,:] = helix_amp_mat

print(f'Time for extracting the data - Helix control: \t {time.time()-t}')


# mixed control ------------------

p_mat_mixedControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
yaw_mat_mixedControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
helix_amp_mat_mixedControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))

t = time.time()

for w_ind in np.arange(len(weight_coefficient_array)):

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_mixed_s25_ws3to5_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_1 = pickle.load(f)
    ws_ind_temp_1 = np.array([0,1,2])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_mixed_s25_ws6to8_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_2 = pickle.load(f)
    ws_ind_temp_2 = np.array([3,4,5])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_mixed_s25_ws9to11_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_3 = pickle.load(f)
    ws_ind_temp_3 = np.array([6,7,8])

    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_25/data/wffcLUT_HKN_mixed_s25_ws12to14_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_4 = pickle.load(f)
    ws_ind_temp_4 = np.array([9,10,11])

    data_list = [data_temp_1,data_temp_2,data_temp_3,data_temp_4]
    ws_ind_temp_list = [ws_ind_temp_1,ws_ind_temp_2,ws_ind_temp_3,ws_ind_temp_4]

    yaw_mat,helix_amp_mat = extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array)
    p_mat = calculatePmat_withUncertainty(wfm,x,y,wd_array,ws_array,yaw_mat,helix_amp_mat,sigma=sigma,n=n_values)

    p_mat_mixedControl_ampMax25_Pareto[w_ind,:,:,:] = p_mat
    yaw_mat_mixedControl_ampMax25_Pareto[w_ind,:,:,:] = yaw_mat
    helix_amp_mat_mixedControl_ampMax25_Pareto[w_ind,:,:,:] = helix_amp_mat

print(f'Time for extracting the data - Mixed control: \t {time.time()-t}')



# %%
# CALCULATE AEP GAINS - helix_amp_max = 2.5 deg

# extract porbability of flow cases (per turbine)
prob_mat = simres_baseline_0std.P.values

# calculate AEP baseline
aep_baseline = 8760*np.sum(p_mat_baseline*prob_mat)/1e9

# calculate AEP gains

aep_yawControl_ampMax25_Pareto = 8760*np.sum(p_mat_yawControl_ampMax25_Pareto*prob_mat[na,:,:,:],axis=(1,2,3))/1e9
aep_helixControl_ampMax25_Pareto = 8760*np.sum(p_mat_helixControl_ampMax25_Pareto*prob_mat[na,:,:,:],axis=(1,2,3))/1e9
aep_mixedControl_ampMax25_Pareto = 8760*np.sum(p_mat_mixedControl_ampMax25_Pareto*prob_mat[na,:,:,:],axis=(1,2,3))/1e9

aep_gain_yawControl_ampMax25_Pareto = 100*(aep_yawControl_ampMax25_Pareto-aep_baseline)/aep_baseline
aep_gain_helixControl_ampMax25_Pareto = 100*(aep_helixControl_ampMax25_Pareto-aep_baseline)/aep_baseline
aep_gain_mixedControl_ampMax25_Pareto = 100*(aep_mixedControl_ampMax25_Pareto-aep_baseline)/aep_baseline


# calculate percentage of operation

wffc_binary_mat_yawControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
wffc_binary_mat_helixControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
wffc_binary_mat_mixedControl_ampMax25_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))

wffc_binary_mat_yawControl_ampMax25_Pareto[(np.abs(yaw_mat_yawControl_ampMax25_Pareto)>0)|(helix_amp_mat_yawControl_ampMax25_Pareto>0)]=1
wffc_binary_mat_helixControl_ampMax25_Pareto[(np.abs(yaw_mat_helixControl_ampMax25_Pareto)>0)|(helix_amp_mat_helixControl_ampMax25_Pareto>0)]=1
wffc_binary_mat_mixedControl_ampMax25_Pareto[(np.abs(yaw_mat_mixedControl_ampMax25_Pareto)>0)|(helix_amp_mat_mixedControl_ampMax25_Pareto>0)]=1

perc_operation_i_yawControl_ampMax25_Pareto = 100*np.sum(wffc_binary_mat_yawControl_ampMax25_Pareto*prob_mat[na,:,:,:],axis=(2,3))
perc_operation_i_helixControl_ampMax25_Pareto = 100*np.sum(wffc_binary_mat_helixControl_ampMax25_Pareto*prob_mat[na,:,:,:],axis=(2,3))
perc_operation_i_mixedControl_ampMax25_Pareto = 100*np.sum(wffc_binary_mat_mixedControl_ampMax25_Pareto*prob_mat[na,:,:,:],axis=(2,3))

perc_operation_mean_yawControl_ampMax25_Pareto = np.mean(perc_operation_i_yawControl_ampMax25_Pareto,axis=(1))
perc_operation_mean_helixControl_ampMax25_Pareto = np.mean(perc_operation_i_helixControl_ampMax25_Pareto,axis=(1))
perc_operation_mean_mixedControl_ampMax25_Pareto = np.mean(perc_operation_i_mixedControl_ampMax25_Pareto,axis=(1))



#%%
# SAVE DATA TO PLOT PARETO - helix_amp_max = 2.5 deg

with open(f'data_HKN_Pareto_v3_25deg.pkl', 'wb') as f:
    pickle.dump({'aep_baseline' : aep_baseline,
                 'prob_mat' : prob_mat,
                 'aep_yawControl_ampMax25_Pareto' : aep_yawControl_ampMax25_Pareto,
                 'aep_helixControl_ampMax25_Pareto' : aep_helixControl_ampMax25_Pareto,
                 'aep_mixedControl_ampMax25_Pareto' : aep_mixedControl_ampMax25_Pareto,
                 'wffc_binary_mat_yawControl_ampMax25_Pareto' : wffc_binary_mat_yawControl_ampMax25_Pareto,
                 'wffc_binary_mat_helixControl_ampMax25_Pareto' : wffc_binary_mat_helixControl_ampMax25_Pareto,
                 'wffc_binary_mat_mixedControl_ampMax25_Pareto' : wffc_binary_mat_mixedControl_ampMax25_Pareto,
                 'perc_operation_i_yawControl_ampMax25_Pareto' : perc_operation_i_yawControl_ampMax25_Pareto,
                 'perc_operation_i_helixControl_ampMax25_Pareto' : perc_operation_i_helixControl_ampMax25_Pareto,
                 'perc_operation_i_mixedControl_ampMax25_Pareto' : perc_operation_i_mixedControl_ampMax25_Pareto,
                }, f)




#%%
# EXTRACT DATA TO PLOT PARETO

with open(f'data_HKN_Pareto_v3_5deg.pkl', 'rb') as f:
    data_Pareto = pickle.load(f)

aep_baseline = data_Pareto['aep_baseline']
prob_mat = data_Pareto['prob_mat']
aep_yawControl_ampMax5_Pareto = data_Pareto['aep_yawControl_ampMax5_Pareto']
aep_helixControl_ampMax5_Pareto = data_Pareto['aep_helixControl_ampMax5_Pareto']
aep_mixedControl_ampMax5_Pareto = data_Pareto['aep_mixedControl_ampMax5_Pareto']
wffc_binary_mat_yawControl_ampMax5_Pareto = data_Pareto['wffc_binary_mat_yawControl_ampMax5_Pareto']
wffc_binary_mat_helixControl_ampMax5_Pareto = data_Pareto['wffc_binary_mat_helixControl_ampMax5_Pareto']
wffc_binary_mat_mixedControl_ampMax5_Pareto = data_Pareto['wffc_binary_mat_mixedControl_ampMax5_Pareto']
perc_operation_i_yawControl_ampMax5_Pareto = data_Pareto['perc_operation_i_yawControl_ampMax5_Pareto']
perc_operation_i_helixControl_ampMax5_Pareto = data_Pareto['perc_operation_i_helixControl_ampMax5_Pareto']
perc_operation_i_mixedControl_ampMax5_Pareto = data_Pareto['perc_operation_i_mixedControl_ampMax5_Pareto']

aep_gain_yawControl_ampMax5_Pareto = 100*(aep_yawControl_ampMax5_Pareto-aep_baseline)/aep_baseline
aep_gain_helixControl_ampMax5_Pareto = 100*(aep_helixControl_ampMax5_Pareto-aep_baseline)/aep_baseline
aep_gain_mixedControl_ampMax5_Pareto = 100*(aep_mixedControl_ampMax5_Pareto-aep_baseline)/aep_baseline

perc_operation_mean_yawControl_ampMax5_Pareto = np.mean(perc_operation_i_yawControl_ampMax5_Pareto,axis=(1))
perc_operation_mean_helixControl_ampMax5_Pareto = np.mean(perc_operation_i_helixControl_ampMax5_Pareto,axis=(1))
perc_operation_mean_mixedControl_ampMax5_Pareto = np.mean(perc_operation_i_mixedControl_ampMax5_Pareto,axis=(1))



with open(f'data_HKN_Pareto_v3_25deg.pkl', 'rb') as f:
    data_Pareto = pickle.load(f)

aep_baseline = data_Pareto['aep_baseline']
prob_mat = data_Pareto['prob_mat']
aep_yawControl_ampMax25_Pareto = data_Pareto['aep_yawControl_ampMax25_Pareto']
aep_helixControl_ampMax25_Pareto = data_Pareto['aep_helixControl_ampMax25_Pareto']
aep_mixedControl_ampMax25_Pareto = data_Pareto['aep_mixedControl_ampMax25_Pareto']
wffc_binary_mat_yawControl_ampMax25_Pareto = data_Pareto['wffc_binary_mat_yawControl_ampMax25_Pareto']
wffc_binary_mat_helixControl_ampMax25_Pareto = data_Pareto['wffc_binary_mat_helixControl_ampMax25_Pareto']
wffc_binary_mat_mixedControl_ampMax25_Pareto = data_Pareto['wffc_binary_mat_mixedControl_ampMax25_Pareto']
perc_operation_i_yawControl_ampMax25_Pareto = data_Pareto['perc_operation_i_yawControl_ampMax25_Pareto']
perc_operation_i_helixControl_ampMax25_Pareto = data_Pareto['perc_operation_i_helixControl_ampMax25_Pareto']
perc_operation_i_mixedControl_ampMax25_Pareto = data_Pareto['perc_operation_i_mixedControl_ampMax25_Pareto']

aep_gain_yawControl_ampMax25_Pareto = 100*(aep_yawControl_ampMax25_Pareto-aep_baseline)/aep_baseline
aep_gain_helixControl_ampMax25_Pareto = 100*(aep_helixControl_ampMax25_Pareto-aep_baseline)/aep_baseline
aep_gain_mixedControl_ampMax25_Pareto = 100*(aep_mixedControl_ampMax25_Pareto-aep_baseline)/aep_baseline

perc_operation_mean_yawControl_ampMax25_Pareto = np.mean(perc_operation_i_yawControl_ampMax25_Pareto,axis=(1))
perc_operation_mean_helixControl_ampMax25_Pareto = np.mean(perc_operation_i_helixControl_ampMax25_Pareto,axis=(1))
perc_operation_mean_mixedControl_ampMax25_Pareto = np.mean(perc_operation_i_mixedControl_ampMax25_Pareto,axis=(1))




# %%
# plot PARETO

savefig = False
#name_path = r'figures/LUT_HKN/'
name_path = r'figures/WES_review/'
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']


# all strategies together
plt.figure(figsize=(6,5))
plt.plot(aep_gain_helixControl_ampMax25_Pareto,-perc_operation_mean_helixControl_ampMax25_Pareto,c=colors[4],marker='s',markersize=7,linestyle='--',linewidth=2.5,label=r'Helix ($\beta_{\mathrm{max}}=2.5^\circ$)')
plt.plot(aep_gain_mixedControl_ampMax25_Pareto,-perc_operation_mean_mixedControl_ampMax25_Pareto,c=colors[0],marker='s',markersize=7,linestyle='--',linewidth=2.5,label=r'Combined ($\beta_{\mathrm{max}}=2.5^\circ$)')
plt.plot(aep_gain_yawControl_ampMax5_Pareto,-perc_operation_mean_yawControl_ampMax5_Pareto,c=colors[2],marker='o',markersize=7,linestyle='-',linewidth=2.5,label=r'Wake steering')
plt.plot(aep_gain_helixControl_ampMax5_Pareto,-perc_operation_mean_helixControl_ampMax5_Pareto,c=colors[4],marker='o',markersize=7,linestyle='-',linewidth=2.5,label=r'Helix ($\beta_{\mathrm{max}}=5^\circ$)')
plt.plot(aep_gain_mixedControl_ampMax5_Pareto,-perc_operation_mean_mixedControl_ampMax5_Pareto,c=colors[0],marker='o',markersize=7,linestyle='-',linewidth=2.5,label=r'Combined ($\beta_{\mathrm{max}}=5^\circ$)')
plt.xlabel(r'AEP gain [%]')
plt.ylabel(r'- $\mathrm{COT}$ [%]')
plt.xlim([0.,1.])
plt.ylim([-50,2])
plt.legend(loc='lower left')#, bbox_to_anchor=(0.5, 1.35), ncol=1)
if savefig: plt.savefig(name_path+'HKN_LUT_Pareto_all_v5.pdf',format='pdf')
plt.show()


#%%
# plot PERCENTAGE OPERATION LAYOUTS (mixed)

savefig = False
name_path = r'figures/LUT_HKN/'

# choose 3 weight values to plot
weight_coefficient_values_array = np.array([0,0.25,0.5,0.75,1,2.5,5,7.5,10])
w_ind_1 = 2
w_ind_2 = 4
w_ind_3 = 5


fig,axs = plt.subplots(nrows=1,ncols=4,figsize=(10,4),gridspec_kw={'width_ratios':[1,1,1,0.05],'wspace':0.2})

axs[0].set_title(f'AEP gain: {aep_gain_mixedControl_ampMax5_Pareto[w_ind_1]:.2f} %'+"\n"+r"$\overline{p}_{\mathrm{LT}}$"+fr"$={perc_operation_mean_mixedControl_ampMax5_Pareto[w_ind_1]:.1f}$ %")
c_values = perc_operation_i_mixedControl_ampMax5_Pareto[w_ind_1,:]
sc = axs[0].scatter(x,y,c=c_values,cmap='Purples',vmin=0,vmax=40,edgecolors='black',linewidths=0.25)
axs[0].set_aspect('equal')
axs[0].axis('off')

axs[1].set_title(f'AEP gain: {aep_gain_mixedControl_ampMax5_Pareto[w_ind_2]:.2f} %'+"\n"+r"$\overline{p}_{\mathrm{LT}}$"+fr"$={perc_operation_mean_mixedControl_ampMax5_Pareto[w_ind_2]:.1f}$ %")
c_values = perc_operation_i_mixedControl_ampMax5_Pareto[w_ind_2,:]
sc = axs[1].scatter(x,y,c=c_values,cmap='Purples',vmin=0,vmax=40,edgecolors='black',linewidths=0.25)
axs[1].set_aspect('equal')
axs[1].axis('off')

axs[2].set_title(f'AEP gain: {aep_gain_mixedControl_ampMax5_Pareto[w_ind_3]:.2f} %'+"\n"+r"$\overline{p}_{\mathrm{LT}}$"+fr"$={perc_operation_mean_mixedControl_ampMax5_Pareto[w_ind_3]:.1f}$ %")
c_values = perc_operation_i_mixedControl_ampMax5_Pareto[w_ind_3,:]
sc = axs[2].scatter(x,y,c=c_values,cmap='Purples',vmin=0,vmax=40,edgecolors='black',linewidths=0.25)
axs[2].set_aspect('equal')
axs[2].axis('off')

fig.colorbar(sc, cax=axs[3], label=r'$p_{\mathrm{LT},i}$ [%]')

if savefig: plt.savefig(name_path+'perc_operation_layout_mixedOpt_Pareto.svg',format='svg')

plt.show()





#%%
# PLOT CONTROL ROSE --------------------------------------------------------------------

#%%
# extract data

def extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array):
    yaw_mat = np.zeros((len(x),len(wd_array),len(ws_array)))
    helix_amp_mat = np.zeros((len(x),len(wd_array),len(ws_array)))
    for i in np.arange(len(data_list)):
        yaw_mat_temp = data_list[i]['yaw_opt']
        helix_amp_mat_temp = data_list[i]['helix_amp_opt']
        yaw_mat[:,:,ws_ind_temp_list[i]] = yaw_mat_temp
        helix_amp_mat[:,:,ws_ind_temp_list[i]] = helix_amp_mat_temp
    return yaw_mat,helix_amp_mat

wd_array = np.arange(0,360,1)
ws_array = np.arange(3,26,1)

weight_coefficient_array = np.array(['0250'])

# extract yaw data 
yaw_mat_yawControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
helix_amp_mat_yawControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
t = time.time()
for w_ind in np.arange(len(weight_coefficient_array)):
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_yaw_s25_ws3to5_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_1 = pickle.load(f)
    ws_ind_temp_1 = np.array([0,1,2])
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_yaw_s25_ws6to8_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_2 = pickle.load(f)
    ws_ind_temp_2 = np.array([3,4,5])
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_yaw_s25_ws9to11_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_3 = pickle.load(f)
    ws_ind_temp_3 = np.array([6,7,8])
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_yaw_s25_ws12to14_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_4 = pickle.load(f)
    ws_ind_temp_4 = np.array([9,10,11])
    data_list = [data_temp_1,data_temp_2,data_temp_3,data_temp_4]
    ws_ind_temp_list = [ws_ind_temp_1,ws_ind_temp_2,ws_ind_temp_3,ws_ind_temp_4]
    yaw_mat,helix_amp_mat = extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array)
    yaw_mat_yawControl_ampMax5_Pareto[w_ind,:,:,:] = yaw_mat
    helix_amp_mat_yawControl_ampMax5_Pareto[w_ind,:,:,:] = helix_amp_mat
print(f'Time for extracting the data - Yaw control: \t {time.time()-t}')

# extract helix data
yaw_mat_helixControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
helix_amp_mat_helixControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
t = time.time()
for w_ind in np.arange(len(weight_coefficient_array)):
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_helix_s25_ws3to5_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_1 = pickle.load(f)
    ws_ind_temp_1 = np.array([0,1,2])
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_helix_s25_ws6to8_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_2 = pickle.load(f)
    ws_ind_temp_2 = np.array([3,4,5])
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_helix_s25_ws9to11_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_3 = pickle.load(f)
    ws_ind_temp_3 = np.array([6,7,8])
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_helix_s25_ws12to14_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_4 = pickle.load(f)
    ws_ind_temp_4 = np.array([9,10,11])
    data_list = [data_temp_1,data_temp_2,data_temp_3,data_temp_4]
    ws_ind_temp_list = [ws_ind_temp_1,ws_ind_temp_2,ws_ind_temp_3,ws_ind_temp_4]
    yaw_mat,helix_amp_mat = extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array)
    yaw_mat_helixControl_ampMax5_Pareto[w_ind,:,:,:] = yaw_mat
    helix_amp_mat_helixControl_ampMax5_Pareto[w_ind,:,:,:] = helix_amp_mat
print(f'Time for extracting the data - Helix control: \t {time.time()-t}')

# extract mixed data
yaw_mat_mixedControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
helix_amp_mat_mixedControl_ampMax5_Pareto = np.zeros((len(weight_coefficient_array),len(x),len(wd_array),len(ws_array)))
t = time.time()
for w_ind in np.arange(len(weight_coefficient_array)):
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_mixed_s25_ws3to5_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_1 = pickle.load(f)
    ws_ind_temp_1 = np.array([0,1,2])
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_mixed_s25_ws6to8_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_2 = pickle.load(f)
    ws_ind_temp_2 = np.array([3,4,5])
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_mixed_s25_ws9to11_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_3 = pickle.load(f)
    ws_ind_temp_3 = np.array([6,7,8])
    with open(f'LUT_HKN_DelftBlue_simulations/LUT_HKN_Pareto_v2/helix_amp_max_5/data/wffcLUT_HKN_mixed_s25_ws12to14_w'+weight_coefficient_array[w_ind]+'e5.pkl', 'rb') as f:
        data_temp_4 = pickle.load(f)
    ws_ind_temp_4 = np.array([9,10,11])
    data_list = [data_temp_1,data_temp_2,data_temp_3,data_temp_4]
    ws_ind_temp_list = [ws_ind_temp_1,ws_ind_temp_2,ws_ind_temp_3,ws_ind_temp_4]
    yaw_mat,helix_amp_mat = extract_WFFC(data_list,ws_ind_temp_list,wd_array,ws_array)
    yaw_mat_mixedControl_ampMax5_Pareto[w_ind,:,:,:] = yaw_mat
    helix_amp_mat_mixedControl_ampMax5_Pareto[w_ind,:,:,:] = helix_amp_mat
print(f'Time for extracting the data - Mixed control: \t {time.time()-t}')


#%%
# plot

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


plot_optControlRose(ind_turbine = 0,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_mat_mixedControl_ampMax5_Pareto[0,:,:,:],
                    helix_amp_ilk = helix_amp_mat_mixedControl_ampMax5_Pareto[0,:,:,:],
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'mixedControlRose_wt0_25std_w0250_v2.pdf',
                    format_fig = 'pdf')

plot_optControlRose(ind_turbine = 30,
                    x = x,
                    y = y,
                    yaw_ilk = yaw_mat_mixedControl_ampMax5_Pareto[0,:,:,:],
                    helix_amp_ilk = helix_amp_mat_mixedControl_ampMax5_Pareto[0,:,:,:],
                    x_boundaries = x_boundaries,
                    y_boundaries = y_boundaries,
                    savefig = savefig,
                    name_path = name_path,
                    name_fig = 'mixedControlRose_wt30_25std_w0250_v2.pdf',
                    format_fig = 'pdf')







# %%
