#%%
# IMPORT PACKAGES

import numpy as np
from functools import partial
import time
import matplotlib.pyplot as plt
from numpy import newaxis as na
import pickle
import utm
import xarray as xr

# import py_wake_helix models
from py_wake_helix.py_wake_helix import helix_power_ct_function
from py_wake_helix.py_wake_helix import PropagateDownwind_helix
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeficit
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeflection

# import py_wake_helix_tools models
from py_wake_helix.py_wake_helix_tools import Power_wrapper

# import optimizer
from msr import MSR_optimizer
from msr import ObjFuncComponent

# import py_pywake models
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.site import XRSite
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.superposition_models import SquaredSum

#%%
# DEFINE PYWAKE MODELS

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


# flow condition
wd = 201.
ws = 8.

# calculate effective wind speed (need for Geometric yaw)
ws_eff = wfm(x,y,wd=np.array([wd]),ws=np.array([ws]),yaw=np.zeros((len(x),1,1)),helix_amp=np.zeros((len(x),1,1)),tilt=0.).WS_eff_ilk.reshape(-2)


# define objective functions

def calculate_power_wake_steering(x,y,wfm,wd,ws,yaw):
    f_temp =  Power_wrapper(x = x,
                            y = y,
                            wfm = wfm,
                            wd = wd,
                            ws = ws)
    helix_amp =  np.zeros_like(yaw)
    return f_temp(yaw[:,na,na],helix_amp[:,na,na])

def calculate_power_combined(x,y,wfm,wd,ws,yaw,helix_amp):
    f_temp =  Power_wrapper(x = x,
                            y = y,
                            wfm = wfm,
                            wd = wd,
                            ws = ws)
    return f_temp(yaw[:,na,na],helix_amp[:,na,na])


#%%
# test MSR: WAKE STEERING - Refine

# create objective function object
f_obj = ObjFuncComponent(obj_func = calculate_power_wake_steering,
                         input_keys = ['yaw'],
                         x = x,
                         y = y,
                         wfm = wfm,
                         wd = wd,
                         ws = ws)

# create optimizer object
optimizer_MSR = MSR_optimizer(x = x,
                              y = y,
                              wd = wd,
                              f_obj = f_obj,
                              n_step = 3,
                              exclusivity = True
                              )

# add strategy (Wake steering - Refine)
optimizer_MSR.add_strategy(str_name = 'Wake steering',
                           var_name = 'yaw',
                           opt_method = 'Refine',
                           n_values = 5,
                           cmin = -30.,
                           cmax = 30.)

t = time.time()
optimizer_MSR.optimize()
c_opt = optimizer_MSR.c_opt
yaw_opt_Refine = c_opt['yaw']
f_opt_Refine = optimizer_MSR.f_opt
t_Refine = time.time()-t

print(f'Optimization (Wake steering - Refine) - Time:{time.time()-t}')
print(f'Yaw: {yaw_opt_Refine}')
print(f'F value: {optimizer_MSR.f_opt}')


#%%
# test MSR: WAKE STEERING - Discrete

# create objective function object
f_obj = ObjFuncComponent(obj_func = calculate_power_wake_steering,
                         input_keys = ['yaw'],
                         x = x,
                         y = y,
                         wfm = wfm,
                         wd = wd,
                         ws = ws)

# create optimizer object
optimizer_MSR = MSR_optimizer(x = x,
                              y = y,
                              wd = wd,
                              f_obj = f_obj,
                              n_step = 3,
                              exclusivity = True
                              )

# add strategy (Wake steering - Discrete)
optimizer_MSR.add_strategy(str_name = 'Wake steering',
                           var_name = 'yaw',
                           opt_method = 'Discrete',
                           c_values_array = np.array([-20.,0.,20.]))

t = time.time()
optimizer_MSR.optimize()
c_opt = optimizer_MSR.c_opt
yaw_opt_Discrete = c_opt['yaw']
f_opt_Discrete = optimizer_MSR.f_opt
t_Discrete = time.time()-t

print(f'Optimization (Wake steering - Discrete) - Time:{time.time()-t}')
print(f'Yaw: {yaw_opt_Discrete}')
print(f'F value: {optimizer_MSR.f_opt}')


#%%
# test MSR: WAKE STEERING - Geometric yaw

# create objective function object
f_obj = ObjFuncComponent(obj_func = calculate_power_wake_steering,
                         input_keys = ['yaw'],
                         x = x,
                         y = y,
                         wfm = wfm,
                         wd = wd,
                         ws = ws)

# create optimizer object
optimizer_MSR = MSR_optimizer(x = x,
                              y = y,
                              wd = wd,
                              f_obj = f_obj,
                              n_step = 3,
                              exclusivity = True
                              )

# add strategy (Wake steering - Geometric yaw)
optimizer_MSR.add_strategy(str_name = 'Wake steering',
                           var_name = 'yaw',
                           opt_method = 'Geometric yaw',
                           geom_yaw_method = 'Exponential corrected',
                           ws_rated = 11.,
                           diameter = diameter,
                           ws_eff = ws_eff)

t = time.time()
optimizer_MSR.optimize()
c_opt = optimizer_MSR.c_opt
yaw_opt_Geom = c_opt['yaw']
f_opt_Geom = optimizer_MSR.f_opt
t_Geom = time.time()-t

print(f'Optimization (Wake steering - Geom) - Time:{time.time()-t}')
print(f'Yaw: {yaw_opt_Geom}')
print(f'F value: {optimizer_MSR.f_opt}')


#%%
# test MSR: WAKE STEERING + HELIX - Refine

# create objective function object
f_obj = ObjFuncComponent(obj_func = calculate_power_combined,
                         input_keys = ['yaw','helix_amp'],
                         x = x,
                         y = y,
                         wfm = wfm,
                         wd = wd,
                         ws = ws)

# create optimizer object
optimizer_MSR = MSR_optimizer(x = x,
                              y = y,
                              wd = wd,
                              f_obj = f_obj,
                              n_step = 3,
                              exclusivity = True
                              )

# add strategy (Wake steering - Refine)
optimizer_MSR.add_strategy(str_name = 'Wake steering',
                           var_name = 'yaw',
                           opt_method = 'Refine',
                           n_values = 5,
                           cmin = -30.,
                           cmax = 30.)

# add strategy (Helix - Refine)
optimizer_MSR.add_strategy(str_name = 'Helix',
                           var_name = 'helix_amp',
                           opt_method = 'Refine',
                           n_values = 5,
                           cmin = 0.,
                           cmax = 5.)

t = time.time()
optimizer_MSR.optimize()
c_opt = optimizer_MSR.c_opt
yaw_opt_combined = c_opt['yaw']
helix_amp_opt_combined = c_opt['helix_amp']
f_opt_combined = optimizer_MSR.f_opt
t_combined = time.time()-t

print(f'Optimization (Combined - Refine) - Time:{time.time()-t}')
print(f'Yaw: {yaw_opt_combined}')
print(f'Helix: {helix_amp_opt_combined}')
print(f'F value: {optimizer_MSR.f_opt}')


# %%
# plot comparison

savefig = False
name_path = r'figures/'
colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']


# calculate improvement
f_baseline = calculate_power_combined(x,y,wfm,wd,ws,yaw=np.zeros(len(x)),helix_amp=np.zeros(len(x)))
f_gain_Refine = 100*(f_opt_Refine-f_baseline)/f_baseline
f_gain_Discrete = 100*(f_opt_Discrete-f_baseline)/f_baseline
f_gain_Geom = 100*(f_opt_Geom-f_baseline)/f_baseline
f_gain_combined = 100*(f_opt_combined-f_baseline)/f_baseline

labels = ['Yaw-Refine','Yaw-Discrete','Yaw-Geom','Yaw/Helix-Refine']

# plot imp power gains rovement
f_gains = [f_gain_Refine,f_gain_Discrete,f_gain_Geom]#,f_gain_combined]
plt.title(f'MSR performance for HKN - WD: {wd} deg , WS: {ws} m/s')
plt.bar(np.arange(len(f_gains)),f_gains)
plt.xticks(ticks=range(len(f_gains)),labels=labels[:-1])
plt.xlabel('MSR strategy and optimization method')
plt.ylabel('Power gain [%]')
if savefig: plt.savefig(name_path+'MSR_yaw_performance_power.svg',format='svg')
plt.show()

# plot time
t_list = [t_Refine,t_Discrete,t_Geom]#,t_combined]
plt.title(f'MSR performance for HKN - WD: {wd} deg , WS: {ws} m/s')
plt.bar(np.arange(len(t_list)),t_list)
plt.xticks(ticks=range(len(t_list)),labels=labels[:-1])
plt.xlabel('MSR strategy and optimization method')
plt.ylabel('Time [s]')
if savefig: plt.savefig(name_path+'MSR_yaw_performance_time.svg',format='svg')
plt.show()


# plot control variables - Yaw-Refine (n_values=5,n_step=3)
plt.figure(figsize=(4,4))
plt.title('Yaw-Refine (n_values=5,n_step=3)')
sc = plt.scatter(x,y,c=yaw_opt_Refine,cmap='bwr',vmin=-25.,vmax=25.,edgecolors='black')
cbar = plt.colorbar(sc)
cbar.set_label('Yaw angle [deg]')
plt.axis('off')
plt.axis('equal')
if savefig: plt.savefig(name_path+'MSR_yaw_values_refine.svg',format='svg')
plt.show()

# plot control variables - Yaw-Discrete (values=[-20,0,20])
plt.figure(figsize=(4,4))
plt.title('Yaw-Discrete (values=[-20,0,20])')
sc = plt.scatter(x,y,c=yaw_opt_Discrete,cmap='bwr',vmin=-25.,vmax=25.,edgecolors='black')
cbar = plt.colorbar(sc)
cbar.set_label('Yaw angle [deg]')
plt.axis('off')
plt.axis('equal')
if savefig: plt.savefig(name_path+'MSR_yaw_values_discrete.svg',format='svg')
plt.show()

# plot control variables - Yaw-Geom (ExpCorr not tuned)
plt.figure(figsize=(4,4))
plt.title('Yaw-Geom (ExpCorr not tuned)')
sc = plt.scatter(x,y,c=yaw_opt_Geom,cmap='bwr',vmin=-25.,vmax=25.,edgecolors='black')
cbar = plt.colorbar(sc)
cbar.set_label('Yaw angle [deg]')
plt.axis('off')
plt.axis('equal')
if savefig: plt.savefig(name_path+'MSR_yaw_values_geom.svg',format='svg')
plt.show()

# plot control variables - Yaw-Helix combined
fig, ax = plt.subplots(figsize=(6,4))
ax.set_title('Yaw/Helix-Refine (n_values=5,n_step=3)')
fil_helix = helix_amp_opt_combined>0.
fil_yaw = np.abs(yaw_opt_combined)>0.
sc1 = ax.scatter(x[fil_yaw],y[fil_yaw],c=yaw_opt_combined[fil_yaw],cmap='bwr',vmin=-25.,vmax=25)
sc2 = ax.scatter(x[fil_helix],y[fil_helix],c=helix_amp_opt_combined[fil_helix],cmap='Greens',vmin=0.,vmax=5.)
sc = ax.scatter(x,y,facecolors='none',edgecolors='black')
cbar1 = fig.colorbar(sc1,ax=ax, orientation='vertical',fraction=0.046,pad=0.1)
cbar1.set_label('Yaw angle [deg]')
cbar2 = fig.colorbar(sc2,ax=ax,orientation='vertical',fraction=0.046,pad=0.02)
cbar2.set_label('Helix amplitude [deg]')
ax.axis('off')
ax.set_aspect('equal')
plt.tight_layout()
if savefig: plt.savefig(name_path+'MSR_combined_values_refine.svg',format='svg')
plt.show()





# %%
