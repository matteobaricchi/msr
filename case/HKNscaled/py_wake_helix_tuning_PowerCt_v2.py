# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:26:28 2025

@author: matteobaricchi
"""
#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# import new models
from py_wake_helix.py_wake_helix import helix_power_ct_function
from py_wake_helix.py_wake_helix import PropagateDownwind_helix
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeficit
from py_wake_helix.py_wake_helix import EmpiricalGaussianDeflection


#%% METHOD
# - The following coefficients are tuned based on the power loss: helix_a, helix_power_b
# - The following coefficients are tuned based on the thrust loss: helix_thrust_b

# in this version 2, the coefficients helix_power_c and helix_thrust_c are not tuned and are assumed equal to the IEA15MW turbine in FLORIS



#%% IMPORT DATA - POWER CT

#os.chdir(r'C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\data_tuning\DATA_HelixEngineeringModel_tuning_20250220')
path_name = r'data_tuning\DATA_HelixEngineeringModel_tuning_20250220\\'

df_helix_power_loss_GP = pd.read_csv(path_name+'helix_power_loss.csv')           # Daan - data obtained with GP - U=8m/s TI=0.04
df_helix_thrust_loss_GP = pd.read_csv(path_name+'helix_thrust_loss.csv')         # Daan - data obtained with GP - U=8m/s TI=0.04
df_helix_loss_LES = pd.read_csv(path_name+'power_curves_TI2p.csv')               # Tim - TI=0.02 - St=0.2


#%% FLORIS functions

# FLORIS values
helix_a0 = 1.809
helix_power_b0 = 4.828e-03
helix_power_c0 = 4.017e-11
helix_thrust_b0 = 1.390e-03
helix_thrust_c0 = 5.084e-04

# create a wrapper function (8 m/s)
def power_loss_function_FLORIS(helix_amp):
    u = 8.
    u_array = np.ones(len(helix_amp))*u
    p = helix_power_ct_function(u=u_array, run_only=0, helix_amp=helix_amp, helix_a = helix_a0, helix_power_b = helix_power_b0, helix_power_c = helix_power_c0, helix_thrust_b = helix_thrust_b0, helix_thrust_c = helix_thrust_c0)
    p0 = helix_power_ct_function(u=np.array([u]), run_only=0, helix_amp=np.array([0.]), helix_a = helix_a0, helix_power_b = helix_power_b0, helix_power_c = helix_power_c0, helix_thrust_b = helix_thrust_b0, helix_thrust_c = helix_thrust_c0)
    return p/p0

# create a wrapper function (8 m/s)
def ct_loss_function_FLORIS(helix_amp):
    u = 8.
    u_array = np.ones(len(helix_amp))*u
    ct = helix_power_ct_function(u=u_array, run_only=1, helix_amp=helix_amp, helix_a = helix_a0, helix_power_b = helix_power_b0, helix_power_c = helix_power_c0, helix_thrust_b = helix_thrust_b0, helix_thrust_c = helix_thrust_c0)
    ct0 = helix_power_ct_function(u=np.array([u]), run_only=1, helix_amp=np.array([0.]), helix_a = helix_a0, helix_power_b = helix_power_b0, helix_power_c = helix_power_c0, helix_thrust_b = helix_thrust_b0, helix_thrust_c = helix_thrust_c0)
    return ct/ct0

# FLORIS initial data
helix_amp_array_floris = np.linspace(0,5.5,100,endpoint=True)
p_loss_floris = power_loss_function_FLORIS(helix_amp_array_floris)
ct_loss_floris = ct_loss_function_FLORIS(helix_amp_array_floris)


#%% TUNING POWER CURVE

# create a wrapper function (8 m/s)
def power_loss_function(helix_amp,helix_a,helix_power_b):
    u = 8.
    u_array = np.ones(len(helix_amp))*u
    p = helix_power_ct_function(u=u_array, run_only=0, helix_amp=helix_amp, helix_a = helix_a, helix_power_b = helix_power_b, helix_power_c = helix_power_c0, helix_thrust_b = helix_thrust_b0, helix_thrust_c = helix_thrust_c0)
    p0 = helix_power_ct_function(u=np.array([u]), run_only=0, helix_amp=np.array([0.]), helix_a = helix_a, helix_power_b = helix_power_b, helix_power_c = helix_power_c0, helix_thrust_b = helix_thrust_b0, helix_thrust_c = helix_thrust_c0)
    return p/p0


# ==========================================================================================
# EXTRACT DATA =============================================================================
# ==========================================================================================

# extract data (from LES)
u = 8.
fil_u = df_helix_loss_LES['Wind Speed (m/s)']==u
helix_amp_array_LES = np.array([0.,1.,3.,5.])
p_array = np.zeros(len(helix_amp_array_LES))
operation_array = np.array(['BL','1D','3D','5D'])
for i in np.arange(len(helix_amp_array_LES)):
    fil_operation = df_helix_loss_LES['Operation']==operation_array[i]
    df_temp = df_helix_loss_LES[fil_u&fil_operation]
    p_array[i] = df_temp['Power (kW)'].tolist()[0]
p_loss_array_LES = p_array/p_array[0]

# extract data (from GP)
fil_St = df_helix_power_loss_GP['pitch_St']==0.25
df_temp = df_helix_power_loss_GP[fil_St]
helix_amp_array_GP = np.array(df_temp['pitch_amp'].tolist())
p_loss_array_GP = np.array(df_temp['power_loss'].tolist())
p_loss_array_GP[0] = 1.


# ==========================================================================================
# FIT DATA =================================================================================
# ==========================================================================================

# fit the data to LES
p0 = np.array([helix_a0,helix_power_b0])
popt, pcov = curve_fit(power_loss_function, helix_amp_array_LES, p_loss_array_LES, p0=p0, bounds=(0,np.inf))
helix_a_LEStuned = popt[0]
helix_power_b_LEStuned = popt[1]
print(f'Optimal coefficients (LES tuning): {popt}')
helix_amp_array_LEStuned = np.linspace(0,5.5,100,endpoint=True)
p_loss_LEStuned = power_loss_function(helix_amp_array_LEStuned,helix_a_LEStuned,helix_power_b_LEStuned)

# fit the data to GP
p0 = np.array([helix_a0,helix_power_b0])
popt, pcov = curve_fit(power_loss_function, helix_amp_array_GP, p_loss_array_GP, p0=p0, bounds=(0,np.inf))
helix_a_GPtuned = popt[0]
helix_power_b_GPtuned = popt[1]
print(f'Optimal coefficients (GP tuning): {popt}')
helix_amp_array_GPtuned = np.linspace(0,5.5,100,endpoint=True)
p_loss_GPtuned = power_loss_function(helix_amp_array_GPtuned,helix_a_GPtuned,helix_power_b_GPtuned)


# ==========================================================================================
# SAVE DATA ================================================================================
# ==========================================================================================

# save optimal coefficients (based on GP - St=0.25, U=10 m/s)
helix_a_opt = helix_a_GPtuned
helix_power_b_opt = helix_power_b_GPtuned
helix_power_c_opt = helix_power_c0


#%% TUNING CT CURVE

# create a wrapper function (8 m/s)
def ct_loss_function(helix_amp,helix_thrust_b):
    u = 8.
    u_array = np.ones(len(helix_amp))*u
    ct = helix_power_ct_function(u=u_array, run_only=1, helix_amp=helix_amp, helix_a = helix_a_opt, helix_power_b = helix_power_b_opt, helix_power_c = helix_power_c_opt, helix_thrust_b = helix_thrust_b, helix_thrust_c = helix_thrust_c0)
    ct0 = helix_power_ct_function(u=np.array([u]), run_only=1, helix_amp=np.array([0.]), helix_a = helix_a_opt, helix_power_b = helix_power_b_opt, helix_power_c = helix_power_c_opt, helix_thrust_b = helix_thrust_b, helix_thrust_c = helix_thrust_c0)
    return ct/ct0


# ==========================================================================================
# EXTRACT DATA =============================================================================
# ==========================================================================================

# extract data (from LES)
u = 8.
fil_u = df_helix_loss_LES['Wind Speed (m/s)']==u
helix_amp_array_LES = np.array([0.,1.,3.,5.])
t_array = np.zeros(len(helix_amp_array_LES))
operation_array = np.array(['BL','1D','3D','5D'])
for i in np.arange(len(helix_amp_array_LES)):
    fil_operation = df_helix_loss_LES['Operation']==operation_array[i]
    df_temp = df_helix_loss_LES[fil_u&fil_operation]
    t_array[i] = df_temp['Thrust (kN)'].tolist()[0]
ct_loss_array_LES = t_array/t_array[0]

# extract data (from GP)
fil_St = df_helix_thrust_loss_GP['pitch_St']==0.25
df_temp = df_helix_thrust_loss_GP[fil_St]
helix_amp_array_GP = np.array(df_temp['pitch_amp'].tolist())
ct_loss_array_GP = np.array(df_temp['thrust_loss'].tolist())
ct_loss_array_GP[0] = 1.


# ==========================================================================================
# FIT DATA =================================================================================
# ==========================================================================================

# fit the data to LES
p0 = np.array([helix_thrust_b0])
popt, pcov = curve_fit(ct_loss_function, helix_amp_array_LES, ct_loss_array_LES, p0=p0, bounds=(0,np.inf))
helix_thrust_b_LEStuned = popt[0]
print(f'Optimal coefficients (LES tuning): {popt}')
helix_amp_array_LEStuned = np.linspace(0,5.5,100,endpoint=True)
ct_loss_LEStuned = ct_loss_function(helix_amp_array_LEStuned,helix_thrust_b_LEStuned)

# fit the data to GP
p0 = np.array([helix_thrust_b0])
popt, pcov = curve_fit(ct_loss_function, helix_amp_array_GP, ct_loss_array_GP, p0=p0, bounds=(0,np.inf))
helix_thrust_b_GPtuned = popt[0]
print(f'Optimal coefficients (GP tuning): {popt}')
helix_amp_array_GPtuned = np.linspace(0,5.5,100,endpoint=True)
ct_loss_GPtuned = ct_loss_function(helix_amp_array_GPtuned,helix_thrust_b_GPtuned)


# ==========================================================================================
# SAVE DATA ================================================================================
# ==========================================================================================

# save optimal coefficients (based on GP - St=0.25, U=10 m/s)
helix_thrust_b_opt = helix_thrust_b_GPtuned
helix_thrust_c_opt = helix_thrust_c0

#%% PRINT OPTIMAL RESULTS

print('Optimal coefficients (tuned on GP data)')
print(f'Helix a: \t \t \t {helix_a_opt}')
print(f'Helix power b: \t \t {helix_power_b_opt}')
print(f'Helix power c: \t \t {helix_power_c_opt}')
print(f'Helix thrust b: \t {helix_thrust_b_opt}')
print(f'Helix thrust c: \t {helix_thrust_c_opt}')


#%% PLOT

savefig = False
#name_path = r"C:\Users\matteobaricchi\OneDrive - Delft University of Technology\Desktop\helix codesign layout optimization\pywake_mixed_operation\figures\\"
name_path = r"..\figures_WES_paper\\"

colors = ['#001221','#538de5','#41c3d3','#ea9bd5','#ff9887']

#plt.title('Power loss')
plt.plot(helix_amp_array_floris,p_loss_floris,c=colors[1],linestyle='dashed',label='FLORIS default coefficients')
plt.plot(helix_amp_array_GPtuned,p_loss_GPtuned,c=colors[0],label='Tuned coefficients for IEA 22 MW')
plt.scatter(helix_amp_array_GP,p_loss_array_GP,c=colors[3],label='OpenFAST/LES data for IEA 22 MW',marker='o')
plt.legend()
plt.xlabel('Helix amplitude [deg]')
plt.ylabel(r'$P\,/\,P_{\mathrm{baseline}}$')
if savefig: plt.savefig(name_path+'power_loss_tuning.pdf',format='pdf')
plt.show()

#plt.title('Thrust loss')
plt.plot(helix_amp_array_floris,ct_loss_floris,c=colors[1],linestyle='dashed',label='FLORIS default coefficients')
plt.plot(helix_amp_array_GPtuned,ct_loss_GPtuned,c=colors[0],label='Tuned coefficients for IEA 22 MW')
plt.scatter(helix_amp_array_GP,ct_loss_array_GP,c=colors[3],label='OpenFAST/LES data for IEA 22 MW',marker='o')
plt.xlabel('Helix amplitude [deg]')
plt.ylabel(r'$C_t\,/\,C_{t,\mathrm{baseline}}$')
if savefig: plt.savefig(name_path+'thrust_loss_tuning.pdf',format='pdf')
plt.legend()
plt.show()


# plt.title('Power loss')
# plt.plot(helix_amp_array_floris,p_loss_floris,c='b',linestyle='dashed',label='FLORIS function')
# plt.plot(helix_amp_array_LEStuned,p_loss_LEStuned,c='k',label='LEStuned function')
# plt.plot(helix_amp_array_GPtuned,p_loss_GPtuned,c='c',label='GPtuned function')
# plt.scatter(helix_amp_array_LES,p_loss_array_LES,c='g',label='LES',marker='*')
# plt.scatter(helix_amp_array_GP,p_loss_array_GP,c='r',label='GP',marker='.')
# plt.legend()
# plt.xlabel('Helix amplitude [deg]')
# plt.ylabel('P/P_baseline')
# if savefig: plt.savefig(name_path+'power_loss_tuning.svg',format='svg')
# plt.show()

# plt.title('Thrust loss')
# plt.plot(helix_amp_array_floris,ct_loss_floris,c='b',linestyle='dashed',label='FLORIS function')
# plt.plot(helix_amp_array_LEStuned,ct_loss_LEStuned,c='k',label='LEStuned function')
# plt.plot(helix_amp_array_GPtuned,ct_loss_GPtuned,c='c',label='GPtuned function')
# plt.scatter(helix_amp_array_LES,ct_loss_array_LES,c='g',label='LES',marker='*')
# plt.scatter(helix_amp_array_GP,ct_loss_array_GP,c='r',label='GP',marker='.')
# plt.xlabel('Helix amplitude [deg]')
# plt.ylabel('Ct/Ct_baseline')
# if savefig: plt.savefig(name_path+'thrust_loss_tuning.svg',format='svg')
# plt.legend()
# plt.show()



#%% CHECK DEPENDENCE ON WIND SPEED



u_test = np.array([7.,8.,9.,10.])
helix_amp_test = np.arange(0,5.5,0.5)

p_mat_test = np.zeros((len(u_test),len(helix_amp_test)))
ct_mat_test = np.zeros((len(u_test),len(helix_amp_test)))


for i in np.arange(len(u_test)):
    
    u = np.ones(len(helix_amp_test))*u_test[i]
    p_mat_test[i,:] = helix_power_ct_function(u,
                                              run_only = 0,
                                              helix_amp = helix_amp_test,
                                              helix_a = helix_a_opt,
                                              helix_power_b = helix_power_b_opt,
                                              helix_power_c = helix_power_c_opt,
                                              helix_thrust_b = helix_thrust_b_opt,
                                              helix_thrust_c = helix_thrust_c_opt,
                                              )
    ct_mat_test[i,:] = helix_power_ct_function(u,
                                                run_only = 1,
                                                helix_amp = helix_amp_test,
                                                helix_a = helix_a_opt,
                                                helix_power_b = helix_power_b_opt,
                                                helix_power_c = helix_power_c_opt,
                                                helix_thrust_b = helix_thrust_b_opt,
                                                helix_thrust_c = helix_thrust_c_opt,
                                              )




# for i in np.arange(len(u_test)):
    
#     u = np.ones(len(helix_amp_test))*u_test[i]
#     p_mat_test[i,:] = helix_power_ct_function(u,
#                                               run_only = 0,
#                                               helix_amp = helix_amp_test,
#                                               helix_a = helix_a0,
#                                               helix_power_b = helix_power_b0,
#                                               helix_power_c = helix_power_c0,
#                                               helix_thrust_b = helix_thrust_b0,
#                                               helix_thrust_c = helix_thrust_c0,
#                                               )
#     ct_mat_test[i,:] = helix_power_ct_function(u,
#                                                 run_only = 1,
#                                                 helix_amp = helix_amp_test,
#                                                 helix_a = helix_a0,
#                                                 helix_power_b = helix_power_b0,
#                                                 helix_power_c = helix_power_c0,
#                                                 helix_thrust_b = helix_thrust_b0,
#                                                 helix_thrust_c = helix_thrust_c0,
#                                               )









colors = ['r','g','b','k']

plt.title('Power loss')
for i in np.arange(len(u_test)):
    plt.plot(helix_amp_test,p_mat_test[i,:]/p_mat_test[i,0],c=colors[i],label=f'ws = {u_test[i]} m/s')
plt.legend()
plt.grid('on')
plt.show()

plt.title('Thrust loss')
for i in np.arange(len(u_test)):
    plt.plot(helix_amp_test,ct_mat_test[i,:]/ct_mat_test[i,0],c=colors[i],label=f'ws = {u_test[i]} m/s')
plt.legend()
plt.grid('on')
plt.show()



#%%


# # extract data (from LES)
# u = 9.
# fil_u = df_helix_loss_LES['Wind Speed (m/s)']==u
# helix_amp_array_LES = np.array([0.,1.,3.,5.])
# p_array = np.zeros(len(helix_amp_array_LES))
# operation_array = np.array(['BL','1D','3D','5D'])
# for i in np.arange(len(helix_amp_array_LES)):
#     fil_operation = df_helix_loss_LES['Operation']==operation_array[i]
#     df_temp = df_helix_loss_LES[fil_u&fil_operation]
#     p_array[i] = df_temp['Power (kW)'].tolist()[0]
# p_loss_array_LES_9 = p_array/p_array[0]

# u = 8.
# fil_u = df_helix_loss_LES['Wind Speed (m/s)']==u
# helix_amp_array_LES = np.array([0.,1.,3.,5.])
# p_array = np.zeros(len(helix_amp_array_LES))
# operation_array = np.array(['BL','1D','3D','5D'])
# for i in np.arange(len(helix_amp_array_LES)):
#     fil_operation = df_helix_loss_LES['Operation']==operation_array[i]
#     df_temp = df_helix_loss_LES[fil_u&fil_operation]
#     p_array[i] = df_temp['Power (kW)'].tolist()[0]
# p_loss_array_LES_8 = p_array/p_array[0]


# u = 7.
# fil_u = df_helix_loss_LES['Wind Speed (m/s)']==u
# helix_amp_array_LES = np.array([0.,1.,3.,5.])
# p_array = np.zeros(len(helix_amp_array_LES))
# operation_array = np.array(['BL','1D','3D','5D'])
# for i in np.arange(len(helix_amp_array_LES)):
#     fil_operation = df_helix_loss_LES['Operation']==operation_array[i]
#     df_temp = df_helix_loss_LES[fil_u&fil_operation]
#     p_array[i] = df_temp['Power (kW)'].tolist()[0]
# p_loss_array_LES_7 = p_array/p_array[0]


# u = 6.
# fil_u = df_helix_loss_LES['Wind Speed (m/s)']==u
# helix_amp_array_LES = np.array([0.,1.,3.,5.])
# p_array = np.zeros(len(helix_amp_array_LES))
# operation_array = np.array(['BL','1D','3D','5D'])
# for i in np.arange(len(helix_amp_array_LES)):
#     fil_operation = df_helix_loss_LES['Operation']==operation_array[i]
#     df_temp = df_helix_loss_LES[fil_u&fil_operation]
#     p_array[i] = df_temp['Power (kW)'].tolist()[0]
# p_loss_array_LES_6 = p_array/p_array[0]



# plt.title('Power loss')
# plt.plot(helix_amp_array_floris,p_loss_floris,c='k',linestyle='dashed',label='FLORIS function (8 ms)')
# plt.plot(helix_amp_array_GPtuned,p_loss_GPtuned,c='k',label='GPtuned function (8 m/s)')
# plt.scatter(helix_amp_array_LES,p_loss_array_LES_9,c='g',label='LES (9 ms)',marker='*')
# plt.scatter(helix_amp_array_LES,p_loss_array_LES_8,c='y',label='LES (8 ms)',marker='*')
# plt.scatter(helix_amp_array_LES,p_loss_array_LES_7,c='r',label='LES (7 ms)',marker='*')
# plt.scatter(helix_amp_array_LES,p_loss_array_LES_6,c='c',label='LES (6 ms)',marker='*')
# plt.legend()
# plt.show()
