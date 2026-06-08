
# import main packages
import numpy as np
from scipy.interpolate import interp1d
from numpy import newaxis as na

# import py_wake packages
from py_wake.wind_farm_models.engineering_models import EngineeringWindFarmModel
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.superposition_models import WeightedSum, CumulativeWakeSum
from py_wake.deficit_models.deficit_model import WakeDeficitModel
from py_wake.deflection_models import DeflectionModel


# import other packages
from tqdm import tqdm



#%% POWER CT FUNCTION

def helix_power_ct_function(u,
                            run_only,
                            helix_amp=0,
                            helix_a = 1.809,
                            helix_power_b = 4.828e-03,
                            helix_power_c = 4.017e-11,
                            helix_thrust_b = 1.390e-03,
                            helix_thrust_c = 5.084e-04,
                            ):

    ws_turbine = np.array([0,2,3,4,5,6,7,8,9,10,11,12, 13,14,15,16,17,18,19,20,21,22,23,24,25])
    p_baseline = np.array([0, 0,466, 1103, 2158, 3722, 5878, 8657, 12110, 16910, 22000, 22000, 22000, 22000, 22000, 22000, 22000, 22000, 22000, 22000, 22000,22000 ,22000 ,22000 ,22000])
    ct_baseline = np.array([0, 0,0.8234, 0.8298, 0.8459, 0.8646, 0.8767, 0.8866, 0.8861, 0.8099, 0.7569, 0.45, 0.34, 0.27, 0.24, 0.2, 0.17, 0.13, 0.10, 0.09, 0.075,0.065, 0.06, 0.05, 0.046])
    
    # flatten input
    dims = u.shape
    u_m = u.flatten()
    
    if (type(helix_amp) == int) | (type(helix_amp) == float):
        helix_amp_m = np.ones_like(u_m)*helix_amp
    elif u.shape == helix_amp.shape:
        helix_amp_m = helix_amp.flatten()
    else:
        print('Error')
    
    # extract dimensions
    M = len(u_m)
        
    # initialize variables
    p_m = np.zeros(M)
    ct_m = np.zeros(M)
    
    # extract the values of helix amplitude
    helix_amp_unique = np.unique(helix_amp_m)
    
    # iterate for each helix amplitude
    for h in np.arange(len(helix_amp_unique)):
        
        # create filter
        fil = helix_amp_m==helix_amp_unique[h]
        
        # calculate power and ct curves based on helix_amp
        p_awc = p_baseline*(1-(helix_power_b+helix_power_c*p_baseline*1e3)*(helix_amp_unique[h]**helix_a))
        ct_awc = ct_baseline*(1-(helix_thrust_b+helix_thrust_c*ct_baseline)*(helix_amp_unique[h]**helix_a)) 
        
        # define interpolation functions
        interp_func_p = interp1d(ws_turbine,p_awc,kind='linear',bounds_error=False,fill_value=0)
        interp_func_ct = interp1d(ws_turbine,ct_awc,kind='linear',bounds_error=False,fill_value=0)

        # calculate power and ct
        p_m[fil] = interp_func_p(u_m[fil])
        ct_m[fil] = interp_func_ct(u_m[fil])

    # restore original dimensions
    p = np.reshape(p_m,dims)
    ct = np.reshape(ct_m,dims)

    # define output
    if run_only==0:
        return p
    elif run_only==1:
        return ct
    
    
    
    
#%% MODIFIED PY_WAKE CLASS: PropagateDownwind


class PropagateDownwind_helix(PropagateDownwind):

    def _propagate_deficit(self, wd,
                           wt_order_indices_ld,
                           WS_ilk,
                           TI_eff_ilk,
                           D_i,
                           I, L, K, **kwargs):
        """
        Additional suffixes:

        - m: turbines and wind directions (il.flatten())
        - n: from_turbines, to_turbines and wind directions (iil.flatten())

        """

        deficit_nk = []
        blockage_nk = []
        uc_nk = []
        sigma_sqr_nk = []
        cw_nk = []
        hcw_nk = []
        dh_nk = []

        def ilk2mk(v_ilk):
            dtype = (float, np.complex128)[np.iscomplexobj(v_ilk)]
            _K = np.shape(v_ilk)[2]
            return np.broadcast_to(np.asarray(v_ilk).astype(dtype), (I, L, _K)).reshape((I * L, _K))

        WS_mk = ilk2mk(WS_ilk)
        WD_mk, TI_mk, h_mk = [ilk2mk(kwargs[k + '_ilk']) for k in ['WD', 'TI', 'h']]
        WS_eff_mk, TI_eff_mk = [], []
        yaw_mk = ilk2mk(kwargs.get('yaw_ilk', [[[0]]]))
        tilt_mk = ilk2mk(kwargs.get('tilt_ilk', [[[0]]]))        
        helix_amp_mk = ilk2mk(kwargs.get('helix_amp_ilk', [[[0]]]))                                                 # added line
        modified_input_dict_mk = []
        WS_free_mk = []
        D_mk = []
        ct_jlk = []
        
        # initialize mixing based on awc
        mixing_tot_ilk = self.wake_deficitModel._awc_added_mixing(kwargs.get('helix_amp_ilk', [[[0]]]))              # added line
        mixing_tot_mk = ilk2mk(mixing_tot_ilk).copy()                                                                # added line

        if self.turbulenceModel:
            add_turb_nk = []

        i_wd_l = np.arange(L).astype(int)

        wt_kwargs = self.get_wt_kwargs(TI_eff_ilk, kwargs)

        # Iterate over turbines in down wind order
        for j in tqdm(range(I), disable=I <= 1 or not self.verbose, desc="Calculate flow interaction", unit="wt"):
            i_wt_l = wt_order_indices_ld[:, j]
            # current wt (j'th most upstream wts for all wdirs)
            m = i_wt_l * L + i_wd_l

            # Calculate effectiv wind speed at current turbines(all wind directions and wind speeds) and
            # look up power and thrust coefficient
            if j == 0:  # Most upstream turbines (no wake)
                WS_eff_lk = WS_mk[m]
                WS_eff_mk.append(WS_eff_lk)
                if self.turbulenceModel:
                    TI_eff_lk = TI_mk[m]
                    TI_eff_mk.append(np.broadcast_to(TI_eff_lk, (L, K)))
            else:  # 2..n most upstream turbines (wake)
                def get_value2WT(value_nk):
                    """Get value input to current turbine, j
                    value_nk triangular is a list j elements.
                    First element contains e.g. the defict to
                    """
                    return np.array([d_nk2[i] for d_nk2, i in zip(value_nk, range(j)[::-1])])

                sp_kwargs = {'deficit_jxxx': get_value2WT(deficit_nk)}
                if isinstance(self.superpositionModel, (WeightedSum, CumulativeWakeSum)):
                    sp_kwargs.update({k: get_value2WT(v_nk) for k, v_nk in [('sigma_sqr_jxxx', sigma_sqr_nk),
                                                                            ('cw_jxxx', cw_nk),
                                                                            ('hcw_jxxx', hcw_nk),
                                                                            ('dh_jxxx', dh_nk)]})

                    if isinstance(self.superpositionModel, WeightedSum):
                        sp_kwargs.update({'WS_xxx': WS_mk[m],
                                          'convection_velocity_jxxx': get_value2WT(uc_nk)})
                    else:
                        sp_kwargs.update({'WS0_xxx': np.array(WS_free_mk),
                                          'WS_eff_xxx': np.array(WS_eff_mk),
                                          'ct_xxx': np.array(ct_jlk),
                                          'D_xx': np.array(D_mk)})

                WS_eff_lk = WS_mk[m] - self.superpositionModel.superpose_deficit(**sp_kwargs)
                if self.blockage_deficitModel:
                    WS_eff_lk -= self.blockage_superpositionModel(get_value2WT(blockage_nk))
                WS_eff_mk.append(WS_eff_lk)

                if self.turbulenceModel:
                    add_turb2WT = np.array([d_nk2[i] for d_nk2, i in zip(add_turb_nk, range(j)[::-1])])
                    TI_eff_lk = self.turbulenceModel.calc_effective_TI(TI_mk[m], add_turb2WT)
                    TI_eff_mk.append(TI_eff_lk)
            # assemble free wind speed (ask mmpe why it is not) and diameter to allow cumulative superposition
            WS_free_mk.append(WS_mk[m])
            D_mk.append(D_i[i_wt_l])

            # Calculate Power/CT
            def mask(k, v):
                if len(np.squeeze(v).shape) == 0:
                    return np.squeeze(v)
                v = np.asarray(v)
                if v.shape[:2] == (I, L):
                    return v[i_wt_l, i_wd_l]
                elif v.shape[0] == I:
                    return v[i_wt_l].flatten()
                else:
                    assert v.shape[1] == L
                    return v[0, i_wd_l]

            _wt_kwargs = {k: mask(k, v) for k, v in wt_kwargs.items()}
            if 'TI_eff' in _wt_kwargs:
                _wt_kwargs['TI_eff'] = TI_eff_mk[-1]

            ct_lk = self.windTurbines.ct(WS_eff_lk, **_wt_kwargs)

            ct_jlk.append(ct_lk)

            if j < I - 1 or len(self.inputModifierModels):
                i_dw = wt_order_indices_ld[:, j + 1:]

                # Calculate required args4deficit parameters
                arg_funcs = {'WS_ilk': lambda: WS_mk[m][na],
                             'WS_jlk': lambda: np.moveaxis(np.array(
                                 [WS_ilk[(slice(0, 1), j)[WS_ilk.shape[0] > 1], (0, l)[WS_ilk.shape[1] > 1]]
                                  for j, l in zip(i_dw, i_wd_l)]), 0, 1),
                             'WS_eff_ilk': lambda: WS_eff_mk[-1][na],
                             'TI_ilk': lambda: TI_mk[m][na],
                             'TI_eff_ilk': lambda: TI_eff_mk[-1][na],
                             'D_src_il': lambda: D_i[i_wt_l][na],
                             'yaw_ilk': lambda: yaw_mk[m][na],
                             'tilt_ilk': lambda: tilt_mk[m][na],
                             'helix_amp_ilk': lambda: helix_amp_mk[m][na],                                          # added line
                             'mixing_tot_ilk_': lambda: mixing_tot_mk[m][na],                                       # added line
                             'D_dst_ijl': lambda: D_i[wt_order_indices_ld[:, j + 1:]].T[na],
                             'h_ilk': lambda: h_mk[m][na],
                             'ct_ilk': lambda: ct_lk[na],
                             'IJLK': lambda: (1, i_dw.shape[1], L, K),
                             'WD_ilk': lambda: WD_mk[m][na],
                             **{k + '_ilk': lambda k=k: ilk2mk(kwargs[k + '_ilk'])[m][na] for k in 'xyh'},
                             'type_il': lambda: kwargs['type_i'][i_wt_l][na]

                             }
                model_kwargs = {k: arg_funcs[k]() for k in self.args4all if k in arg_funcs}
                
                # custom model arguments
                custom_args = (set([k for k in self.args4all if k.endswith('_ilk')]) - set(model_kwargs)) & set(kwargs)
                model_kwargs.update({k: ilk2mk(kwargs[k])[m][na] for k in custom_args})

                dw_ijlk, hcw_ijlk, dh_ijlk = self.site.distance(
                    wd_l=wd, WD_ilk=WD_mk[m][na], src_idx=i_wt_l, dst_idx=i_dw.T)
                
                for inputModidifierModel in self.inputModifierModels:
                    modified_input_dict = inputModidifierModel(**model_kwargs)
                    modified_input_dict_mk.append(modified_input_dict)
                    model_kwargs.update(modified_input_dict)

                if self.wec != 1:
                    hcw_ijlk = hcw_ijlk / self.wec

                if self.deflectionModel:
                    dw_ijlk, hcw_ijlk, dh_ijlk = self.deflectionModel.calc_deflection(
                        dw_ijlk=dw_ijlk, hcw_ijlk=hcw_ijlk, dh_ijlk=dh_ijlk, **model_kwargs)

                model_kwargs.update({'dw_ijlk': dw_ijlk, 'hcw_ijlk': hcw_ijlk, 'dh_ijlk': dh_ijlk})
                if 'z_ijlk' in self.args4all:
                    model_kwargs['z_ijlk'] = h_mk[m][na, na] + dh_ijlk

                hcw_nk.append(hcw_ijlk[0])
                dh_nk.append(dh_ijlk[0])

                if 'cw_ijlk' in self.args4all:
                    # sqrt(a**2+b**2) as hypot does not support complex numbers
                    model_kwargs['cw_ijlk'] = np.sqrt(dh_ijlk**2 + hcw_ijlk**2)
                    cw_nk.append(model_kwargs['cw_ijlk'][0])

                if 'wake_radius_ijl' in self.args4all:
                    model_kwargs['wake_radius_ijl'] = self.wake_deficitModel.wake_radius(**model_kwargs)[..., 0]

                if 'wake_radius_ijlk' in self.args4all:
                    # mixing is neglected for the computation of the wake radius                                    # added line
                    model_kwargs['wake_radius_ijlk'] = self.wake_deficitModel.wake_radius(**model_kwargs)
                    
                    
                    
                # ======================================================================================================
                # Calculate wake-induced-mixing 
                # ======================================================================================================

                # calculate wake-induced-mixing by the iterating tubrine to the downstream turbines                 # added line
                mixing_ijlk = self.wake_deficitModel._calc_mixing_i_to_j(**model_kwargs)                            # added line
                                
                # flatten index of downstream turines ( f=jl.flatten() )                                            # added line
                i_dw_jl = (i_dw * L + i_wd_l[:,na]).T                                                               # added line
                i_dw_f = i_dw_jl.flatten()                                                                          # added line
                
                # flatten mixing_ijlk                                                                               # added line
                mixing_fk = mixing_ijlk[0].reshape(i_dw_jl.shape[0]*L,K)                                            # added line

                # sum up the wake-induced-mixing contribution of the j-th turbine                                   # added line
                mixing_tot_mk[i_dw_f] += mixing_fk                                                                  # added line
                
                # update main dictionary                                                                            # added line
                model_kwargs.update({'mixing_tot_ilk_':mixing_tot_mk[m][na]})                                       # added line
                

                # ======================================================================================================
                # Calculate deficit
                # ======================================================================================================
                if isinstance(self.superpositionModel, (WeightedSum, CumulativeWakeSum)):
                    # only cw needs to be rotor averaged as remaining super position input is
                    # the same all over the rotor
                    if self.wake_deficitModel.rotorAvgModel:
                        cw_nk[-1] = (self.wake_deficitModel.rotorAvgModel(lambda ** kwargs: kwargs['cw_ijlk'],
                                                                          **model_kwargs))[0]
                    if isinstance(self.superpositionModel, WeightedSum):
                        deficit, uc, sigma_sqr, _ = self._calc_deficit_convection(**model_kwargs)
                        uc_nk.append(uc[0])
                        sigma_sqr_nk.append(sigma_sqr[0])
                    elif isinstance(self.superpositionModel, CumulativeWakeSum):
                        # only sigma needed in cumulative wake model, centerline deficit computed inside superpostion model
                        # sigma set to zero upstream to ensure downwind activation only
                        sigma_sqr = (self.wake_deficitModel.sigma_ijlk(**model_kwargs))**2 * (dw_ijlk > 1e-10)
                        sigma_sqr_nk.append(sigma_sqr[0])
                        deficit = np.zeros_like(sigma_sqr)
                else:
                    deficit, blockage = self._calc_deficit(**model_kwargs)
                deficit_nk.append(deficit[0])
                if self.blockage_deficitModel:
                    blockage_nk.append(blockage[0])

                if self.turbulenceModel:

                    # Calculate added turbulence
                    add_turb_nk.append(self.turbulenceModel(**model_kwargs)[0])

        WS_eff_jlk, ct_jlk = np.array(WS_eff_mk), np.array(ct_jlk)

        wt_inv_indices = (np.argsort(wt_order_indices_ld, 1).T * L + np.arange(L).astype(int)[na]).flatten()
        WS_eff_ilk = WS_eff_jlk.reshape((I * L, K))[wt_inv_indices].reshape((I, L, K))
        
        ct_ilk = ct_jlk.reshape((I * L, K))[wt_inv_indices].reshape((I, L, K))
        if self.turbulenceModel:
            TI_eff_jlk = np.array(TI_eff_mk)
            TI_eff_ilk = TI_eff_jlk.reshape((I * L, K))[wt_inv_indices].reshape((I, L, K))

        if len(self.inputModifierModels):
            for k in modified_input_dict_mk[0].keys():
                mi_jlk = np.array([mi_dict[k] for mi_dict in modified_input_dict_mk])
                kwargs[k] = mi_jlk.reshape((I * L, K))[wt_inv_indices].reshape((I, L, K))
                
                
        return WS_eff_ilk, TI_eff_ilk, ct_ilk, kwargs


    def _calc_deficit(self, dw_ijlk, **kwargs):
        return EngineeringWindFarmModel._calc_deficit(self, dw_ijlk, **kwargs)

    def _calc_wt_interaction(self, wd, WS_eff_ilk, **kwargs):
        WS_ilk = kwargs.pop('WS_ilk')

        dw_order_indices_ld = self.site.distance.dw_order_indices(wd)[:, 0]
        return self._propagate_deficit(wd, dw_order_indices_ld, WS_ilk, **kwargs)
    
    

#%% EMPIRICAL GAUSSIAN DEFICIT MODEL


class EmpiricalGaussianDeficit(WakeDeficitModel):
    
    args4deficit = ['WS_ilk', 'dw_ijlk', 'cw_ijlk', 'hcw_ijlk', 'dh_ijlk', 'D_dst_ijl','ct_ilk','helix_amp_ilk', 'wake_radius_ijlk', 'mixing_tot_ilk_','WS_jlk','WS_eff_ilk']
    
    def __init__(self,
                 wake_expansion_rates = [0.023, 0.008],
                 breakpoints_D = [10],
                 sigma_0_D = 0.28,
                 smoothing_length_D = 2.0,
                 mixing_gain_velocity = 2.0,
                 awc_wake_exp = 1.2,
                 awc_wake_denominator = 400,
                 rotorAvgModel=None,
                 groundModel=None,
                 use_effective_ws=True,
                 #use_effective_ti=False
                 ):
        
        WakeDeficitModel.__init__(self,
                                  rotorAvgModel=rotorAvgModel,
                                  groundModel=groundModel,
                                  use_effective_ws=use_effective_ws)

        self.wake_expansion_rates = wake_expansion_rates
        self.breakpoints_D = breakpoints_D
        self.sigma_0_D = sigma_0_D
        self.smoothing_length_D = smoothing_length_D
        self.mixing_gain_velocity = mixing_gain_velocity
        self.awc_wake_exp = awc_wake_exp
        self.awc_wake_denominator = awc_wake_denominator
        
        
        
    def il2ijlk(self,v,J,K):
        return np.tile(v[:,na,:,na],(1,J,1,K))

    def il2ilk(self,v,K):
        return np.tile(v[:,:,na],(1,1,K))

    def ilk2ijlk(self,v,J):
        return np.tile(v[:,na,:,:],(1,J,1,1))

    def jlk2ijlk(self,v,I):
        return np.tile(v[na,:,:,:],(I,1,1,1))

    def ijl2ijlk(self,v,K):
        return np.tile(v[:,:,:,na],(1,1,1,K))




    def empirical_gauss_model_wake_width(self,
                                         dw_ijlk,
                                         D_src_il,
                                         wake_expansion_rates,
                                         breakpoints_D,
                                         sigma_0_ilk,
                                         smoothing_length_D,
                                         mixing_tot_final_ilk_,
                                         I,J,L,K
                                         ):
        
        # assumption: correct input format for wake_expansion_rates and breakpoints (no check)    
        sigma_ijlk = (wake_expansion_rates[0] + self.ilk2ijlk(mixing_tot_final_ilk_,J)) * dw_ijlk + self.ilk2ijlk(sigma_0_ilk,J)

        for bp_ind, bp in enumerate(breakpoints_D):
            
            sigma_ijlk += (wake_expansion_rates[bp_ind+1] - wake_expansion_rates[bp_ind]) * self.sigmoid_integral(np.tile(dw_ijlk,(1,1,1,K)), center_ijlk=bp*self.il2ijlk(D_src_il,J,K), width_ijlk=smoothing_length_D*self.il2ijlk(D_src_il,J,K))

        return sigma_ijlk




    def sigmoid_integral(self, dw_ijlk, center_ijlk, width_ijlk):
        
        # intialize smoothed distance
        dw_smooth_ijlk = np.zeros_like(dw_ijlk)
        
        # define smoothed distance above the smoothing zone
        above_smoothing_zone = (dw_ijlk-center_ijlk) > width_ijlk/2
        dw_smooth_ijlk[above_smoothing_zone] = (dw_ijlk-center_ijlk)[above_smoothing_zone]
        
        # define smoothed distance inside the smoothing zone
        in_smoothing_zone = ((dw_ijlk-center_ijlk) >= -width_ijlk/2) & ((dw_ijlk-center_ijlk) <= width_ijlk/2)
        dw_smooth_ijlk[in_smoothing_zone] = (width_ijlk*(((dw_ijlk-center_ijlk)/width_ijlk + 0.5)**6 - 3*((dw_ijlk-center_ijlk)/width_ijlk + 0.5)**5 + 5/2*((dw_ijlk-center_ijlk)/width_ijlk + 0.5)**4))[in_smoothing_zone]
        
        return dw_smooth_ijlk
    


    def _sigma_ijlk(self,dw_ijlk,D_src_il,mixing_tot_ilk_,sigma_0_ilk,I,J,L,K):
                
        return self.empirical_gauss_model_wake_width(dw_ijlk,
                                                D_src_il,
                                                self.wake_expansion_rates,
                                                self.breakpoints_D,                             # different from FLORIS (this is still the normalized value with D)
                                                sigma_0_ilk,
                                                self.smoothing_length_D,                        # different from FLORIS (this is still the normalized value with D)
                                                self.mixing_gain_velocity*mixing_tot_ilk_,
                                                I,J,L,K
                                                )

        
    #def sigma_ijlk(self,dw_ijlk,cw_ijlk,D_src_il,mixing_tot_ilk_,yaw_ilk,tilt_ilk,**kwargs):
    def sigma_ijlk(self,dw_ijlk,D_src_il,mixing_tot_ilk_,yaw_ilk,tilt_ilk,**kwargs):
        
        # extract dimensions
        I,J,L,_ = dw_ijlk.shape
        K = mixing_tot_ilk_.shape[2]
        
        # initial wake widths
        sigma_y0_ilk = self.sigma_0_D * self.il2ilk(D_src_il,K) * np.cos(np.radians(yaw_ilk))
        sigma_z0_ilk = self.sigma_0_D * self.il2ilk(D_src_il,K) * np.cos(np.radians(tilt_ilk))
                
        # wake expansion in the lateral (y) and vertical (z) directions
        sigma_y_ijlk = self._sigma_ijlk(dw_ijlk,D_src_il,mixing_tot_ilk_,sigma_y0_ilk,I,J,L,K)
        sigma_z_ijlk = self._sigma_ijlk(dw_ijlk,D_src_il,mixing_tot_ilk_,sigma_z0_ilk,I,J,L,K)
        
        # calculate average sigma (PRINCIPLE: area ellipse = area circle)
        return np.sqrt(sigma_y_ijlk*sigma_z_ijlk)     # geometric mean 

    

    def calc_deficit(self, WS_ilk, dw_ijlk, cw_ijlk, hcw_ijlk, dh_ijlk, ct_ilk, D_src_il, yaw_ilk, tilt_ilk, helix_amp_ilk, mixing_tot_ilk_, **kwargs):
                

        # differences from FLORIS:        
        # - MASK ARE NOT IMPLEMENTED : upstream_mask and downstream_mask
        # - VEER IS NEGLECTED
                
        # extract dimensions
        I,J,L,K = cw_ijlk.shape
        
        # initial wake widths
        sigma_y0_ilk = self.sigma_0_D * self.il2ilk(D_src_il,K) * np.cos(np.radians(yaw_ilk))
        sigma_z0_ilk = self.sigma_0_D * self.il2ilk(D_src_il,K) * np.cos(np.radians(tilt_ilk))
        
        # wake expansion in the lateral (y) and vertical (z) directions
        sigma_y_ijlk = self._sigma_ijlk(dw_ijlk,D_src_il,mixing_tot_ilk_,sigma_y0_ilk,I,J,L,K)
        sigma_z_ijlk = self._sigma_ijlk(dw_ijlk,D_src_il,mixing_tot_ilk_,sigma_z0_ilk,I,J,L,K)
        
        # scaling factor for the Gaussian curve
        ct_ijlk = self.ilk2ijlk(ct_ilk,J)
        c_norm = 1/(8*(self.sigma_0_D)**2)
        sigma_y0_ijlk = self.ilk2ijlk(sigma_y0_ilk,J)
        sigma_z0_ijlk = self.ilk2ijlk(sigma_z0_ilk,J)
        C_ijlk = c_norm * ( 1- np.sqrt( 1- (sigma_y0_ijlk*sigma_z0_ijlk*ct_ijlk)/(sigma_y_ijlk*sigma_z_ijlk) ) )
        
        # exponent of the Gaussian curve
        #dh_ijlk = ijl2ijlk(dh_ijl,K)
        exponent_ijlk = (hcw_ijlk**2)/(2*sigma_y_ijlk**2) + (dh_ijlk**2)/(2*sigma_z_ijlk**2)
        
        # calculate deficit
        deficit_ijlk = self.ilk2ijlk(WS_ilk, J) * C_ijlk * np.exp(-exponent_ijlk)
                
        return deficit_ijlk
    
    
    
    def wake_radius(self,dw_ijlk,D_src_il,mixing_tot_ilk_,yaw_ilk,tilt_ilk,**kwargs):
        
        # average sigma 
        # sigma_ijlk = self.sigma_ijlk(dw_ijlk,cw_ijlk,D_src_il,mixing_tot_ilk_,yaw_ilk,tilt_ilk)
        sigma_ijlk = self.sigma_ijlk(dw_ijlk,D_src_il,mixing_tot_ilk_,yaw_ilk,tilt_ilk)
        
        # calculate radius (r = 2*sigma)
        return 2.0 * sigma_ijlk
    


    def _calc_mixing_i_to_j(self,D_dst_ijl,dw_ijlk,cw_ijlk,ct_ilk,wake_radius_ijlk,helix_amp_ilk,**kwargs):
        
        # this function calculates the wake-induced-mixing at point j caused by a turbine i 
        
        # extract dimensions
        I,J,L,K = cw_ijlk.shape
        
        # extract the radius of the destination turbine at point j
        R_ijlk = self.ijl2ijlk(D_dst_ijl,K)/2
                           
        # CASE 1: no area overlap
        fil_no_overlap = cw_ijlk >= wake_radius_ijlk + R_ijlk
        A_overlap_ijlk = np.zeros_like(cw_ijlk)
        
        # CASE 2: full area overlap
        fil_full_overlap = cw_ijlk <= np.abs(wake_radius_ijlk-R_ijlk)
        A_overlap_ijlk[fil_full_overlap] = (np.pi * (np.minimum(wake_radius_ijlk,R_ijlk))**2)[fil_full_overlap]
        
        # CASE 3: partial area overlap
        fil_partial_overlap = ~ (fil_no_overlap | fil_full_overlap)
        dist_wake = ( wake_radius_ijlk**2 - R_ijlk**2 + cw_ijlk**2 )[fil_partial_overlap] / ( 2 * cw_ijlk )[fil_partial_overlap]
        dist_turb = ( R_ijlk**2 - wake_radius_ijlk**2 + cw_ijlk**2 )[fil_partial_overlap] / ( 2 * cw_ijlk )[fil_partial_overlap]
        A_overlap_wake = (wake_radius_ijlk[fil_partial_overlap])**2 * np.arccos( dist_wake / wake_radius_ijlk[fil_partial_overlap] ) - dist_wake * np.sqrt( (wake_radius_ijlk[fil_partial_overlap])**2 - dist_wake**2 )
        A_overlap_turb = (R_ijlk[fil_partial_overlap])**2 * np.arccos( dist_turb / R_ijlk[fil_partial_overlap] ) - dist_turb * np.sqrt( (R_ijlk[fil_partial_overlap])**2 - dist_turb**2 )
        A_overlap_ijlk[fil_partial_overlap] = A_overlap_wake + A_overlap_turb
        
        # calculate aera overlap ratio
        A_overlap_ratio_ijlk = A_overlap_ijlk/(np.pi*(R_ijlk)**2)
                
        # calculate induction factor
        ct_ijlk = self.ilk2ijlk(ct_ilk,J)
        a_ijlk = (1 - np.sqrt(1 - ct_ijlk))/2
                
        # calculate wake-induced-mixing at point j caused by turbine i 
        #wim_i_to_j_ijlk = (( A_overlap_ratio_ijlk * a_ijlk ) / ( (dw_ijlk)/(2*R_ijlk) )) + ((self.ilk2ijlk(helix_amp_ilk,J))**self.awc_wake_exp)/self.awc_wake_denominator
        wim_i_to_j_ijlk = ( A_overlap_ratio_ijlk * a_ijlk ) / ( (dw_ijlk)/(2*R_ijlk) )
                
        return wim_i_to_j_ijlk
    
    
    
    def _awc_added_mixing(self,helix_amp_ilk):
        
        wim_awc_ilk = (helix_amp_ilk**self.awc_wake_exp)/self.awc_wake_denominator
        
        return wim_awc_ilk
        




#%% EMPIRICAL GAUSSIAN DEFLECTION MODEL


class EmpiricalGaussianDeflection(DeflectionModel):
    
    def __init__(self,
                 hcw_deflection_gain_D = 3.0,   # horizontal deflection gain (due to yaw)
                 dh_deflection_gain_D = 3.0,    # vertical deflection gain (due to tilt)
                 deflection_rate = 22.0,
                 mixing_gain_deflection = 0.0
                 ):
        
        self.hcw_deflection_gain_D = hcw_deflection_gain_D
        self.dh_deflection_gain_D = dh_deflection_gain_D
        self.deflection_rate = deflection_rate
        self.mixing_gain_deflection = mixing_gain_deflection
        
    
    

    def calc_deflection(self, dw_ijlk, hcw_ijlk, dh_ijlk, yaw_ilk, tilt_ilk, mixing_tot_ilk_, D_src_il, ct_ilk, **_):
        
        hcw_deflection_gain_il = self.hcw_deflection_gain_D*D_src_il
        dh_deflection_gain_il = self.dh_deflection_gain_D*D_src_il
        
        A_hcw_ilk = (hcw_deflection_gain_il[:,:,na] * ct_ilk * (yaw_ilk*(np.pi/180))) / (1 + self.mixing_gain_deflection * mixing_tot_ilk_)
        A_dh_ilk = (dh_deflection_gain_il[:,:,na] * ct_ilk * (tilt_ilk*(np.pi/180))) / (1 + self.mixing_gain_deflection * mixing_tot_ilk_)
        
        log_term_ijlk = np.log( ( (dw_ijlk/D_src_il[:,na,:,na]) - self.deflection_rate )/( (dw_ijlk/D_src_il[:,na,:,na]) + self.deflection_rate ) + 2 )
        
        hcw_deflection = A_hcw_ilk[:,na,:,:] * log_term_ijlk    # [m]
        dh_deflection = A_dh_ilk[:,na,:,:] * log_term_ijlk      # [m]
                
        hcw_ijlk = hcw_ijlk + hcw_deflection
        dh_ijlk = dh_ijlk + dh_deflection
        
        return dw_ijlk, hcw_ijlk, dh_ijlk





#%% TEST

# # import packages
# import matplotlib.pyplot as plt
# from py_wake.wind_turbines import WindTurbine
# from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
# from py_wake.site import UniformWeibullSite
# from py_wake.rotor_avg_models import GaussianOverlapAvgModel
# from py_wake.superposition_models import LinearSum



# # deifne site (HKN)
# wd_site = np.linspace(0,360,12,endpoint=False)
# p_wd_site = np.array([0.066,0.063,0.063,0.064,0.054,0.052,0.072,0.129,0.150,0.116,0.091,0.080])
# a_site = np.array([9.56,9.21,9.38,9.78,9.23,9.20,10.96,12.73,12.75,12.17,11.22,10.59])
# k_site = np.array([2.18,2.36,2.40,2.34,2.30,2.20,2.11,2.33,2.42,2.20,2.15,2.11])
# site = UniformWeibullSite(p_wd=p_wd_site,a=a_site,k=k_site,ti=0.04)

# # define turbine
# powerCtFunction = PowerCtFunction(
#     input_keys=['ws','helix_amp'],
#     power_ct_func=helix_power_ct_function,
#     power_unit='kW',
# )
# wind_turbine = WindTurbine(name='IEA22MW_helix',
#                 diameter=283.2,
#                 hub_height=170.0,
#                 powerCtFunction=powerCtFunction)    
# diameter = wind_turbine.diameter()

# # initialize wind farm model (empirical gaussian model)
# wfm_empgauss = PropagateDownwind_helix(site, wind_turbine,
#                         wake_deficitModel=EmpiricalGaussianDeficit(),
#                         superpositionModel=LinearSum(),
#                         deflectionModel=EmpiricalGaussianDeflection(),
#                         turbulenceModel=None,
#                         rotorAvgModel=GaussianOverlapAvgModel())



# # define layout
# x = np.array([0,4,8])*diameter
# y = np.array([0,0,0])*diameter

# # define wind speed and wind direction
# wd_array = np.array([270])
# ws_array = np.array([8])

# # extract dimensions
# I = len(x)
# L = len(wd_array)
# K = len(ws_array)

# # define helix amplitude
# helix_amp_ilk = np.zeros((I,L,K))
# helix_amp = np.arange(0,6)

# # define yaw angles
# yaw_ilk = np.zeros((I,L,K))


# ws_eff_array = np.zeros((I,len(helix_amp)))
# p_array = np.zeros((I,len(helix_amp)))
# ct_array = np.zeros((I,len(helix_amp)))

# for h_ind in np.arange(len(helix_amp)):
    
#     helix_amp_ilk[0,0,0] = helix_amp[h_ind]

#     # run simulation
#     simres = wfm_empgauss(x,y,wd=wd_array,ws=ws_array,yaw=yaw_ilk,tilt=0,helix_amp=helix_amp_ilk)

#     # extract results
#     ws_eff_ilk = simres.WS_eff_ilk
#     p_ilk = simres.power_ilk
#     ct_ilk = simres.ct_ilk
    
#     # assign values to main arrays
#     ws_eff_array[:,h_ind] = ws_eff_ilk[:,0,0]
#     p_array[:,h_ind] = p_ilk[:,0,0]/1e6
#     ct_array[:,h_ind] = ct_ilk[:,0,0]


# # plot

# colors_array = ['b','g','r','k','c']

# fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8), sharex=True)

# axes[0].set_title('3 aligned turbines (H-BL-BL) - 4 D spacing - 8 m/s')

# # effective wind speed
# axes[0].plot(helix_amp,ws_eff_array[0,:],c=colors_array[0],label='Turbine 1',marker='*')
# axes[0].plot(helix_amp,ws_eff_array[1,:],c=colors_array[1],label='Turbine 2',marker='*')
# axes[0].plot(helix_amp,ws_eff_array[2,:],c=colors_array[2],label='Turbine 3',marker='*')
# axes[0].set_ylabel('Wind speed [m/s]')
# axes[0].legend()
# axes[0].grid()

# # power
# axes[1].plot(helix_amp,p_array[0,:],c=colors_array[0],label='Turbine 1',marker='*')
# axes[1].plot(helix_amp,p_array[1,:],c=colors_array[1],label='Turbine 2',marker='*')
# axes[1].plot(helix_amp,p_array[2,:],c=colors_array[2],label='Turbine 3',marker='*')
# axes[1].plot(helix_amp,np.sum(p_array[:,:],axis=(0)),c=colors_array[3],label='Total',marker='*')
# axes[1].set_ylabel('Power [MW]')
# axes[1].legend()
# axes[1].grid()

# # power
# axes[2].plot(helix_amp,ct_array[0,:],c=colors_array[0],label='Turbine 1',marker='*')
# axes[2].plot(helix_amp,ct_array[1,:],c=colors_array[1],label='Turbine 2',marker='*')
# axes[2].plot(helix_amp,ct_array[2,:],c=colors_array[2],label='Turbine 3',marker='*')
# axes[2].set_ylabel('Ct [-]')
# axes[2].set_xlabel('Helix amplitude (turbine 1) [deg]')
# axes[2].legend()
# axes[2].grid()

# plt.tight_layout()
# plt.show()

    
    
