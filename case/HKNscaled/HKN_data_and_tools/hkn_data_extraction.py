
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression


from py_wake.site import XRSite
import utm





def extract_HKNsite(filename_site,filename_boundaries,filename_layout):

    # TODO: HUB HEIGHT CORRECTION FOR THE WIND SPEED (Weibull A parameter)
    # now the wind speed refers to the hub height of the 11MW reference case 
    
    # import databse
    df_hkn_site = pd.read_csv(filename_site)
    df_hkn_boundaries = pd.read_csv(filename_boundaries)
    df_hkn_layout = pd.read_csv(filename_layout)
    
    # ==================================================================================
    # process boundaries data
    # ==================================================================================
    
    #extraction and conversion from lat/lon to distances [m])
    ind_lon = np.arange(0,len(df_hkn_boundaries.columns.tolist()),2)
    ind_lat = np.arange(1,len(df_hkn_boundaries.columns.tolist())+1,2)
    hkn_boundaries_lon = np.array(df_hkn_boundaries).reshape(-1)[ind_lon]
    hkn_boundaries_lat = np.array(df_hkn_boundaries).reshape(-1)[ind_lat]
    hkn_boundaries_xy = utm.from_latlon(hkn_boundaries_lat,hkn_boundaries_lon)
    hkn_boundaries_x = hkn_boundaries_xy[0]
    hkn_boundaries_y = hkn_boundaries_xy[1]
    
    # extract layout
    hkn_wt_x = np.array(df_hkn_layout['x'])
    hkn_wt_y = np.array(df_hkn_layout['y'])
    
    
    # ==================================================================================
    # process site data
    # ==================================================================================
    
    # convert spatial coordinates from lat/lon to distances [m] -----------------------
    hkn_site_lon = np.array(df_hkn_site['geometry/coordinates/0'])
    hkn_site_lat = np.array(df_hkn_site['geometry/coordinates/1'])
    n_samples_lon = 30
    n_samples_lat = 33
    hkn_site_xy = utm.from_latlon(hkn_site_lat,hkn_site_lon)
    hkn_site_x = hkn_site_xy[0]
    hkn_site_y = hkn_site_xy[1]
    
    # extend domain (for extrapolation)
    fraction_extrap = 0.05  # define a fraction of extrapolation
    hkn_site_x_min = np.min(hkn_site_x)-fraction_extrap*(np.max(hkn_site_x)-np.min(hkn_site_x))
    hkn_site_x_max = np.max(hkn_site_x)+fraction_extrap*(np.max(hkn_site_x)-np.min(hkn_site_x))
    hkn_site_y_min = np.min(hkn_site_y)-fraction_extrap*(np.max(hkn_site_y)-np.min(hkn_site_y))
    hkn_site_y_max = np.max(hkn_site_y)+fraction_extrap*(np.max(hkn_site_y)-np.min(hkn_site_y))
    hkn_x_dim = np.linspace(hkn_site_x_min,hkn_site_x_max,n_samples_lon,endpoint=True)
    hkn_y_dim = np.linspace(hkn_site_y_min,hkn_site_y_max,n_samples_lat,endpoint=True)
    hkn_site_x_grid,hkn_site_y_grid = np.meshgrid(hkn_x_dim,hkn_y_dim,indexing='ij')
    
    
    # process bathymetry --------------------------------------------------------------
    
    hkn_site_bathymetry = np.array(df_hkn_site['properties/elevation'])
    hkn_site_bathymetry_grid = griddata((hkn_site_x,hkn_site_y),hkn_site_bathymetry,(hkn_site_x_grid,hkn_site_y_grid),method='nearest')
    
    
    # process wind resource data ------------------------------------------------------
    
    # define dimensions (cerate x,y grid for spatial information)
    n_sectors = 16
    hkn_wd_dim = np.linspace(0,360,n_sectors)
    
    # initialize output
    hkn_a_weibull_mat = np.zeros((len(hkn_x_dim),len(hkn_y_dim),len(hkn_wd_dim)))
    hkn_k_weibull_mat = np.zeros((len(hkn_x_dim),len(hkn_y_dim),len(hkn_wd_dim)))
    hkn_sec_freq_mat =  np.zeros((len(hkn_x_dim),len(hkn_y_dim),len(hkn_wd_dim)))
    
    # extract mean wind speed for each position
    hkn_ws_mean =  griddata((hkn_site_x,hkn_site_y),np.array(df_hkn_site['properties/wind_speed_mean']),(hkn_site_x_grid,hkn_site_y_grid),method='nearest')
    
    # extract values
    for i in np.arange(n_sectors):
        a_weibull_colname = f'properties/wind_weibull_a/{i}'
        k_weibull_colname = f'properties/wind_weibull_k/{i}'
        sec_freq_colname = f'properties/wind_sector_frequencies/{i}'
        
        # interpolate results for (x_dim,y_dim)
        hkn_a_weibull_mat[:,:,i] = griddata((hkn_site_x,hkn_site_y),np.array(df_hkn_site[a_weibull_colname]),(hkn_site_x_grid,hkn_site_y_grid),method='nearest')
        hkn_k_weibull_mat[:,:,i] = griddata((hkn_site_x,hkn_site_y),np.array(df_hkn_site[k_weibull_colname]),(hkn_site_x_grid,hkn_site_y_grid),method='nearest')
        hkn_sec_freq_mat[:,:,i] = griddata((hkn_site_x,hkn_site_y),np.array(df_hkn_site[sec_freq_colname]),(hkn_site_x_grid,hkn_site_y_grid),method='nearest')/100
    
    
    # ==========================================================================================
    # create pywake site 
    # ==========================================================================================
    
    ds_hkn = xr.Dataset(
        data_vars={
            'Sector_frequency':(['x','y','wd'],hkn_sec_freq_mat),
            'Weibull_A':(['x','y','wd'],hkn_a_weibull_mat),
            'Weibull_k':(['x','y','wd'],hkn_k_weibull_mat),
            'TI':0.1    
            },
        coords={
            'x':hkn_x_dim,
            'y':hkn_y_dim,
            'wd':hkn_wd_dim
            }
        )
    
    hkn_site = XRSite(ds_hkn)
    
    return hkn_site,hkn_ws_mean,hkn_site_bathymetry_grid,hkn_site_x_grid,hkn_site_y_grid,hkn_boundaries_x,hkn_boundaries_y,hkn_wt_x,hkn_wt_y



def extract_HKNprice(filename_prices,filename_wind_resources):
    
    # import databse
    df_prices_2030_hkn = pd.read_csv(filename_prices)
    df_weather_2012_hkn = pd.read_csv(filename_wind_resources)
    
    # extract price data
    price_timeseries = np.array(df_prices_2030_hkn['NL_R'])
    
    # extract weather data (the number indicates the hub height)
    ws_1_timeseries = np.array(df_weather_2012_hkn['WS_1'])
    ws_50_timeseries = np.array(df_weather_2012_hkn['WS_50'])
    ws_100_timeseries = np.array(df_weather_2012_hkn['WS_100'])
    ws_150_timeseries = np.array(df_weather_2012_hkn['WS_150'])
    ws_200_timeseries = np.array(df_weather_2012_hkn['WS_200'])
    wd_1_timeseries = np.array(df_weather_2012_hkn['WD_1'])
    wd_50_timeseries = np.array(df_weather_2012_hkn['WD_50'])
    wd_100_timeseries = np.array(df_weather_2012_hkn['WD_100'])
    wd_150_timeseries = np.array(df_weather_2012_hkn['WD_150'])
    wd_200_timeseries = np.array(df_weather_2012_hkn['WD_200'])
    
    # interpolation at hub height for the wind direction
    hub_height = 119.0
    wd_timeseries = np.zeros(len(wd_1_timeseries))
    for i in np.arange(len(ws_1_timeseries)):
        height_array = np.array([1,50,100,150,200])
        wd_array = np.array([wd_1_timeseries[i],wd_50_timeseries[i],wd_100_timeseries[i],wd_150_timeseries[i],wd_200_timeseries[i]])
        wd_function = interpolate.interp1d(height_array,wd_array,kind='cubic')
        wd_timeseries[i] = wd_function(hub_height)%360
    
    
    # regression at the hub height (using the shear power law) for the wind speed
    ws_timeseries = np.zeros(len(ws_1_timeseries))
    
    for i in np.arange(len(ws_1_timeseries)):
        
        height_array = np.array([1,50,100,150,200])
        ws_array = np.array([ws_1_timeseries[i],ws_50_timeseries[i],ws_100_timeseries[i],ws_150_timeseries[i],ws_200_timeseries[i]])
        ws_log = np.zeros(len(height_array)-1)
        h_log = np.zeros(len(height_array)-1)
        
        for j in np.arange(len(height_array)-1):
            ws_log[j] = np.log(ws_array[j+1]/ws_array[j])
            h_log[j] = np.log(height_array[j+1]/height_array[j])
    
        # find alpha
        model = LinearRegression()
        model.fit(h_log.reshape(-1, 1),ws_log)
        alpha = model.coef_[0]
        
        # find wind speed at hub height
        ws_timeseries[i] = ws_100_timeseries[i]*(hub_height/100)**alpha
    
    
    # SAMPLE PRICE FOR EACH FLOW CASE (combintation of wd and wd) --------------------------------------------
    # Problem: the results are biased by the fact that there is no price data for the least frequent flow cases
    
    # define wind speed and wind direction bins
    ws_bin_size_price = 5
    wd_bin_size_price = 30
    ws_array_price = np.arange(4,26,ws_bin_size_price)
    wd_array_price = np.arange(0,360,wd_bin_size_price)
    
    # assign to each flow case (combination) the correspondent price
    ws_ind_timeseries = -np.ones(len(ws_timeseries))
    wd_ind_timeseries = -np.ones(len(wd_timeseries))
    ws_lb = ws_array_price-ws_bin_size_price/2
    ws_ub = ws_array_price+ws_bin_size_price/2
    wd_lb = (wd_array_price-wd_bin_size_price/2)%360
    wd_ub = (wd_array_price+wd_bin_size_price/2)%360
    
    for i in np.arange(len(ws_timeseries)):
        if (ws_timeseries[i]>=np.min(ws_array_price))&(ws_timeseries[i]<=np.max(ws_array_price)):   # exclude the case above max ws
            ws_ind_timeseries[i]=np.where(((ws_timeseries[i]>=ws_lb)&(ws_timeseries[i]<ws_ub)))[0][0]
        if (wd_timeseries[i]>=wd_lb[0])|(wd_timeseries[i]<wd_ub[0]):    # check if it is the first wd bin (values around 0deg)
            wd_ind_timeseries[i] = 0
        else:
            wd_ind_timeseries[i]=np.where(((wd_timeseries[i]>=wd_lb)&(wd_timeseries[i]<wd_ub)))[0][0]
    
    # create average price matrix (average price for each flow case)
    price_mat = np.zeros((len(wd_array_price),len(ws_array_price)))
    for i_wd in np.arange(len(wd_array_price)):
        for i_ws in np.arange(len(ws_array_price)):
            fil_wd = wd_ind_timeseries==i_wd
            fil_ws = ws_ind_timeseries==i_ws
            if np.sum(fil_wd&fil_ws)>0:
                price_mat[i_wd,i_ws] = np.mean(price_timeseries[fil_wd&fil_ws])
    
    
    # SAMPLE PRICE FOR EACH WIND DIRECTION -------------------------------------------------------------
    
    # define wind direction bin
    wd_bin_size_price_wd = 15
    wd_array_price_wd = np.arange(0,360,wd_bin_size_price_wd)
    
    # assign to each wd the correspondent price
    wd_ind_timeseries = -np.ones(len(wd_timeseries))
    wd_lb = (wd_array_price_wd-wd_bin_size_price_wd/2)%360
    wd_ub = (wd_array_price_wd+wd_bin_size_price_wd/2)%360
    
    for i in np.arange(len(ws_timeseries)):    
        if (wd_timeseries[i]>=wd_lb[0])|(wd_timeseries[i]<wd_ub[0]):    # check if it is the first wd bin (values around 0deg)
            wd_ind_timeseries[i] = 0
        else:
            wd_ind_timeseries[i]=np.where(((wd_timeseries[i]>=wd_lb)&(wd_timeseries[i]<wd_ub)))[0][0]
    
    # create average price matrix (average price for each flow case)
    price_array_wd = np.zeros((len(wd_array_price_wd)))
    for i_wd in np.arange(len(wd_array_price_wd)):
            fil_wd = wd_ind_timeseries==i_wd
            if np.sum(fil_wd)>0:
                price_array_wd[i_wd] = np.mean(price_timeseries[fil_wd])
    
    
    # SAMPLE PRICE FOR EACH WIND SPEED -------------------------------------------------------------------
    
    # define wind direction bin
    ws_bin_size_price_ws = 1
    ws_array_price_ws = np.arange(4,26,ws_bin_size_price_ws)
    
    # assign to each wd the correspondent price
    ws_ind_timeseries = -np.ones(len(ws_timeseries))
    ws_lb = ws_array_price_ws-ws_bin_size_price_ws/2
    ws_ub = ws_array_price_ws+ws_bin_size_price_ws/2
    
    for i in np.arange(len(ws_timeseries)):    
        if (ws_timeseries[i]>=np.min(ws_array_price_ws))&(ws_timeseries[i]<=np.max(ws_array_price_ws)):   # exclude the case above max ws
            ws_ind_timeseries[i]=np.where(((ws_timeseries[i]>=ws_lb)&(ws_timeseries[i]<ws_ub)))[0][0]
    
    # create average price matrix (average price for each flow case)
    price_array_ws = np.zeros((len(ws_array_price_ws)))
    for i_ws in np.arange(len(ws_array_price_ws)):
            fil_ws = ws_ind_timeseries==i_ws
            if np.sum(fil_ws)>0:
                price_array_ws[i_ws] = np.mean(price_timeseries[fil_ws])
                
    return price_mat,wd_array_price,ws_array_price,price_array_wd,wd_array_price_wd,price_array_ws,ws_array_price_ws

