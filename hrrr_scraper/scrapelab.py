# -*- coding: utf-8 -*-

# from atmPy.data_archives.NOAA_ESRL_GMD_GRAD.surfrad import surfrad
from functools import partial
import ftplib
import pygrib
import xarray as xr
import numpy as np
import pathlib as pl
import shutil
import pandas as pd
import psutil

import multiprocessing as mp
import time

params = [### verticle profiles
            {'parameterName' : 'Mass density', 'typeOfLevel' : 'hybrid', 'my_name': 'aerosol_mass_density_vp'},
            {'parameterName' : 'Specific humidity', 'my_name': 'specific_humidity_vp', 'typeOfLevel' : 'hybrid'},
            {'parameterName' : '32', 'my_name': 'fraction_of_cloud_cover_vp', 'typeOfLevel' : 'hybrid'},
            {'parameterName' : 'Geopotential height', 'my_name': 'level_height_geo_potential_vp', 'typeOfLevel' : 'hybrid'},
            {'parameterName' : 'Temperature', 'my_name': 'temperature_vp', 'typeOfLevel' : 'hybrid'},
            
            {'parameterName': 'Pressure', 'typeOfLevel': 'hybrid', 'my_name': 'pressure_vp'},
            {'name': 'Fraction of cloud cover', 'typeOfLevel': 'hybrid', 'my_name': 'fraction_of_cloud_cover_vp'},
            {'parameterName': 'Specific humidity', 'typeOfLevel': 'hybrid', 'my_name': 'specific_humidity_vp'},
            {'name': 'U component of wind', 'typeOfLevel': 'hybrid', 'my_name': 'u_component_of_wind_vp'},
            {'name': 'V component of wind', 'typeOfLevel': 'hybrid', 'my_name': 'v_component_of_wind_vp'},
            {'name': 'Vertical velocity', 'typeOfLevel': 'hybrid', 'my_name': 'vertical_velocity_vp'},
            {'name': 'Turbulent kinetic energy', 'typeOfLevel': 'hybrid', 'my_name': 'turbulent_kinetic_energy_vp'},
            
            {'parameterName': 'Cloud mixing ratio', 'typeOfLevel': 'hybrid', 'my_name': 'cloud_mixing_ratio_vp'},
            {'parameterName': 'Ice water mixing ratio', 'typeOfLevel': 'hybrid', 'my_name': 'ice_water_mixing_ratio_vp'},
            
            # trouble getting all the work done ... exclude the next two
            {'parameterName': 'Rain mixing ratio', 'typeOfLevel': 'hybrid', 'my_name': 'rain_mixing_ratio_vp'},
            {'parameterName': 'Snow mixing ratio', 'typeOfLevel': 'hybrid', 'my_name': 'snow_mixing_ratio_vp'},


            ### at surface
            {'parameterName' : 'Mass density', 'typeOfLevel' : 'heightAboveGround', 'my_name': 'aerosol_mass_density_ground_level'},
            {'parameterName' : 'Downward long-wave radiation flux', 'my_name': 'downward_long_wave_radiation_flux_surface'},
            {'parameterName' : 'Upward long-wave radiation flux', 'my_name': 'upward_long_wave_radiation_flux_surface', 'typeOfLevel' : 'surface'},
            {'parameterName' : 'Downward short-wave radiation flux', 'my_name': 'downward_short_wave_radiation_flux_surface'},
            {'name' : 'Upward short-wave radiation flux', 'my_name': 'uward_short_wave_radiation_flux_surface', 'typeOfLevel' : 'surface'},
            {'name' : 'Orography', 'my_name': 'orography'},
            {'parameterName' : 'Temperature', 'my_name': 'temperature_surface', 'typeOfLevel' : 'surface'}, 
            {'parameterName': 'Geopotential height', 'typeOfLevel':'isothermal', 'level': 253, 'my_name': 'planetary_boundary_layer_height'},  
            
            {'parameterName': 'Visibility', 'typeOfLevel': 'surface', 'my_name': 'visibility_surface'},
            {'parameterName': 'Wind speed (gust)', 'typeOfLevel': 'surface', 'my_name': 'wind_speed_gust_surface'},
            {'name': 'Surface pressure', 'typeOfLevel': 'surface', 'my_name': 'surface_pressure_surface'},
            {'parameterName': 'Plant canopy surface water', 'typeOfLevel': 'surface', 'my_name': 'plant_canopy_surface_water_surface'},
            {'parameterName': 'Snow cover', 'typeOfLevel': 'surface', 'my_name': 'snow_cover_surface'},
            {'parameterName': 'Snow depth', 'typeOfLevel': 'surface', 'my_name': 'snow_depth_surface'},
            {'parameterName': 'Precipitation rate', 'typeOfLevel': 'surface', 'my_name': 'precipitation_rate_surface'},
            {'parameterName': 'Sensible heat net flux', 'typeOfLevel': 'surface', 'my_name': 'sensible_heat_net_flux_surface'},
            {'parameterName': 'Latent heat net flux', 'typeOfLevel': 'surface', 'my_name': 'latent_heat_net_flux_surface'},
            {'parameterName': 'Ground heat flux', 'typeOfLevel': 'surface', 'my_name': 'ground_heat_flux_surface'},
            {'name': 'GPP coefficient from Biogenic Flux Adjustment System', 'typeOfLevel': 'surface', 'my_name': 'gpp_coefficient_from_biogenic_flux_adjustment_system_surface'},
            {'parameterName': 'Convective available potential energy', 'typeOfLevel': 'surface', 'my_name': 'convective_available_potential_energy_surface'},
            {'parameterName': 'Convective inhibition', 'typeOfLevel': 'surface', 'my_name': 'convective_inhibition_surface'},
          
            ### column
            {'parameterName' : 'Total column', 'my_name': 'aerosol_integrated_density_column'},
            {'parameterName' : 'Precipitable water', 'my_name': 'precipitable_water_column'},
            {'parameterName' : 'Total cloud cover', 'my_name': 'total_cloud_cover', 'typeOfLevel' : 'atmosphere'},
            {'parameterName' : '102', 'my_name': 'aerosol_optical_depth'},
            # clouds
            {'parameterName' : 'Geopotential height', 'my_name': 'cloud_base_gph', 'typeOfLevel' : 'cloudBase'},
            {'parameterName' : 'Geopotential height', 'my_name': 'cloud_top_gph', 'typeOfLevel' : 'cloudTop'},
            {'parameterName' : '196', 'my_name': 'composite_reflectivity', 'typeOfLevel' : 'atmosphere'},
          
#           {'parameterName' : '', 'my_name': '', 'typeOfLevel' : ''},
         ]

attrs = [
 'parameterCategory',
 'parameterNumber',
 'parameterUnits',
 'parameterName',
 'paramIdECMF',
 'paramId',
 'shortNameECMF',
 'shortName',
 'unitsECMF',
 'units',
 'nameECMF',
 'name',
 'cfNameECMF',
 'cfName',
 'cfVarNameECMF',
 'cfVarName',
 'modelName',
]



def get_closest_gridpoint(grid, site):
    out = {}
    lon_g = grid.longitude.values
    lat_g = grid.latitude.values
    
    if len(lon_g.shape) == 3:
        lon_g = lon_g[0,:,:]
        lat_g = lat_g[0,:,:]
        
    lon_s = site.lon
    lat_s = site.lat

    p = np.pi / 180
    a = 0.5 - np.cos((lat_s-lat_g)*p)/2 + np.cos(lat_g*p) * np.cos(lat_s*p) * (1-np.cos((lon_s-lon_g)*p))/2
    dist = 12742 * np.arcsin(np.sqrt(a))
    
    # get closest
    argmin = np.unravel_index(dist.argmin(), dist.shape)
    out['argmin_x'] = argmin[0]
    out['argmin_y'] = argmin[1]
    out['lat_g'] = lat_g[argmin]
    out['lon_g'] = lon_g[argmin]
    out['lat_s'] = lat_s
    out['lon_s'] = lon_s
    out['dist_min'] = dist[argmin]
    return out

def match_hrrr2sites(hrrr_ds, sites, discard_outsid_grid = 2.2, interp_vertical = None, 
#                      alt_format = ['agl','height'], 
                     verbose = False, test = False,
                     ):
    """
    discard_outsid_grid: int
        maximum distance to closest gridpoint before considered outside grid and discarded. HRRR has a 3km grid -> minimum possible distance: np.sqrt(18)/2 = 2.12"""
    # get hrrr data at sites
    if type(sites).__name__ == 'Network':
        sites = sites.stations._stations_list
    elif type(sites).__name__ == 'Station':
        sites = [sites]
        
    res_list = []
    relevant_sites = []
#     df_out = pd.DataFrame()
    for site in sites:
#         rest = {}
#         rest['site'] = site
        out = get_closest_gridpoint(hrrr_ds, site)
#         return out
        dist_min = out['dist_min']
        if verbose:
            print(f'{site.abb}: {dist_min}', end = '')
        if out['dist_min'] >= discard_outsid_grid:
            if verbose:
                print(' (outside)', end = '')
            continue
        if verbose:
            print('', end = ', ')
        res_list.append(out)
#         df_out[site.abb] = pd.Series(out)
        relevant_sites.append(site.abb)
    # site_abb = [site.abb for site in sites]
    return pd.DataFrame(res_list, 
#                         dtype = {'argmin_x': np.int16}, 
                        index = relevant_sites)
#     return df_out.transpose()

def fn2datetime(row):
    try:
        out = pd.to_datetime(row.files_on_ftp[:7],format = '%y%j%H')
    except ValueError:
        out = np.nan
    return out

def make_workplan(list_of_files_onftp, path2data, path2data_tmp, max_forcast_interval=18):
    workplan = pd.DataFrame(list_of_files_onftp, columns = ['files_on_ftp'])
    workplan['cycle_datetime'] = workplan.apply(fn2datetime, axis=1)
    workplan.dropna(inplace=True)

    workplan['forcast_interval'] = workplan.apply(lambda row: int(row.files_on_ftp[-4:-2]), axis=1)

    ### select forcast intervals
    workplan = workplan[workplan.forcast_interval <= max_forcast_interval]

    ### generate temporary output path
    workplan['path2tempfile'] = workplan.apply(lambda row: path2data_tmp.joinpath(row.files_on_ftp), axis=1)
    
    ### generate output path
    workplan['path2file']  = workplan.apply(lambda row: path2data.joinpath(f'{row.cycle_datetime.year:04d}{row.cycle_datetime.month:02d}{row.cycle_datetime.day:02d}_{row.cycle_datetime.hour:02d}_fi{row.forcast_interval:02d}' + '.nc'), axis=1)

    ### only files that don't exist yet
    workplan['output_file_exits'] = workplan.apply(lambda row: row.path2file.is_file(), axis=1)

    workplan = workplan[~workplan.output_file_exits].copy()
    return workplan

def read_all_fields_deprecated(grbs, sites):
    ### read all fields into a single dataframe and then dataset
    grb_info_list = []
    for e,grb in enumerate(grbs):
    #     print(e, end = ' ')
        grb_info = grb.__str__().split(':')
        grb_info_list.append(grb_info)
    #     break

            # grab the first field
    #         grb = grbs[grb_info[0]]
        values, lat, lon = grb.data()

        da = xr.DataArray(values.astype(np.float32),coords = {'latitude':(['x','y'], lat),
                                           'longitude':(['x','y'], lon)}, 
                     dims = ['x', 'y'])


        ds = xr.Dataset({'grb':da})



        if e==0:
            df_out = match_hrrr2sites(ds, sites)

        sert = df_out.apply(lambda row: ds.isel(x= int(row.argmin_x), y=int(row.argmin_y)).grb.values.copy(), axis = 1).astype(np.float32)
        df_out[grb_info[0]] = sert

        # if e == 10:
        #     break
    #     else:

    #     break

    grb_info_df = pd.DataFrame(grb_info_list)     

    df_out.index.name = 'site'

    ds_out = xr.Dataset(df_out)
    return {'ds_out': ds_out, 'grb_info_df':grb_info_df}

def read_selected_fields(grbs, vp = True, raise_error_when_varible_missing = True):
    def grb_to_grid(grb_obj):
        """Takes a single grb object containing multiple
        levels. Assumes same time, pressure levels. Compiles to a cube"""
        n_levels = len(grb_obj)
        levels = np.array([grb_element['level'] for grb_element in grb_obj])
        # print(f'{levels.dtype}, {levels.max()}')
        indexes = np.argsort(levels)#[::-1] # highest pressure first
        cube = np.zeros([n_levels, grb_obj[0].values.shape[0], grb_obj[1].values.shape[1]], dtype = np.float32)
        for i in range(n_levels):
            cube[i,:,:] = grb_obj[indexes[i]].values
            
        # print(f'{cube.dtype}, {cube.max()}')
        cube_dict = {'data' : cube, 'units' : grb_obj[0]['units'],
                     'levels' : levels[indexes]}
        return cube_dict
    
    
    grb = grbs[1] # just a random parameter ... 76 is an the ground smoke concentration ... 76 did not work with custom files
    lat, lon = grb.latlons()

    ds = xr.Dataset()

    ### get all parameters with the hybrid typeOfLevel
    if vp:
        param_sel = []
        for par in params:
            if 'typeOfLevel' in par.keys():
                if par['typeOfLevel'] == 'hybrid':
                    param_sel.append(par)
    
        for par in param_sel:
            part = par.copy()
            part.pop('my_name')
            try:
                grbsel = grbs.select(**part)
            except ValueError as err:
                if err.args[0] == "no matches found":
                    # print(err, end = ' ')
                    print(part)
                raise
            # grbsel
    
            out = grb_to_grid(grbsel)
    
            levels = out['levels']
    
            da = xr.DataArray(out['data'],coords = {'latitude':(['x','y'], lat),
                                               'longitude':(['x','y'], lon),
                                               'level' :levels}, 
                         dims = ['level','x','y'])
            grb = grbsel[0]
            da.attrs = {k: grb[k] for k in grb.keys() if k in attrs}
            ds[par['my_name']] = da
    
    # print('3d done')
    ### get the 2d stuff

    param_sel = []
    for par in params:
        if 'typeOfLevel' in par.keys():
            if par['typeOfLevel'] == 'hybrid':
                continue
        param_sel.append(par)

    for par in param_sel:
    #     break
        part = par.copy()
        part.pop('my_name')
        
        if raise_error_when_varible_missing:
            
            # grbsel = grbs.select(**part)
            
            try:
                grbsel = grbs.select(**part)
            except ValueError as err:
                if err.args[0] == "no matches found":
                    # print(err, end = ' ')
                    print(part)
                raise
            
            
        else:
            try:
                grbsel = grbs.select(**part)
            except ValueError:
                continue

        assert(len(grbsel) == 1)
        grb = grbsel[0]
        da = xr.DataArray(grb.values,coords = {'latitude':(['x','y'], lat),
                                           'longitude':(['x','y'], lon)}, 
                     dims = ['x','y'])
        da.attrs = {k: grb[k] for k in grb.keys() if k in attrs}
        ds[par['my_name']] = da
    return ds

def select_site_locations(ds, sitematchtablel):
    ds_at_site_list = []
    for idx, row in sitematchtablel.iterrows():
        # ds_orig = ds.copy()
        # ds = ds_orig.copy()

        ds_at_site = ds.isel(x= int(row.argmin_x), y=int(row.argmin_y))

        ds_at_site = ds_at_site.drop(['longitude', 'latitude'])

        for k in row.index:
            ds_at_site[k] = row[k]

        ds_at_site = ds_at_site.expand_dims({"site": [row.name]})
        ds_at_site_list.append(ds_at_site)

    ds_at_sites = xr.concat(ds_at_site_list, dim = 'site')
    return ds_at_sites

def get_univied_above_ground_level_altitude(ds_at_sites):
    alt_soll = np.array([1.00e+01, 2.00e+01, 4.00e+01, 6.00e+01, 8.00e+01, 1.00e+02,
       1.20e+02, 1.40e+02, 1.60e+02, 1.80e+02, 2.00e+02, 2.25e+02,
       2.50e+02, 2.75e+02, 3.00e+02, 3.50e+02, 4.00e+02, 4.50e+02,
       5.00e+02, 6.00e+02, 7.00e+02, 8.00e+02, 9.00e+02, 1.00e+03,
       1.10e+03, 1.20e+03, 1.30e+03, 1.40e+03, 1.50e+03, 1.60e+03,
       1.70e+03, 1.80e+03, 1.90e+03, 2.00e+03, 2.20e+03, 2.40e+03,
       2.60e+03, 2.80e+03, 3.00e+03, 3.20e+03, 3.40e+03, 3.60e+03,
       3.80e+03, 4.00e+03, 4.20e+03, 4.40e+03, 4.60e+03, 4.80e+03,
       5.00e+03, 5.50e+03, 6.00e+03, 6.50e+03, 7.00e+03, 8.00e+03,
       9.00e+03, 1.00e+04, 1.10e+04, 1.20e+04, 1.30e+04, 1.40e+04,
       1.50e+04], dtype=np.float32)
    
    
    ds_at_sites['level_height_above_groundlevel'] = ds_at_sites.level_height_geo_potential_vp - ds_at_sites.orography

    # select variables that have vertical resolution
    vp_vars = [var for var in ds_at_sites.variables if 'level' in ds_at_sites[var].dims][1:]
    ds_vponly = ds_at_sites[vp_vars]

    for e,site in enumerate(ds_at_sites.site.values):
        dsvpoas = ds_vponly.sel(site = site)
        dsvpoas = dsvpoas.swap_dims({'level':'level_height_above_groundlevel'})
        dsvpoas = dsvpoas.drop(labels='level')

        dsvpoas = dsvpoas.interp(level_height_above_groundlevel = alt_soll)
        dsvpoas = dsvpoas.rename_dims({'level_height_above_groundlevel': 'altitude'})
        dsvpoas = dsvpoas.rename({'level_height_above_groundlevel': 'altitude'})
        if e == 0:
            ds_vp = dsvpoas
        else:
            ds_vp = xr.concat([ds_vp, dsvpoas], dim = 'site')

    ds_at_sites = ds_at_sites.drop(vp_vars+['level']).merge(ds_vp)
    return ds_at_sites

def process_workplan_row(row, ftp_settings = None, sites = None, verbose = False):
    timeit = verbose
    ftp_server = ftp_settings['ftp_server']
    ftp_login = ftp_settings['ftp_login']
    ftp_password = ftp_settings['ftp_password']
    ftp_path2files = ftp_settings['ftp_path2files']
    local_file_source = ftp_settings['local_files_source']
    
    raise_error_when_varible_missing = True
    if local_file_source:
        vp = False
        raise_error_when_varible_missing = False
    elif pl.Path(ftp_path2files).name == 'wrftwo':
        vp = False
    else:
        vp = True
    
    # if verbose:
    #     print(f'{row.files_on_ftp}', end = ':')
    #     print('downloading', end = '...')
    ### download file
    if timeit:
        tdls = time.time()
    if not row.path2tempfile.is_file():
        if local_file_source:
            shutil.copy(local_file_source.joinpath(row.files_on_ftp),row.path2tempfile)
        else:
            ### connect to ftp
            ftp = ftplib.FTP(ftp_server) 
            ftp.login(ftp_login, ftp_password) 
    #         out['ftp'] = ftp
            ### navigate on ftp
            ftp.cwd(ftp_path2files)
            ftp.retrbinary(f'RETR {row.files_on_ftp}', open(row.path2tempfile, 'wb').write)
            ftp.close()
    else:
        if verbose:
            print('file exist ... skip!', end = '...')
    if timeit:
        tdle = time.time()
        dt = tdle-tdls
        print(f'download time: {dt:.2f}')
    # grbs.close()

    fname =  row.path2tempfile.as_posix()
    # fname = '/mnt/telg/tmp/hrrr_tmp/2033621001600'

    # open file
    grbs = pygrib.open(fname)

    # some older version
    # out = read_all_fields_deprecated(grbs, sites)
    # ds_out = out['ds_out']
    # grb_info_df = out['grb_info_df']

    ds = read_selected_fields(grbs, vp = vp, raise_error_when_varible_missing = raise_error_when_varible_missing)
    if timeit:
        trsfe = time.time()
        dt = trsfe - tdle
        print(f'read slected field time: {dt:.2f}')
    sitematchtablel = match_hrrr2sites(ds, sites)
    ds = select_site_locations(ds, sitematchtablel)
    if timeit:
        tssle = time.time()
        dt = tssle - trsfe
        print(f'select site locations time: {dt:.2f}')
    if vp:
        ds = get_univied_above_ground_level_altitude(ds)
    if timeit:
        tguaglae = time.time()
        dt = tguaglae-tssle
        print(f'generate altitude time: {dt:.2f}')
    ds = ds.expand_dims({"forecast_hour": [row.forcast_interval], "datetime": [row.cycle_datetime]})

    ### save to file
    encoding = {k:{"dtype": "float32", "zlib": True,  "complevel": 9,} for k in ds.variables}
    encoding['argmin_x']['dtype'] = 'int16'
    encoding['argmin_x']['_FillValue'] = -9999
    encoding['argmin_y']['dtype'] = 'int16'
    encoding['argmin_y']['_FillValue'] = -9999
    encoding['site']['dtype'] = 'object'
    encoding['forecast_hour']['dtype'] = 'int8'
    encoding.pop('datetime')

    # fout = '/mnt/telg/tmp/blablub.nc'
    fout = row.path2file
    ds.to_netcdf(fout, encoding=encoding, )
    # grb_info_df.to_csv(row.path2file.with_suffix('.csv'))
    row.path2tempfile.unlink()
    grbs.close()
    if timeit:
        tsavee = time.time()
        dt = tsavee - tguaglae
        print(f'save time: {dt:.2f}')
    return 1#ds_out

def scrape_hrrr_conus(sites,
                    local_file_source = False,
                    path2data_tmp = '/mnt/telg/tmp/hrrr_tmp/',
                    path2data = '/mnt/telg/tmp/hrrr_tmp_inter/',
                    ftp_server = 'gsdftp.fsl.noaa.gov',
                    ftp_login = "anonymous",
                    ftp_password = "hagen.telg@noaa.gov",
                    ftp_path2files = '/hrrr/conus/wrfnat',
                    no_of_cpu = 5,
                    #interp_vertical = None,
                    error_when_not_enough_mem = True,
                    verbose = False,
                    test = False,
                    # returns = None,
                    # save = True,
                    # vp = True,
                    # dp = True,
                    # srf = True,
                    ):
    """
    

    Parameters
    ----------
    sites : TYPE
        DESCRIPTION.
    local_file_source : bool, optional
        If True a local folder will be treated as if on the ftp server. Data 
        will still be copied to the temporary folder. This is done more 
        elegantly in the satellite scraper. The default is False.
    path2data_tmp : TYPE, optional
        DESCRIPTION. The default is '/mnt/telg/tmp/hrrr_tmp/'.
    path2data : TYPE, optional
        DESCRIPTION. The default is '/mnt/telg/tmp/hrrr_tmp_inter/'.
    ftp_server : TYPE, optional
        DESCRIPTION. The default is 'gsdftp.fsl.noaa.gov'.
    ftp_login : TYPE, optional
        DESCRIPTION. The default is "anonymous".
    ftp_password : TYPE, optional
        DESCRIPTION. The default is "hagen.telg@noaa.gov".
    ftp_path2files : TYPE, optional
        DESCRIPTION. The default is '/hrrr/conus/wrfnat'.
    no_of_cpu : TYPE, optional
        DESCRIPTION. The default is 5.
    #interp_vertical : TYPE, optional
        DESCRIPTION. The default is None.
    error_when_not_enough_mem : TYPE, optional
        DESCRIPTION. The default is True.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.
    test : TYPE, optional
        DESCRIPTION. The default is False.
    # returns : TYPE, optional
        DESCRIPTION. The default is None.
    # save : TYPE, optional
        DESCRIPTION. The default is True.
    # vp : TYPE, optional
        DESCRIPTION. The default is True.
    # dp : TYPE, optional
        DESCRIPTION. The default is True.
    # srf : TYPE, optional
        DESCRIPTION. The default is True.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """

    # log = {}
    out = {}

    path2data_tmp = pl.Path(path2data_tmp)
    path2data = pl.Path(path2data)
    # create if do note exist
    path2data_tmp.mkdir(exist_ok=True)
    path2data.mkdir(exist_ok=True)
    
    if local_file_source:
        local_file_source = pl.Path(local_file_source)
        assert(local_file_source.is_dir())

    ### test if enough disk space is available
    vm = psutil.virtual_memory()
    if error_when_not_enough_mem:
        assert(vm.free > (32611156000 * 0.02 * no_of_cpu)), 'probably not enough Memory available to run this'
    du_daily = 27.5 * 24 # space needed for final data for single day in MB
    du_daily_tmp = 1e3 # space needed for temp data for single day in mB (size of single file, no of cycles, forcast intervals)
    du = shutil.disk_usage(path2data)
    assert(du.free * 1e-6 > (2 * du_daily)), "not enough space for final data"
    du = shutil.disk_usage(path2data_tmp)
    assert(du.free * 1e-6 > (2 * du_daily_tmp)), "not enough space for temporafy files"

    ### connect to ftp
    if not local_file_source:
        ftp = ftplib.FTP(ftp_server) 
        ftp.login(ftp_login, ftp_password) 
        out['ftp'] = ftp
        ### navigate on ftp
        bla = ftp.cwd(ftp_path2files)
        if verbose:
            print(bla)

    ### workplan
    if not local_file_source:
        workplan = make_workplan(ftp.nlst(), path2data, path2data_tmp)
    else:
        flst = [p2f.name for p2f in local_file_source.glob('*') if p2f.suffix == '']
        workplan = make_workplan(flst, path2data, path2data_tmp)

    if not local_file_source:
        ftp.close() #this is necessary because it might take a while between downloads, which can cuase the connection to close

    # workplan = workplan.iloc[40:]

    #limit the size of workplan .... can probably go later on
    
    if test == 2:
        wpt = workplan.iloc[:1]
    elif test == 3:
        wpt = workplan.iloc[:no_of_cpu]
    else:
        wpt = workplan
    
    print(f'no files to process: {wpt.shape[0]}')
    out['workplan'] = wpt
    # def fkt2(row):
    #     print(row.files_on_ftp)
    #     return 2
    # #     return row.files_on_ftp
    
    ftp_settings = {}
    ftp_settings['ftp_server'] = ftp_server
    ftp_settings['ftp_login'] = ftp_login
    ftp_settings['ftp_password'] = ftp_password
    ftp_settings['ftp_path2files'] = ftp_path2files
    ftp_settings['local_files_source'] = local_file_source

    out['ftp_settings'] = ftp_settings
    if test == 1:
        return out

    if wpt.shape[0] == 0:
        return out

    pool = mp.Pool(processes=no_of_cpu)
    idx, rows = zip(*list(wpt.iterrows()))
    out['pool_return'] = pool.map(partial(process_workplan_row, **{'ftp_settings': ftp_settings, 'sites': sites}), rows)
    pool.close()
    return out

def concat2daily_files(path2scraped_files = '/mnt/telg/tmp/hrrr_tmp_inter/',
                       path2concat_files = '/mnt/telg/projects/smoke_at_gml_sites/data/wrfnat/',
                       test = False):
    ########
    out = {}
    # some cleanup
    path2scraped_files = pl.Path(path2scraped_files)
    path2concat_files = pl.Path(path2concat_files)
    path2concat_files.mkdir(exist_ok=True)

    ## make a workplan
    workplan = pd.DataFrame(path2scraped_files.glob('*.nc'), columns=['path2scraped_files'])
    # workplan['hrrr_at_sites'] = None

    # get datetime
    workplan['date'] = workplan.apply(lambda row: pd.to_datetime(row.path2scraped_files.name[:8], format = '%Y%m%d'), axis = 1)

    # get forcast cycle
    workplan['frcst_cycle'] = workplan.apply(lambda row: int(row.path2scraped_files.name[9:11]), axis = 1)

    # get forecast hour
    workplan['frcst_hour'] = workplan.apply(lambda row: int(row.path2scraped_files.name[14:16]), axis = 1)

    #remove last day ... only work on the days before last to get daily files
    workplan.sort_values('date', inplace=True)
    last_day = workplan.date.unique()[-1]
    workplan = workplan[workplan.date != last_day].copy()
    # last_day, workplan

    # output paths
    workplan['path2concat_files'] = workplan.apply(lambda row: path2concat_files.joinpath(f'smoke_at_gml_{row.date.year:04d}{row.date.month:02d}{row.date.day:02d}.nc'), axis = 1)

    # remove if output path exists
    workplan['p2rf_exists'] = workplan.apply(lambda row: row.path2concat_files.is_file(), axis = 1)
    workplan = workplan[ ~ workplan.p2rf_exists].copy()
    out['workplan'] = workplan

    workplan.sort_values(['date', 'frcst_cycle', 'frcst_hour'], inplace=True)

    for date,date_group in workplan.groupby('date'):
        fc_list = []
        for frcst_cycle, fc_group in date_group.groupby('frcst_cycle'):
            fc_list.append(xr.open_mfdataset(fc_group.path2scraped_files, concat_dim='forecast_hour'))
            
        try:
            ds = xr.concat(fc_list, dim = 'datetime')
        except ValueError as err:
            errmsg = err.args[0]
            err.args = (f'Problem encontered while processing date {date}: {errmsg}',)
            raise
            
        fn_out = date_group.path2concat_files.unique()

        assert(len(fn_out) == 1)
        ds.to_netcdf(fn_out[0])
    return out