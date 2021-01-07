#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 20:07:17 2020

@author: hagen
"""
import ftplib
import pygrib
import xarray as xr
import numpy as np
import pathlib as pl
import shutil
import pandas as pd
import psutil

def get_the_level_elevation(grb_info_vp, grbs, oro, fmt = 'agl'):
    # get orography
#     if fmt == 'agl':
#         oro = grbs[int(grb_info[grb_info[1] == 'Orography'][0])].values
    
    givpt = grb_info_vp[grb_info_vp[1] == 'Geopotential Height']
    ds_gph_levels = xr.Dataset()
    for idx, row in givpt.iterrows():
        grb = grbs[int(row[0])]
        values, lat, lon = grb.data()
        if fmt == 'agl':
            values = values - oro
        da = xr.DataArray(values,coords = {'latitude':(['x','y'], lat),
                                      'longitude':(['x','y'], lon)}, 
                         dims = ['x', 'y'])
        ds_gph_levels[row[5]] = da
    
    return ds_gph_levels



def get_2d_data(grb_info,grbs, 
#                 variables = 'all',
                test = False,
                verbose = False):
    # all parameters that have vertical profile info
    grb_info_2d = grb_info[~(grb_info[4] == 'hybrid')]
    grb_info_2d = grb_info_2d[~(grb_info_2d[4] == 'depthBelowLandLayer')]

    # if variables == 'all':
    #     variables_vp = list(grb_info_vp[1].unique())
    #     variables_vp.pop(variables_vp.index('Geopotential Height'))
    # else:
    #     variables_vp = variables

    ds_2d_data = xr.Dataset()

    e = 0
    for idx,row in grb_info_2d.iterrows():
        vname = f'{row[1]}_{row[0]}'
        e = e+1
        if verbose:
            print(f'{e}: {vname}')
        if test:
            if e == 5:
                break

        # get the data of the variable
        grb = grbs[int(row[0])]
        values, lat, lon = grb.data()

        da = xr.DataArray(values,coords = {'latitude':(['x','y'], lat),
                                           'longitude':(['x','y'], lon)}, 
                     dims = ['x', 'y'])

        # concat all verticle levels

        # add to dataset
        ds_2d_data[vname] = da
        

    return ds_2d_data

def get_depth_profiles(grb_info,grbs, variables = 'all',test = False):
    # all parameters that have vertical profile info
    # test = True
#     variables = 'all'
    grb_info_depth = grb_info[grb_info[4] == 'depthBelowLandLayer']

    if variables == 'all':
        variables_depth = list(grb_info_depth[1].unique())
    else:
        variables_depth = variables

    ds_depth_profiles = xr.Dataset()

    e = 0
    for var in variables_depth:
        
        e = e+1
        if test:
            print(f'e:{e}')
            if e == 3:
                break
        givpt = grb_info_depth[grb_info_depth[1] == var]

        da_list = []
        i = 0
        for idx, row in givpt.iterrows():
            i += 1

            if test:
                print(f'i:{i}')
                if i == 5:
                    break

            # get the data of the variable
            grb = grbs[int(row[0])]
            values, lat, lon = grb.data()
            depth = np.ones(lat.shape) * float(row[5].split(' ')[1])

            # add an axis for the height
            lat = lat[np.newaxis]
            lon = lon[np.newaxis]
            values = values[np.newaxis]
            depth = depth[np.newaxis]

            values.shape, lat.shape, lon.shape, depth.shape

            da = xr.DataArray(values,coords = {'latitude':(['z','x','y'], lat),
                                               'longitude':(['z','x','y'], lon),
                                               'depth' :(['z','x','y'], depth)}, 
                         dims = ['z','x', 'y'])
            da_list.append(da)

        # concat all verticle levels
        davpt = xr.concat(da_list, dim = 'z')

        # add to dataset
        vname = f'{row[1]}_dp'
        ds_depth_profiles[vname] = davpt

    return ds_depth_profiles

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
    out['argmin'] = argmin
    out['lat_g'] = lat_g[argmin]
    out['lon_g'] = lon_g[argmin]
    out['dist_min'] = dist[argmin]
    return out



def fn2datetime(row):
    try:
        out = pd.to_datetime(row.files_on_ftp[:7],format = '%y%j%H')
    except ValueError:
        out = np.nan
    return out

def make_workplan(list_of_files_onftp, path2data, path2data_tmp, forcast_intervalls = 'all'):
    workplan = pd.DataFrame(list_of_files_onftp, columns = ['files_on_ftp'])
    workplan['cycle_datetime'] = workplan.apply(fn2datetime, axis=1)
    workplan.dropna(inplace=True)

    workplan['forcast_interval'] = workplan.apply(lambda row: int(row.files_on_ftp[-4:-2]), axis=1)

    ### select forcast intervals
    if forcast_intervalls == 'all':
        pass
    else:
        assert(False), 'not implemented yet ... programming required'

    ### generate temporary output path
    workplan['path2tempfile'] = workplan.apply(lambda row: path2data_tmp.joinpath(row.files_on_ftp), axis=1)
    
    ### generate output path
    workplan['path2file']  = workplan.apply(lambda row: path2data.joinpath(f'{row.cycle_datetime.year:04d}{row.cycle_datetime.month:02d}{row.cycle_datetime.day:02d}_{row.cycle_datetime.hour:02d}_fi{row.forcast_interval:02d}' + '.nc'), axis=1)

    ### only files that don't exist yet
    workplan['output_file_exits'] = workplan.apply(lambda row: row.path2file.is_file(), axis=1)

    workplan = workplan[~workplan.output_file_exits].copy()
    return workplan

def get_vertical_profiles(grb_info,grbs, variables = 'all', height_fmt='agl', matchtosites=None, interp_vertical = None, oro_lev_elev = False, test = False, verbose = False, returns = None):
    """
    returns:
        0: returns the orography
        1: returns the firs varible verticle profile
        """
    ### all parameters that have vertical profile info
    if isinstance(matchtosites, type(None)):
        matchtosites = False
    
    grb_info_vp = grb_info[grb_info[4] == 'hybrid']
    
    oro = grbs[int(grb_info[grb_info[1] == 'Orography'][0])].values
#     if height_fmt == 'agl':
#         oro = grbs[int(grb_info[grb_info[1] == 'Orography'][0])].values
#     else:
#         oro = None
        
    if variables == 'all':
        variables_vp = list(grb_info_vp[1].unique())
        variables_vp.pop(variables_vp.index('Geopotential Height'))
    else:
        variables_vp = variables
    
    if oro_lev_elev:
        ds_gph_levels = oro_lev_elev
    else:
        if verbose:
            print("get elevation levels ", end = '...')
        ds_gph_levels = get_the_level_elevation(grb_info_vp, grbs, oro, fmt = height_fmt)
        if returns == 0:
            return ds_gph_levels
        if verbose:
            print("done")
    
    if matchtosites:
        matched_vp_list = []
    else:
        ds_vertical_profiles = xr.Dataset()

    e = 0
    for var in variables_vp:
        if matchtosites:
            ds_vertical_profiles = xr.Dataset()
            
        if verbose:
            tot = len(variables_vp)
            print(f'{e}/{tot}-{var}', end = '.')
        e = e+1
        if test:
            if e == 3:
                break
        givpt = grb_info_vp[grb_info_vp[1] == var]

        da_list = []
        i = 0
        for idx, row in givpt.iterrows():
            i += 1
            if 0:
                print(f'i:{i}')
                if i == 5:
                    break

            # get the data of the variable
            grb = grbs[int(row[0])]
            values, lat, lon = grb.data()
            gph = ds_gph_levels[row[5]].values.copy()

            # add an axis for the height
            lat = lat[np.newaxis]
            lon = lon[np.newaxis]
            values = values[np.newaxis]
            gph = gph[np.newaxis]


            da = xr.DataArray(values,coords = {'latitude':(['z','x','y'], lat),
                                               'longitude':(['z','x','y'], lon),
                                               height_fmt :(['z','x','y'], gph)}, 
                         dims = ['z','x', 'y'])
            da_list.append(da)

        # concat all verticle levels
        davpt = xr.concat(da_list, dim = 'z')

        # add to dataset
        ds_vertical_profiles[row[1]] = davpt
        
        if matchtosites:
            ds_has = match_hrrr2sites(ds_vertical_profiles,matchtosites,interp_vertical=interp_vertical,verbose=False)
            matched_vp_list.append(ds_has)
        if returns == 1:
            return ds_vertical_profiles
        
    if matchtosites:
        return xr.merge(matched_vp_list)
    else:
        return ds_vertical_profiles

def match_hrrr2sites(hrrr_ds, sites, discard_outsid_grid = 2.2, interp_vertical = None, alt_format = ['agl','height'], verbose = False, test = False):
    """
    discard_outsid_grid: int
        maximum distance to closest gridpoint before considered outside grid and discarded. HRRR has a 3km grid -> minimum possible distance: np.sqrt(18)/2 = 2.12"""
#     alt_format = 'gph'
    # get hrrr data at sites
    if type(sites).__name__ == 'Network':
        sites = sites.stations._stations_list
    elif type(sites).__name__ == 'Station':
        sites = [sites]
        
    res_list = []
    for site in sites:
        rest = {}
        rest['site'] = site
        out = get_closest_gridpoint(hrrr_ds, site)
        dist_min = out['dist_min']
        if verbose:
            print(f'{site.abb}: {dist_min}', end = '')
        if out['dist_min'] >= discard_outsid_grid:
            if verbose:
                print(' (outside)', end = '')
            continue
        if verbose:
            print('', end = ', ')
        hrrr_at_site = hrrr_ds.isel(x=out['argmin'][0], y=out['argmin'][1]).copy()

        ######
        if not isinstance(alt_format, type(None)):
            hrrr_at_site = hrrr_at_site.swap_dims({'z':alt_format[0]})
            hrrr_at_site = hrrr_at_site.rename({alt_format[0]:'height'})
#         return hrrr_at_site, site
        hrrr_at_site = hrrr_at_site.drop(['latitude', 'longitude'])
        
        hat_int = xr.Dataset()
        for var in hrrr_at_site:
            if not isinstance(interp_vertical, type(None)):
                hastmp = hrrr_at_site[var].interp(height = interp_vertical,method="linear")
            else:
                hastmp = hrrr_at_site[var]
            hat_int[var] = hastmp
        hrrr_at_site = hat_int
#         return hrrr_at_site, site
        st = pd.Series([site.lon], index = [site.abb])
        st.index.name = 'sites'
        hrrr_at_site['dlon'] = st
        st = pd.Series([site.lat], index = [site.abb])
        st.index.name = 'sites'
        hrrr_at_site['dlat'] = st
        st = pd.Series([site.name], index = [site.abb])
        st.index.name = 'sites'
        hrrr_at_site['dsite'] = st
        ######
        rest['hrrr_at_site'] = hrrr_at_site
#         return hrrr_at_site, site
        res_list.append(rest)
        if test:
            break
    
    if verbose:
        print('')

    # concat data from all sites
    ds_has = xr.concat([res['hrrr_at_site'] for res in res_list], dim = 'sites')
    if not isinstance(alt_format, type(None)):
        ds_has= ds_has.transpose('height','sites')
        ds_has = ds_has.rename({'height':alt_format[1]})
    return ds_has
    
    
def scrape_hrrr_conus_wrfnat(sites,
                             path2data_tmp = '/mnt/telg/tmp/hrrr_tmp/',
                             path2data = '/mnt/data/data/hrrr_smoke/hrrr_conus_wrfnat/',
                             ftp_server = 'gsdftp.fsl.noaa.gov',
                             ftp_login = "anonymous",
                             ftp_password = "hagen.telg@noaa.gov",
                             ftp_path2files = '/hrrr/conus/wrfnat',
                             interp_vertical = None,
                             error_when_not_enough_mem = True,
                             verbose = False,
                             test = False,
                             returns = None,
                             save = True,
                             vp = True,
                             dp = True,
                             srf = True):
    """
    returns:
        1: initialization. (ftp_connection, workplan)
        2: first files grbs and grb_info
    """

    # log = {}
    out = {}
    ds_complete = None # this prevents an errow if the workplan is empty
    
    path2data_tmp = pl.Path(path2data_tmp)
    path2data = pl.Path(path2data)
    # create if do note exist
    path2data_tmp.mkdir(exist_ok=True)
    path2data.mkdir(exist_ok=True)

    ### test if enough disk space is available
    vm = psutil.virtual_memory()
    if error_when_not_enough_mem:
        assert(vm.free > (32611156000 * 0.45)), 'probably not enough Memory available to run this'
    du_daily = 27.5 * 24 # space needed for final data for single day in MB
    du_daily_tmp = 1e3 # space needed for temp data for single day in mB (size of single file, no of cycles, forcast intervals)
    du = shutil.disk_usage(path2data)
    assert(du.free * 1e-6 > (2 * du_daily)), "not enough space for final data"
    du = shutil.disk_usage(path2data_tmp)
    assert(du.free * 1e-6 > (2 * du_daily_tmp)), "not enough space for temporafy files"

    ### connect to ftp
    ftp = ftplib.FTP(ftp_server) 
    ftp.login(ftp_login, ftp_password) 
    out['ftp'] = ftp
    ### navigate on ftp
    bla = ftp.cwd(ftp_path2files)
    if verbose:
        print(bla)
    
    ### workplan
    workplan = make_workplan(ftp.nlst(), path2data, path2data_tmp)
    out['workplan'] = workplan
    
    if returns ==1:
        return out

    ftp.close() #this is necessary because it might take a while between downloads, which can cuase the connection to close
    
    ### go through workplan
    for idx, row in workplan.iterrows():
        if verbose:
            print(f'{row.files_on_ftp}', end = ':')
            print('downloading', end = '...')
        ### download file
        if not row.path2tempfile.is_file():
            ### connect to ftp
            ftp = ftplib.FTP(ftp_server) 
            ftp.login(ftp_login, ftp_password) 
            out['ftp'] = ftp
            ### navigate on ftp
            ftp.cwd(ftp_path2files)
            ftp.retrbinary(f'RETR {row.files_on_ftp}', open(row.path2tempfile, 'wb').write)
            ftp.close()
        else:
            if verbose:
                print('file exist ... skip!', end = '...')

        ### open grib file and get some info
        if verbose:
            print('open grib file', end = '...')
        grbs = pygrib.open(row.path2tempfile.as_posix())
        out['grbs'] = grbs
        grb_info = pd.DataFrame([grb.__str__().split(':') for grb in grbs])
        out['grb_info'] = grb_info
        if returns == 2:
            return out
        # get all profiles
        if vp:
            if verbose:
                print('get profiles', end = '...')
#             if test:
#                 variables = ['Pressure', 'Fraction of cloud cover','Temperature','Specific humidity', 'Mass density']
#             else:
#                 variables = 'all'
#             ds_vertical_profiles = get_vertical_profiles(grb_info,grbs, variables = variables, test = test, verbose = verbose)
#             grb_info = out['grb_info']
#             grbs = out['grbs']
            ds_has_vp = get_vertical_profiles(grb_info, grbs, 
            #                              oro_lev_elev=ds_gph_levels,
                                        matchtosites = sites,
                                         interp_vertical = interp_vertical,
            #                              returns = 0, 
                                         test = test,
                                         verbose = False)
#             ds_has_vp = out2
            # ds_gph_levels = out
            
        # Get all 2d data
        if srf:            
            if verbose:
                print('get 2d data', end = '...')

            ds_2d_data = get_2d_data(grb_info,grbs, verbose=False, test = test)

        if dp:
            # depth data
            if verbose:
                print('get depth data', end = '...')
            ds_depth_profiles = get_depth_profiles(grb_info, grbs, test = test)

        # match to locations
        if verbose:
            print('match locations', end = '...')
#         sites = [surfrad.network.stations.Bondville, surfrad.network.stations.Table_Mountain]
        ds_has_2d = match_hrrr2sites(ds_2d_data,sites, alt_format=None)

#         sites = [surfrad.network.stations.Bondville, surfrad.network.stations.Table_Mountain]
        ds_has_depth = match_hrrr2sites(ds_depth_profiles,sites, alt_format=['depth']*2)

#         sites = [surfrad.network.stations.Bondville, surfrad.network.stations.Table_Mountain]
#         ds_has_vp = match_hrrr2sites(ds_vertical_profiles,sites, interp_vertical=alt_soll)

        # merge and save
        if verbose:
            print('merge and save', end = '...')
        
        ds_complete = xr.merge([ds_has_2d, ds_has_vp,ds_has_depth])

        # test if we did not loose a variable
        no_var_each = len(ds_has_2d.variables)+len(ds_has_depth.variables)+len(ds_has_vp.variables) - (5+5+4)
        no_var_complete = len(ds_complete.variables) - 6
        assert(no_var_each == no_var_complete), f"The result is not the sum of its componentns;\n number of variables combined: {no_var_complete}\n number of variables each: {no_var_each}\n"

        ### add model run datetime and forcast interval
        ds_complete = ds_complete.expand_dims({"forecast_hour": [row.forcast_interval], "datetime": [row.cycle_datetime]})

        encoding = {k:{"dtype": "float32", "zlib": True,  "complevel": 9,} for k in ds_complete.variables}
        encoding['sites']['dtype'] = 'object'
        encoding['dsite']['dtype'] = 'object'
        encoding['forecast_hour']['dtype'] = 'int8'
        encoding.pop('datetime')
        assert(not row.path2file.is_file())
        ds_complete.to_netcdf(row.path2file, format = 'NETCDF4' , encoding= encoding)
        
        ### clean up
        row.path2tempfile.unlink()
        grbs.close()
        
        
        if verbose:
            print('done')
            print('==========')
#         if test:
        # break
        
    return ds_complete


