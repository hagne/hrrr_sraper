# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import xarray as xr
import pathlib as pl
import shutil
import ftplib
import pygrib


def fn2datetime(row):
    try:
        out = pd.to_datetime(row.files_on_ftp[:7],format = '%y%j%H')
    except ValueError:
        out = np.nan
    return out
                       


def make_workplan(list_of_files_onftp, path2data, path2data_tmp):
    workplan = pd.DataFrame(list_of_files_onftp, columns = ['files_on_ftp'])
    workplan['cycle_datetime'] = workplan.apply(fn2datetime, axis=1)
    workplan.dropna(inplace=True)

    workplan['forcast_interval'] = workplan.apply(lambda row: int(row.files_on_ftp[-4:-2]), axis=1)

    # select forcast intervals
    workplan = workplan[workplan.forcast_interval == 0].copy()

    # generate temporary output path
    workplan['path2tempfile'] = workplan.apply(lambda row: path2data_tmp.joinpath(row.files_on_ftp), axis=1)
    
    # generate output path
    workplan['path2file'] = workplan.apply(lambda row: path2data.joinpath(row.files_on_ftp + '.nc'), axis=1)

    # only files that don't exist yet
    # row = workplan.iloc[0]
    workplan['output_file_exits'] = workplan.apply(lambda row: row.path2file.is_file(), axis=1)

    workplan = workplan[~workplan.output_file_exits].copy()
    return workplan

def grig2dataset(grbs, params_of_interest):
    params_df = params_of_interest.copy()
    # get lat and lon
    grb = grbs[1]
    lat, lon = grb.latlons()

    # make a data array for each parameter and add to params_df
    params_df['data_array'] = params_df.apply(lambda row: xr.DataArray(grbs[int(row.message)].values,
                                             coords = {'latitude':(['x','y'], lat),
                                                       'longitude':(['x','y'], lon)}, 
                                             dims = ['x', 'y']),
                     axis = 1)
    # merge all the arrays into a dataset
    ds = xr.Dataset()
    for idx, row in params_df.iterrows():
        ds[row.par_name] = row.data_array
    return ds

def scrape_hrrr_smoke_from_ftp(ftp_path2files = '/hrrr/conus/wrftwo',
                             ftp_login = "anonymous",
                             ftp_password = "hagen.telg@noaa.gov",
                             path2data_tmp = '/mnt/telg/tmp/hrrr_tmp/',
                             path2data = '/mnt/data/data/hrrr_smoke/subset/',
                             downlaod = True,
                             reformat = True,
                             delete = True,
                             verbose = True,
                             test = False,
                             messages = None,
                             params_of_interest = ('smoke_at_groundlevel:76:Mass density:concentration (instant):lambert:heightAboveGround:level 8 m:fcst time 22 hrs:from 202011021800\n'
                                                  'aerosol_optical_depth:115:102:102 (instant):lambert:unknown:level 0 200:fcst time 22 hrs:from 202011021800\n'
                                                  'smoke_collumn:116:Total column:integrated mass density (instant):lambert:unknown:level 0 200:fcst time 22 hrs:from 202011021800\n'
                                                  'cloud_base:125:Geopotential Height:gpm (instant):lambert:cloudBase:level 0:fcst time 22 hrs:from 202011021800\n'
                                                  'cloud_top:130:Geopotential Height:gpm (instant):lambert:cloudTop:level 0:fcst time 22 hrs:from 202011021800\n'
                                                  'radiative_flux_short_down:132:Downward short-wave radiation flux:W m**-2 (instant):lambert:surface:level 0:fcst time 22 hrs:from 202011021800\n'
                                                  'radiative_flux_long_down:133:Downward long-wave radiation flux:W m**-2 (instant):lambert:surface:level 0:fcst time 22 hrs:from 202011021800\n'
                                                  'radiative_flux_short_up:134:Upward short-wave radiation flux:W m**-2 (instant):lambert:surface:level 0:fcst time 22 hrs:from 202011021800\n'
                                                  'radiative_flux_long_up:135:Upward long-wave radiation flux:W m**-2 (instant):lambert:surface:level 0:fcst time 22 hrs:from 202011021800')
                            ):

    log = {}
    # make dictionary from parameters of interest list
    poil = params_of_interest.split('\n')
    keys = ['par_name', 'message', 'unit_1', 'unit_2', 'unknown_1', 'unknown_2', 'unknown_3','unknown_4','unknown_5',]
    params_of_interest = pd.DataFrame([dict(zip(keys, poi.split(':'))) for poi in poil])

    path2data_tmp = pl.Path(path2data_tmp)
    path2data = pl.Path(path2data)
    # create if do note exist
    path2data_tmp.mkdir(exist_ok=True)
    path2data.mkdir(exist_ok=True)

    # test if enough disk space is available
    du_daily = 27.5 * 24 # space needed for final data for single day in MB
    du_daily_tmp = 150 * 24 # space needed for temp data for single day in mB
    du = shutil.disk_usage(path2data)
    assert(du.free * 1e-6 > (2 * du_daily)), "not enough space for final data"
    du = shutil.disk_usage(path2data_tmp)
    assert(du.free * 1e-6 > (2 * du_daily_tmp)), "not enough space for temporafy files"

    # connect to ftp
    ftp = ftplib.FTP('gsdftp.fsl.noaa.gov') 
    ftp.login(ftp_login, ftp_password) 
    # navigate on ftp
    ftp.cwd(ftp_path2files)

    workplan = make_workplan(ftp.nlst(), path2data, path2data_tmp)
    log['workplan'] = workplan
    if test:
        workplan = workplan.iloc[[0]]
    
    # download and save to temporary file
    if downlaod:
        print('downloading ...')
        if isinstance(messages, list):
                messages.append('downloading')
                messages.append('===========')
        for idx,row in workplan.iterrows():
            fof = row.files_on_ftp
            p2tf = row.path2tempfile
            if p2tf.is_file():
                continue
            if verbose:
                print(f'\t{fof} -> {p2tf}', end = '\t...\t')
            if isinstance(messages, list):
                messages.append(f'\t{fof} -> {p2tf}')
            msg = ftp.retrbinary(f'RETR {fof}', open(p2tf, 'wb').write)
            if verbose:
                print(msg)


        print('done')
    ftp.close()
    
    # open reformat save
    if reformat:
        print('open, re-format, save ...')
        if isinstance(messages, list):
                messages.append('\n')
                messages.append('open, re-format, save')
                messages.append('======================')
        for idx,row in workplan.iterrows():
            p2tf = row.path2tempfile
            p2nc = row.path2file
            if verbose:
                print(f'\t{p2tf} -> {p2nc}', end = ' ... ')
            if isinstance(messages, list):
                messages.append(f'\t{p2tf} -> {p2nc}')
            if p2nc.is_file():
                if verbose:
                    print('file exists ... skip.')
                continue
            grbs = pygrib.open(p2tf.as_posix())
            ds = grig2dataset(grbs,params_of_interest)

            ds.to_netcdf(p2nc, format = 'NETCDF4' , encoding= {k:{"dtype": "float32", "zlib": True,  "complevel": 9,} for k in ds.variables})
            grbs.close()
            if verbose:
                print('done')


        print('done')
    
    if delete:
        for idx,row in workplan.iterrows():
            row.path2tempfile.unlink()
    return log
