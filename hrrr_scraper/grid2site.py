# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import pathlib as pl
import pandas as pd


def get_closest_gridpoint(grid, site):
    out = {}
    lon_g = grid.longitude.values
    lat_g = grid.latitude.values

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

def match_hrrr2sites(hrrr_ds, sites, discard_outsid_grid = 2.2, verbose = True):
    """
    discard_outsid_grid: int
        maximum distance to closest gridpoint before considered outside grid and discarded. HRRR has a 3km grid -> minimum possible distance: np.sqrt(18)/2 = 2.12"""
    # get hrrr data at sites
    res_list = []
    for site in sites.stations._stations_list:
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
        hrrr_at_site['site'] = site.abb
        rest['hrrr_at_site'] = hrrr_at_site
        res_list.append(rest)
    
    if verbose:
        print('')

    # concat data from all sites
    ds_has = xr.concat([res['hrrr_at_site'] for res in res_list], dim = 'site')
    return ds_has

def project_grid2site(path2hrrr_files = '/mnt/data/data/hrrr_smoke/subset/',#path to smoke files
                      path2res_files = '/mnt/data/data/hrrr_smoke/smoke_at_gml/', # basically the output folder
                      sites = None):

    # some cleanup
    path2hrrr_files = pl.Path(path2hrrr_files)
    path2res_files = pl.Path(path2res_files)
    path2res_files.mkdir(exist_ok=True)

    ## make a workplan
    workplan = pd.DataFrame(path2hrrr_files.glob('*.nc'), columns=['path2hrrr_files'])
    workplan['hrrr_at_sites'] = None

    # get datetime
    workplan.index = workplan.apply(lambda row: pd.to_datetime(row.path2hrrr_files.name[:7],format = '%y%j%H'), axis = 1)
    workplan.sort_index(inplace=True)

    #remove last day ... only work on the days before last to get daily files
    last_day = np.unique(workplan.apply(lambda row: row.name.date(), axis = 1))[-1]
    workplan = workplan[~(workplan.apply(lambda row: row.name.date(), axis = 1) == last_day)].copy()

    # output paths
    workplan['path2res_file'] = workplan.apply(lambda row: path2res_files.joinpath(f'smoke_at_gml_{row.name.year:04d}{row.name.month:02d}{row.name.day:02d}.nc'), axis = 1)

    # remove if output path exists
    workplan['p2rf_exists'] = workplan.apply(lambda row: row.path2res_file.is_file(), axis = 1)
    workplan = workplan[ ~ workplan.p2rf_exists].copy()

    ## apply match_hrrr2sites on workplan
    for idx, row in workplan.iterrows():    
        hrrr_ds = xr.open_dataset(row.path2hrrr_files)
        has= match_hrrr2sites(hrrr_ds, sites, verbose = False)
        # attach to object so pandas excepts it
        has = type('has',(),{'has':has})
        workplan.loc[idx, 'hrrr_at_sites'] = has

    ## concat hrrr_at_sites into daily datasets and sav
    for p2r,group in workplan.groupby('path2res_file'):
        group.index.name = 'datetime'
        ds_has_ts =  xr.concat([has.has for has in group.hrrr_at_sites], dim = group.index)
        ds_has_ts.to_netcdf(p2r)
    return ds_has_ts
