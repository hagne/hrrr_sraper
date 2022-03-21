# -*- coding: utf-8 -*-

# from atmPy.data_archives.NOAA_ESRL_GMD_GRAD.surfrad import surfrad
from functools import partial
import ftplib
import pygrib
import xarray as xr
import numpy as np
import pathlib as pl
#import shutil
import pandas as pd
#import psutil

import multiprocessing as mp
import time

# import sys

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

# from hrrr_scraper import extra

# tmp = pkg_resources.path('hrrr_scraper.extra', 'hrrr_2d_grb_info_matched.xlsx')
# tmp = pd.read_excel(tmp)
# tmp = next(tmp)
# print(f'tmp: {tmp}')
# p2script = pl.Path(__file__).resolve()
# p2base = p2script.parent.parent.as_posix()

# parameters: https://www.nco.ncep.noaa.gov/pmb/products/hrrr/hrrr.t00z.wrfnatf00.grib2.shtml
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
            {'parameterName': '82', 'typeOfLevel': 'hybrid', 'my_name': 'ice_water_mixing_ratio_vp'},
            
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
            # {'parameterName': 'Geopotential height', 'typeOfLevel':'isothermal', 'level': 253, 'my_name': 'planetary_boundary_layer_height'},  
            {'parameterName': 'Planetary boundary layer height', 'my_name': 'planetary_boundary_layer_height'},  
            
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
            # {'name': 'GPP coefficient from Biogenic Flux Adjustment System', 'typeOfLevel': 'surface', 'my_name': 'gpp_coefficient_from_biogenic_flux_adjustment_system_surface'},
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
            {'shortName' : 'refc', 'my_name': 'composite_reflectivity', 'typeOfLevel' : 'atmosphere'},
          
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

class ProjectorProject(object):
    def __init__(self, sites, 
                 path2raw = '/mnt/telg/tmp/hrrr_tmp/',
                 path2projected_individual = '/mnt/telg/tmp/hrrr_tmp_inter/',
                 path2projected_final = '/mnt/telg/projects/smoke_at_gml_sites/data/wrfnat/',
                 ftp_server = 'ftp.ncep.noaa.gov',
                 ftp_path2files = '/pub/data/nccf/com/hrrr/prod',
                 max_forcast_interval= 18):
        """
        https://nomads.ncep.noaa.gov/

        Parameters
        ----------
        sites : TYPE
            DESCRIPTION.
        path2raw : TYPE, optional
            DESCRIPTION. The default is '/mnt/telg/tmp/hrrr_tmp/'.
        path2projected_individual : TYPE, optional
            DESCRIPTION. The default is '/mnt/telg/tmp/hrrr_tmp_inter/'.
        path2projected_final : TYPE, optional
            DESCRIPTION. The default is '/mnt/telg/projects/smoke_at_gml_sites/data/wrfnat/'.
        ftp_server : TYPE, optional
            DESCRIPTION. The default is 'ftp.ncep.noaa.gov'.
        ftp_path2files : TYPE, optional
            DESCRIPTION. The default is '/pub/data/nccf/com/hrrr/prod'.
        max_forcast_interval : TYPE, optional
            DESCRIPTION. The default is 18.

        Returns
        -------
        None.

        """
        # self.list_of_files_onftp = None
        # self.list_of_files_onftp = None
        self.max_forcast_interval= max_forcast_interval
        
        self.sites = sites
        self.local_file_source = False
        self.path2data_tmp = pl.Path(path2raw)
        self.path2data = pl.Path(path2projected_individual)
        self.path2concatfiles = pl.Path(path2projected_final)
        self.ftp_server = ftp_server
        self.ftp_login = ''
        self.ftp_password = ''
        self.ftp_path2files = ftp_path2files
        
        self._ftp = None
        self._workplan = None
        self.verbose = False
        self.remove_from_workplan_when_output_exists = True
        
        
    @property
    def ftp(self):
        if isinstance(self._ftp, type(None)):
            ftp = ftplib.FTP(self.ftp_server) 
            ftp.login(self.ftp_login, self.ftp_password) 
            # out['ftp'] = ftp
            ### navigate on ftp
            ftp.cwd(self.ftp_path2files)
            # if verbose:
            #     print(bla)
            self._ftp = ftp
        return self._ftp
        
    @property
    def workplan(self):
        if isinstance(self._workplan, type(None)):
            def fn2datetime(row):
            #     try:
                day = pd.to_datetime(row.files_on_ftp.parent.parent.name.split('.')[-1])
                hour = pd.to_timedelta(int(row.files_on_ftp.name.split('.')[1][1:-1]), 'hour')
                out = day + hour
            #     except ValueError:
            #         out = np.nan
                return out
            #get all files from relevant folders
            files_all = []
            daysonftp = self.ftp.nlst()
            if self.verbose:
                print(f'ftp: available days: {daysonftp}')
            for day in daysonftp:
                if self.verbose:
                    print(f'ftp: enter day: {day}')
                    
                self.ftp.cwd(self.ftp_path2files) # not really needed in first loop, but in second!
            #     break
                # enter relevant folders
                self.ftp.cwd(day)
                if self.verbose:
                    print(f'ftp: whats here: {self.ftp.nlst()}')
                    print('ftp: entering conus')
                self.ftp.cwd('conus') # there is also alaska, which might be of interest in the future

                #list files
                files = self.ftp.nlst()
                if self.verbose:
                    # print(f'ftp: whats here: {files}')
                    print(f'ftp: no of files in here {len(files)}')
                # only get the grib files
                files = [f for f in files if f.split('.')[-1] == 'grib2']
                if self.verbose:
                    # print(f'ftp: whats here: {files}')
                    print(f'ftp: no of files ending on grib {len(files)}')
                # wrfnat has the 3d data, there are other files of smaller sized which are merely a subset of values, e.g. surface values
                files =[f for f in files if 'wrfnat' in f.split('.')[-2]]
                if self.verbose:
                    # print(f'ftp: whats here: {files}')
                    print(f"ftp: no of files with: if 'wrfnat' in f.split('.')[-2]: {len(files)}")
                # add full path to list of files
                fld_current = pl.Path(self.ftp.pwd())
                files_all += [fld_current.joinpath(f) for f in files]
            
            # make the workplan from file list
            list_of_files_onftp = files_all
            path2data = self.path2data
            path2data_tmp = self.path2data_tmp 
            max_forcast_interval = self.max_forcast_interval
            
            workplan = pd.DataFrame(list_of_files_onftp, columns = ['files_on_ftp'])
            
            
            workplan['cycle_datetime'] = workplan.apply(fn2datetime, axis=1)
            workplan.dropna(inplace=True)
            
            workplan['forcast_interval'] = workplan.apply(lambda row: int(row.files_on_ftp.name.split('.')[-2][-2:]), axis=1)
            
            ### select forcast intervals
            workplan = workplan[workplan.forcast_interval <= max_forcast_interval]
            
            ### generate temporary output path
            workplan['path2tempfile'] = workplan.apply(lambda row: path2data_tmp.joinpath(row.files_on_ftp.name), axis=1)
            
            ### generate output path
            workplan['path2file']  = workplan.apply(lambda row: path2data.joinpath(f'{row.cycle_datetime.year:04d}{row.cycle_datetime.month:02d}{row.cycle_datetime.day:02d}_{row.cycle_datetime.hour:02d}_fi{row.forcast_interval:02d}' + '.nc'), axis=1)
            
            ### only files that don't exist yet
            # if self.verbose:
            #     # print(f'ftp: whats here: {files}')
            #     print(f'ftp: no of files that do not exist yet {len(files)}')
            workplan['output_file_exits'] = workplan.apply(lambda row: row.path2file.is_file(), axis=1)
            
            if self.remove_from_workplan_when_output_exists:
                workplan = workplan[~workplan.output_file_exits].copy()
            
            workplan.sort_values(['cycle_datetime', 'forcast_interval'], inplace= True)
            
            self._workplan = workplan
            self.ftp.close()
            self._ftp = None
        return self._workplan
    
    @workplan.setter
    def workplan(self, value):
        self._workplan = value
    
    
    def remove_artefacts(self):
        """
        removes old grib files. This is a good idea in case a earlier instance did a partial download which will cause errors.
        """
        for file in self.path2data_tmp.glob('*grib2'):
            file.unlink()
            
        return 
    
    
    def process(self, no_of_cpu = 3, test = False):
        # def process_workplan_row(row):
        #     wpe = WorkplanEntry(self, row)
        #     wpe.process()
        #     return 
        self.remove_artefacts()
        out = {}
        if self.workplan.shape[0] == 0:
            print('noting to process')
            return
        elif test == 2:
            wpt = self.workplan.iloc[:1]
        elif test == 3:
            wpt = self.workplan.iloc[:no_of_cpu]
        else:
            wpt = self.workplan
        
        if no_of_cpu == 1:
            for idx, row in wpt.iterrows():
                wpe = WorkplanEntry(row, project = self)
                wpe.process()
        else:
            pool = mp.Pool(processes=no_of_cpu)
            # print(wpt.shape)
            idx, rows = zip(*list(wpt.iterrows()))
            # out['pool_return'] = pool.map(partial(process_workplan_row, **{'ftp_settings': ftp_settings, 'sites': sites}), rows)
            
            # pool_return = 
            pool.map(partial(WorkplanEntry, **{'project': self, 'autorun': True}), rows)
            # out['pool_return'] = pool_return
            
            pool.close() # no more tasks
            pool.join()
        if test == False:
            concat = Concatonator(path2scraped_files = self.path2data,
                         path2concat_files = self.path2concatfiles,)
            concat.save()
            out['concat'] = concat
        return out



class WorkplanEntry(object):
    def __init__(self, workplanrow, project = None, autorun = False):
        self.project = project
        self.row = workplanrow
        self.verbose = True
        
        self._hrrr_inst = None
        self._projection = None
        if autorun:
            self.process()
        
    
    def download(self):
        if not self.row.path2tempfile.is_file():
            ftp = ftplib.FTP(self.project.ftp_server) 
            ftp.login(self.project.ftp_login, self.project.ftp_password) 
    #         out['ftp'] = ftp
            ### navigate on ftp
            # ftp.cwd(ftp_path2files)
            ftp.retrbinary(f'RETR {self.row.files_on_ftp}', open(self.row.path2tempfile, 'wb').write)
            ftp.close()
        else:
            if self.verbose:
                print('file exist ... skip!', end = '...')
                
    @property    
    def hrrr_inst(self):
        if isinstance(self._hrrr_inst, type(None)):
            self._hrrr_inst = open_grib_file(self.row.path2tempfile)
        return self._hrrr_inst
    
    @property
    def projection(self):
        if isinstance(self._projection, type(None)):
            self._projection = self.hrrr_inst.project2sites(self.project.sites)
        return self._projection
    
    def save_projection(self):
        self.projection.save(self.row.path2file, cycle_datetime = self.row.cycle_datetime, forcast_hour = self.row.forcast_interval)
    
    def process(self, remove_grib_file = True):
        print('dl', end = '')
        self.download()
        print('.', end = '')
        print('pr', end = '')
        self.projection
        self._hrrr_inst = None # with this I am hoping to save up some memory a little bit earlier then it would otherwise
        print('.', end = '')
        print('s', end = '')
        self.save_projection()
        print('.', end = '')
        self._projection = None # To prevent amemory pileup
        # hwn = open_grib_file(self.row.path2tempfile)
        # projection = hwn.project2sites(self.project.sites)
        # projection.save(self.row.path2file, self.row.cycle_datetime, self.row.forcast_interval)
        # ds = ds.expand_dims({"forecast_hour": [self.row.forcast_interval], "datetime": [row.cycle_datetime]})
        if remove_grib_file:
            print('ul', end = '')
            self.row.path2tempfile.unlink()
            print('.', end = '')
        print('..', end = '')
        return 1#hwn 

class HrrrWrfNat(object):
    def __init__(self, data):
        self.ds = data
    
    def project2sites(self, sites, timeit = False, vp = True):
        """
        Project the model output to these sites

        Parameters
        ----------
        sites : dict, list, atmPy site and network instances
            Site information as dictionary or list of dictionaries in following format:
            {'name': 'Table Mountain', 'abb': 'TBL', 'lat': 40.12498, 'lon': -105.2368}.
        timeit : TYPE, optional
            DESCRIPTION. The default is False.
        vp : TYPE, optional
            DESCRIPTION. The default is True.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if timeit:
            times = [time.time(),]
         
        def dict2site(sitedict):
            site = type('station_manual', (),{})
        
            site.name = sitedict['name']
            site.abb = sitedict['abb']
            site.lat = sitedict['lat']
            site.lon = sitedict['lon']
            return site    
        
        if type(sites).__name__ == 'Network':
            sites = sites.stations._stations_list
            sitetype = 'network'
        elif type(sites).__name__ == 'Station':
            sites = [sites]
            sitetype = 'station'
        elif isinstance(sites, dict):
            sites = [dict2site(sites),]
            sitetype = 'manual'
        elif isinstance(sites, list):
            assert(np.all([isinstance(sd,dict) for sd in sites])), 'all sites need to be dicts.'
            sites = [dict2site(sd) for sd in sites]
            sitetype = 'manual'
        else:
            raise ValueError('somthing wrong with sites')
        
        sitematchtablel = match_hrrr2sites(self.ds, sites)
        ds = select_site_locations(self.ds, sitematchtablel)
        if timeit:
            times.append(time.time())
            dt = times[-1] - times[-2]
            print(f'select site locations time: {dt:.2f}')
        if vp:
            ds = get_univied_above_ground_level_altitude(ds)
        if timeit:
            times.append(time.time())
            dt = times[-1] - times[-2]
            print(f'generate altitude time: {dt:.2f}')
        
        #### add site names
        if sitetype == 'network':
            ds.site.attrs['names'] = [f'{st.abb}: {st.name} ({st.state})' for st in sites if st.abb in ds.site.data]
        elif sitetype == 'manual':
            ds.site.attrs['names'] = [f'{st.abb}: {st.name}' for st in sites if st.abb in ds.site.data]
        
        # ds = ds.expand_dims({"forecast_hour": [self.row.forcast_interval], "datetime": [row.cycle_datetime]})
        return Projection(ds)

class Projection(object):
    def __init__(self, data):
        self.ds  = data
        
    def save(self, fname, nameformat = None, cycle_datetime = None, forcast_hour = None, applyencoding = None, verbose = True):
        """
        Save to netcdf

        Parameters
        ----------
        fname : TYPE
            DESCRIPTION.
        cycle_datetime : TYPE, optional
            DESCRIPTION. The default is None.
        forcast_hour : TYPE, optional
            DESCRIPTION. The default is None.
        applyencoding : TYPE, optional
            There is a problem saving with newer verions of libnetcdf. If your 
            version is 4.8.x you might not be able to reopen the saved file 
            unless applyencoding is disabled.None will test the version and 
            decise from there.
            The default is True.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        ds = self.ds
        fname = pl.Path(fname)
        if  isinstance(cycle_datetime, type(None)):
            # ds = ds.expand_dims({"forecast_hour": [self.row.forcast_interval], "datetime": [row.cycle_datetime]})
            cycle_datetime = pd.to_datetime(ds.attrs['cycle_datetime'])
        if  isinstance(forcast_hour, type(None)):
            forcast_hour = int(ds.attrs['forecast_time'])
        
        if fname.is_dir():
            assert(isinstance(nameformat, str)), f'If fname is a directory the nameformat kwarg needs to be set. It is {nameformat} and of type {type(nameformat)}'
        
        
        ds = ds.expand_dims({"datetime": [cycle_datetime]})
        ds = ds.expand_dims({"forecast_hour": [forcast_hour]})  
        
        
        encoding = {k:{"dtype": "float32", "zlib": True,  "complevel": 9,} for k in ds.variables}
        encoding['argmin_x']['dtype'] = 'int16'
        encoding['argmin_x']['_FillValue'] = -9999
        encoding['argmin_y']['dtype'] = 'int16'
        encoding['argmin_y']['_FillValue'] = -9999
        encoding['site']['dtype'] = 'object'
        encoding['forecast_hour'] = {"dtype": "int8", "zlib": True,  "complevel": 9,}
    
        fout = fname #self.row.path2file
        
        #### workaround for problem with saving with newer libnetcdf version
        
        libversion = [int(i) for i in xr.backends.netCDF4_.netCDF4.getlibversion()[:3].split('.')]
        assert(libversion[0] == 4)
        
        if (libversion[1] <= 7) or (applyencoding == True):
            encoding =  encoding
            if verbose:
                print('Encoding applied in to_netcdf')
        elif (libversion[1] > 7) or (applyencoding == False):
            encoding = None
            if verbose:
                print('No encoding applied in to_netcdf')
        else:
            assert(False),'not possible'
        
        
        if fname.is_dir():
            fout = fout.joinpath(nameformat.format(cycle_datetime = cycle_datetime, forcast_hour = forcast_hour))
        
        if verbose:
            print(f'saving to {fout}')
            
        ds.to_netcdf(fout, encoding=encoding, )

        return 1#ds_out
    
class Concatonator(object):
    def __init__(self, path2scraped_files = '/mnt/telg/tmp/hrrr_tmp_inter/',
                       path2concat_files = '/mnt/telg/projects/smoke_at_gml_sites/data/wrfnat/',
                       test = False):
        self.path2scraped_files = pl.Path(path2scraped_files)
        self.path2concat_files = pl.Path(path2concat_files)
        self.path2concat_files.mkdir(exist_ok=True)
        
        self._workplan = None
        self._concatenated = None
        
    @property
    def workplan(self):
        if isinstance(self._workplan, type(None)):
            ## make a workplan
            workplan = pd.DataFrame(self.path2scraped_files.glob('*.nc'), columns=['path2scraped_files'])
            workplan.shape

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
            workplan['path2concat_files'] = workplan.apply(lambda row: self.path2concat_files.joinpath(f'smoke_at_gml_{row.date.year:04d}{row.date.month:02d}{row.date.day:02d}.nc'), axis = 1)

            # remove if output path exists
            workplan['p2rf_exists'] = workplan.apply(lambda row: row.path2concat_files.is_file(), axis = 1)
            workplan = workplan[ ~ workplan.p2rf_exists].copy()

            workplan.sort_values(['date', 'frcst_cycle', 'frcst_hour'], inplace=True)
            
            self._workplan = workplan
            
        return self._workplan
    
    @property
    def concatenated(self):
        if isinstance(self._concatenated, type(None)):
            concat = []
            for date,date_group in self.workplan.groupby('date'):
#                 print(date)
                fc_list = []
                for frcst_cycle, fc_group in date_group.groupby('frcst_cycle'):
#                     print(frcst_cycle)
                    fc_list.append(xr.open_mfdataset(fc_group.path2scraped_files))#, concat_dim='forecast_hour')) # there must have been a change in xarray, that it did not want the concat_dim anymore?!?

                try:
                    ds = xr.concat(fc_list, dim = 'datetime')
                except ValueError as err:
                    errmsg = err.args[0]
                    err.args = (f'Problem encontered while processing date {date}: {errmsg}',)
                    raise

                fn_out = date_group.path2concat_files.unique()
                assert(len(fn_out) == 1)
                concat.append({'dataset': ds, 'fname': fn_out[0]})
            self._concatenated = concat
        return self._concatenated
#                 ds.to_netcdf(fn_out[0])
        
    def save(self):
        for daydict in self.concatenated:
            daydict['dataset'].to_netcdf(daydict['fname'])
        
def open_grib_file(fname, external_params = False, grab_basics = False):
    if isinstance(fname, pl.Path):
        fname = fname.as_posix()
        
    with pygrib.open(fname) as grbs:
        if grab_basics:
            basics = {}
            grb = grbs[1]
            basics['forecastTime'] = grb.forecastTime
            basics['cycledatetime'] = pd.to_datetime(f'{grb.year}-{grb.month:02d}-{grb.day:02d} {grb.hour:02d}:{grb.minute:02d}:{grb.second:02d}')
            out = basics
        else:
            # grbs = pygrib.open(fname)
            ds = read_selected_fields(grbs, external_params = external_params)#, vp = vp, raise_error_when_varible_missing = raise_error_when_varible_missing) 
            #### cycle and forcast hour
            grb = grbs[1]
            ds.attrs['forecast_time'] = f'{grb.forecastTime:02d}'
            ds.attrs['cycle_datetime'] = f'{grb.year}-{grb.month:02d}-{grb.day:02d} {grb.hour:02d}:{grb.minute:02d}:{grb.second:02d}'
        
            # grbs.close()
            out = HrrrWrfNat(ds)
    return out
    
    

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


def read_selected_fields(grbs, vp = True, external_params = False,  raise_error_when_varible_missing = True):
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
    # lat, lon = grb.latlons()
    lat, lon = [i.astype(np.float32) for i in  grb.latlons()]

    ds = xr.Dataset()

#### TODO: extend to other file formats
#### TODO: extend to 3d
    if external_params:
    #### 3d parameters
        #read variable match table
        
        if isinstance(external_params, dict):
            file_type = external_params['file_type']
            variables = external_params['variables']
        
        else:
            file_type = external_params
            variables = 'all'
        
        if file_type == 'HRRRv4_2d':
            # p2varmatch = './extra/hrrr_2d_grb_info_matched.xlsx'
            # df = pd.read_excel(p2varmatch)
            df = pd.read_excel(pkg_resources.open_binary('hrrr_scraper.extra', 'hrrr_2d_grb_info_matched.xlsx'))
        else:
            assert(False), f'Currently external_params can only have the file_type "HRRRv4_2d". {file_type} not recognized.'
        ## pick only those with netcdf names
        df = df[~(df['netcdf variable name'].isna())]
        
        # get the ones that are in the selected variables
        if variables != 'all':
            df = df[df['netcdf variable name'].isin(variables)]
        # params = []
        for idx, row in df.iterrows():
            if row.typeOfLevel =='hybrid':
                assert(False), 'not working yet, program dude!'  
                
        #### get the 2d stuff
        param_sel = []
        for idx, row in df.iterrows():
            if row.typeOfLevel =='hybrid':
                continue
            var_name = row.pop('netcdf variable name')
            long_name = row.pop('long name')
            units_comments = row.pop('units in extracted/other comments')
            grib2_variable_name = row.pop('grib2 variable name')
            msg = row.pop('messagenumber')
                        
            try:
                grbsel = grbs.select(**row.to_dict())
            except ValueError as err:
                if err.args[0] == "no matches found":
                    print('2d problem:', end = ' ')
                    print(row)
                raise
            # return grbs, row
                
                
            # else:
            #     try:
            #         grbsel = grbs.select(**part)
            #     except ValueError:
            #         print('2d problem:', end = ' ')
            #         print(part)
            #         continue
    
            assert(len(grbsel) == 1), f'The grib select function retured {len(grbsel)} messages instead of 1.'
            grb = grbsel[0]
            assert(grb.messagenumber == msg), f'Message number is {grb.messagenumber}. Config file suggests {msg}.'
            
            da = xr.DataArray(grb.values.astype(np.float32),coords = {'latitude':(['x','y'], lat),
                                               'longitude':(['x','y'], lon)}, 
                         dims = ['x','y'])
            da.attrs = {k: grb[k] for k in grb.keys() if k in attrs}
            da.attrs['long_name'] = long_name
            da.attrs['comments'] = units_comments
            da.attrs['grib2 variable name'] = grib2_variable_name
            ds[var_name] = da
        
    #### 3d parameters
    ### get all parameters with the hybrid typeOfLevel
    else:
        if vp:
            param_sel = []
            for par in params:
                if 'typeOfLevel' in par.keys():
                    if par['typeOfLevel'] == 'hybrid':
                        param_sel.append(par)
        
            for par in param_sel:
                # print(par)
                part = par.copy()
                part.pop('my_name')
                try:
                    grbsel = grbs.select(**part)
                except ValueError:# as err:
                    # if err.args[0] == "no matches found":
                        # print(err, end = ' ')
                    print('problem', end = ': ')
                    print(part)
                    continue
                        
                    # raise
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
                        print('2d problem:', end = ' ')
                        print(part)
                    raise
                
                
            else:
                try:
                    grbsel = grbs.select(**part)
                except ValueError:
                    print('2d problem:', end = ' ')
                    print(part)
                    continue
    
            assert(len(grbsel) == 1)
            grb = grbsel[0]
            da = xr.DataArray(grb.values.astype(np.float32),coords = {'latitude':(['x','y'], lat),
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
