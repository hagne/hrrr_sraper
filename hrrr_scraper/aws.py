#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:41:20 2022

@author: hagen
"""

# -*- coding: utf-8 -*-
import pathlib as _pl
import pandas as _pd
import xarray as _xr
import s3fs as _s3fs
# import urllib as _urllib
# import html2text as _html2text
import psutil as _psutil
import numpy as _np


import hrrr_scraper.hrrr_lab
# import hrrr_scraper.aws
# import xarray as _xr

def readme():
    url = 'https://github.com/awslabs/open-data-docs/tree/main/docs/noaa/noaa-hrrr'
    print(f'follow link for readme: {url}')


class HrrrScraperAWSDaily(object):
    def __init__(self,
                 path2out = '/nfs/stu3data2/Model_data/HRRR/HRRRv4_conus_projected/',
                 path2temp_raw = '/home/grad/htelg/tmp/hrrr_tmp/',
                 path2temp_projections = '/home/grad/htelg/tmp/hrrr_tmp_inter/',
                 name_pattern =  'smoke_at_gml_{date}.nc',
                 start = '2024-10-14',  
                 end = '2024-10-16',
                 max_forcast_interval=18,
                 reporter = None, 
                 sites = None):
        self.p2fldcc = _pl.Path(path2out)
        self.path2raw= path2temp_raw
        self.path2projected_individual = path2temp_projections
        # self.path2projected_final= path2projected_final
        self.max_forcast_interval = max_forcast_interval
        self.name_pattern = name_pattern
        # end = pd.Timestamp.now().date()
        self.end = _pd.to_datetime(end)
        self.start = _pd.to_datetime(start) #end - pd.to_timedelta(10, 'd')
        self.reporter = reporter
        self.sites = sites

        
        self._wp = None

    @property
    def workplan(self):
        if isinstance(self._wp, type(None)):
            wp = _pd.DataFrame(index = _pd.date_range(self.start, self.end, freq='d'), columns= ['path2file_concat'])

            def row2p2fn(row):
                dt = row.name
                fn = self.name_pattern.format(date = f'{dt.year:04d}{dt.month:02d}{dt.day:02d}')
                p2fn = self.p2fldcc.joinpath(fn)
                return p2fn
            wp['path2file_concat'] = wp.apply(lambda row: row2p2fn(row), axis = 1)
            
            row = wp.iloc[0]
            row
            
            #remove if files exist
            wp = wp[~(wp.apply(lambda row: row.path2file_concat.is_file(), axis = 1))]
            self._wp = wp
        return self._wp


    def process(self, no_of_cpu=3, verbose = False):
        for idx, hsdrow in self.workplan.iterrows():
            if verbose:
                print(f'starting: {hsdrow}')
            # sites = gml_sites
            
            pp = hrrr_scraper.hrrr_lab.ProjectorProject(self.sites,
                                           path2raw=self.path2raw,
                                           path2projected_individual = self.path2projected_individual,
                                           path2projected_final= '/tmp',# this is actually not used... fix in hrrr_lab #self.path2projected_final,
                                           fname_prefix = self.name_pattern.split('{')[0],
                                           # ftp_path2files='/pub/data/nccf/com/hrrr/prod',
                                           # aws_path=False,
                                           # ftp_server='ftp.ncep.noaa.gov',
                                           ftp_server=False,
                                           aws_path='noaa-hrrr-bdp-pds',
                                           start=hsdrow.name,
                                           end=hsdrow.name+ _pd.to_timedelta(0.99999999999,'d'),
                                           max_forcast_interval=self.max_forcast_interval,
                                           reporter = self.reporter
                                        )

            pp.process(no_of_cpu=no_of_cpu, verbose=verbose)

            if pp.workplan.cycle_datetime.max().date() < _pd.Timestamp.utcnow().date():
                try:
                    ds = _xr.open_mfdataset(pp.workplan.path2file)
                    p2fo = hsdrow.path2file_concat
                    ds['datetime'].encoding.update({'units': 'seconds since 2024-10-16'})
                    ds['argmin_x'].encoding.update({'dtype': 'float32'})
                    ds['argmin_y'].encoding.update({'dtype': 'float32'})
                    ds['lat_g'].encoding.update({'dtype': 'float32'})
                    ds['lon_g'].encoding.update({'dtype': 'float32'})
                    ds['lat_s'].encoding.update({'dtype': 'float32'})
                    ds['lon_s'].encoding.update({'dtype': 'float32'})
                    ds['dist_min'].encoding.update({'dtype': 'float32'})

                    ds.to_netcdf(p2fo)

                    #remove projection files
                    for f in pp.workplan.path2file:
                        f.unlink()
                        
                except:
                    if not isinstance(self.reporter, type(None)):
                        self.reporter.errors_increment(pp.workplan.shape[0])
                        self.reporter.wrapup()

                    raise
            else:
                if verbose:
                    print('Concatination skip, as day is not finished.')
                

        return 






class DeprecatedAwsQuery(object):
    def __init__(self,
                 path2folder_local = '/mnt/telg/tmp/aws_tmp/',
                 # satellite = '16',
                 configuration = 'wrfnat',
                 domain = 'conus',
                 start = '2020-08-08 20:00:00', 
                 end = '2020-08-09 18:00:00',
                 cycle_hours = [0,1],
                 forcast_hours = [6,12],
                 process = None,
                 keep_files = None,
                 overwrite = False,
                ):
        """
        This will initialize a search on AWS.

        Parameters
        ----------
        path2folder_local : TYPE, optional
            DESCRIPTION. The default is '/mnt/telg/tmp/aws_tmp/'.
        satellite : TYPE, optional
            DESCRIPTION. The default is '16'.
        product : str, optional
                wrfnat -> 3D native levels
                wrfprs -> 3D isobaric levels
                wrfsfc -> 2D surface
                wrfsubhf -> 2D Sub-hrly
            Note this is the product name described at 
            https://docs.opendata.aws/noaa-goes16/cics-readme.html 
            but without the scan sector. The default is 'ABI-L2-AOD'.
        scan_sector : str, optional
            (C)onus, (F)ull_disk, (M)eso. The default is 'C'.
        start : TYPE, optional
            DESCRIPTION. The default is '2020-08-08 20:00:00'.
        end : TYPE, optional
            DESCRIPTION. The default is '2020-08-09 18:00:00'.
        process: dict,
            This is still in development and might be buggy.
            Example:
                dict(concatenate = 'daily',
                     function = lambda row: some_function(row, *args, **kwargs),
                     prefix = 'ABI_L2_AOD_processed',
                     path2processed = '/path2processed/')
        keep_files: bool, optional
            Default is True unless process is given which changes the default
            False.

        Returns
        -------
        None.

        """
        # self.satellite = satellite
        self.path2folder_aws =  _pl.Path('noaa-hrrr-bdp-pds')
        
        self.domain = domain 
        self.configuration = configuration
        
        self.start = _pd.to_datetime(start)
        self.end =  _pd.to_datetime(end)        
        self.cycle_hours = cycle_hours
        self.forcast_hours = forcast_hours
        self.path2folder_local = _pl.Path(path2folder_local)
        self.overwrite = overwrite
        
        if isinstance(process, dict):
            self._process = True
            # self._process_concatenate = process['concatenate']
            self._process_function = process['function']
            self._process_name_prefix = process['prefix']
            self._process_path2processed = _pl.Path(process['path2processed'])
            # self._process_path2processed_tmp = self._process_path2processed.joinpath('tmp')
            # self._process_path2processed_tmp.mkdir(exist_ok=True)
            self.keep_files = False
            # self.check_if_file_exist = False
        else:
            self._process = False
            
        self.aws = _s3fs.S3FileSystem(anon=True)
        self.aws.clear_instance_cache() # strange things happen if the is not the only query one is doing during a session
        # properties
        self._workplan = None
        
    @property
    def product(self):
        return self._product
    
    @product.setter
    def product(self, value):
        if value[-1] == self.scan_sector:
            value = value[:-1]
        self._product = value
        return
        
    def info_on_current_query(self):
        nooffiles = self.workplan.shape[0]
        if nooffiles == 0:
            info = 'no file found or all files already on disk.'
        else:
            du = self.estimate_disk_usage()
            disk_space_needed = du['disk_space_needed'] * 1e-6
            disk_space_free_after_download = du['disk_space_free_after_download']
            info = (f'no of files: {nooffiles}\n'
                    f'estimated disk usage: {disk_space_needed:0.0f} mb\n'
                    f'remaining disk space after download: {disk_space_free_after_download:0.0f} %\n')
        return info
    
    # def print_readme(self):
    #     url = 'https://docs.opendata.aws/noaa-goes16/cics-readme.html'
    #     html = _urllib.request.urlopen(url).read().decode("utf-8") 
    #     out = _html2text.html2text(html)
    #     print(out)
    
    def estimate_disk_usage(self, sample_size = 10): #mega bites
        step_size = int(self.workplan.shape[0]/sample_size)
        if step_size < 1:
            step_size = 1
        sizes = self.workplan.iloc[::step_size].apply(lambda row: self.aws.disk_usage(row.path2file_aws), axis = 1)
        # sizes = self.workplan.iloc[::int(self.workplan.shape[0]/sample_size)].apply(lambda row: self.aws.disk_usage(row.path2file_aws), axis = 1)
        disk_space_needed = sizes.mean() * self.workplan.shape[0]
        
        # get remaining disk space after download
        du = _psutil.disk_usage(self.path2folder_local)
        disk_space_free_after_download = 100 - (100* (du.used + disk_space_needed)/du.total )
        out = {}
        out['disk_space_needed'] = disk_space_needed
        out['disk_space_free_after_download'] = disk_space_free_after_download
        return out
        
    @property
    def workplan(self):
        if isinstance(self._workplan, type(None)):
            #### make a data frame to all the available files in the time range
            # create a dataframe with all days in the time range
            df = _pd.DataFrame(index = _pd.date_range(self.start, self.end, freq='d'), columns=['path'])
            
            # create the path to the directory of each row above
            df['path'] = df.apply(lambda row: self.path2folder_aws.joinpath(f'hrrr.{row.name.year}{row.name.month:02d}{row.name.day:02d}').joinpath(self.domain).joinpath('*'), axis= 1)
            
            # get the path to each file in all the folders 
            # cyclhours = [f't{ch:02d}z' for ch in self.cycle_hours]
            files_available = []
            for idx,row in df.iterrows():
                fat = self.aws.glob(row.path.as_posix())
                # get grib files only
                # print(f'found {len(fat)} files')
                fat = [f for f in fat if (self.configuration in f) and (f[-6:] == '.grib2')]
                # # select desired cycle hourse ... lets do this later!
                # if self.cycle_hours != 'all':
                #     cyclhours = [f't{ch:02d}z' for ch in self.cycle_hours]
                #     fat = [f for f in fat if f.split('/')[-1].split('.')[1] in cyclhours]
                # break
                # print(f'found {len(fat)} files')
                files_available += fat
            
            #### Make workplan
            workplan = _pd.DataFrame([_pl.Path(f) for f in files_available], columns=['path2file_aws'])
            workplan['date'] = workplan.apply(lambda row: _pd.to_datetime(row.path2file_aws.parent.parent.name.split('.')[-1]), axis = 1)
            workplan['cycle_hour'] = workplan.apply(lambda row: int(row.path2file_aws.name.split('.')[1][1:3]), axis = 1)
            workplan['datetime_cycle'] = workplan.apply(lambda row: row.date + _pd.to_timedelta(row.cycle_hour, 'hour'), axis = 1)
            workplan.index = workplan.datetime_cycle
            workplan['forecast_hour'] = workplan.apply(lambda row: int(row.path2file_aws.name.split('.')[2][-2:]), axis = 1)
            workplan['datetime_forecast'] = workplan.apply(lambda row: row.date + _pd.to_timedelta(row.cycle_hour, 'hour') + _pd.to_timedelta(row.forecast_hour, 'hour'), axis = 1)
            self.tp_wpI = workplan.copy()
            # select forcast hours and cycle_hours if specified
            if self.forcast_hours != 'all':
                workplan = workplan[workplan.forecast_hour.isin(self.forcast_hours)].copy()
                
            if self.cycle_hours != 'all':
                workplan = workplan[workplan.cycle_hour.isin(self.cycle_hours)].copy()
            self.tp_wpII = workplan.copy()
            # generate path to local copy of grib file
            workplan['path2file_local'] = workplan.apply(lambda row: self.path2folder_local.joinpath(f'hrrr.{self.configuration}.{row.date.year}{row.date.month:02d}{row.date.day:02d}.{row.cycle_hour:02d}.{row.forecast_hour}.grib2'), axis = 1)
            
            #### remove if local file exists
            # if not self._process:
            if not self.overwrite:
                workplan = workplan[~workplan.apply(lambda row: row.path2file_local.is_file(), axis = 1)]
            
            # truncate - this only really relevant if start end end have times in addition to the dates
            workplan = workplan.sort_index()
            workplan = workplan.truncate(self.start, self.end)
            
            #### processing additions
            if self._process:
                assert(False), 'not implemented yet!'
                ### add path to processed file names
                workplan["path2file_local_processed"] = workplan.apply(lambda row: self._process_path2processed.joinpath(f'{self._process_name_prefix}_{row.name.year}{row.name.month:02d}{row.name.day:02d}_{row.name.hour:02d}{row.name.minute:02d}{row.name.second:02d}.nc'), axis = 1)
                ### remove if file exists 
                workplan = workplan[~workplan.apply(lambda row: row.path2file_local_processed.is_file(), axis = True)]
                # workplan['path2file_tmp'] = workplan.apply(lambda row: self._process_path2processed_tmp.joinpath(row.name.__str__()), axis = 1)
            
            
            self._workplan = workplan
        return self._workplan       
    
    
    @workplan.setter
    def workplan(self, new_workplan):
        self._workplan = new_workplan
    
    @property
    def product_available_since(self):
        product_folder = self.path2folder_aws.joinpath(f'{self.product}{self.scan_sector}')
        years = self.aws.glob(product_folder.joinpath('*').as_posix())
        years.sort()
        
        is2000 = True
        while is2000:
            yearfolder = years.pop(0)
            firstyear = yearfolder.split('/')[-1]
            # print(firstyear)
            if firstyear != '2000':
                is2000 = False
                
        yearfolder = _pl.Path(yearfolder)
        days = self.aws.glob(yearfolder.joinpath('*').as_posix())
        days.sort()
        firstday = int(days[0].split('/')[-1])
        firstday_ts = _pd.to_datetime(firstyear) + _pd.to_timedelta(firstday, "D")
        return firstday_ts
        
    def download(self, test = False, overwrite = False, alternative_workplan = False,
                 error_if_low_disk_space = True):
        """
        

        Parameters
        ----------
        test : TYPE, optional
            DESCRIPTION. The default is False.
        overwrite : TYPE, optional
            DESCRIPTION. The default is False.
        alternative_workplan : pandas.Dataframe, optional
            This will ignore the instance workplan and use the provided one 
            instead. The default is False.
        error_if_low_disk_space : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        if isinstance(alternative_workplan, _pd.DataFrame):
            workplan = alternative_workplan
        else:
            workplan = self.workplan
        
        if error_if_low_disk_space:
            disk_space_free_after_download = self.estimate_disk_usage()['disk_space_free_after_download']
            assert(disk_space_free_after_download<90), f"This download will bring the disk usage above 90% ({disk_space_free_after_download:0.0f}%). Turn off this error by setting error_if_low_disk_space to False."
        
        for idx, row in workplan.iterrows():
            if not overwrite:
                if row.path2file_local.is_file():
                    continue
                
            out = self.aws.get(row.path2file_aws.as_posix(), row.path2file_local.as_posix())
            if test:
                break
        return out
    
    
    def process(self):
    # deprecated first grouping is required
        # group = self.workplan.groupby('path2file_local_processed')
        # for p2flp, p2flpgrp in group:
        #     break
        ## for each file in group
        
        for dt, row in self.workplan.iterrows():  
            if row.path2file_local_processed.is_file():
                continue
            if not row.path2file_local.is_file():
    #             print('downloading')
                #### download
                # download_output = 
                self.aws.get(row.path2file_aws.as_posix(), row.path2file_local.as_posix())
            #### process
            try:
                self._process_function(row)
            except:
                print(f'error applying function on one file {row.path2file_local.name}. The raw fill will still be removed (unless keep_files is True) to avoid storage issues')
            #### remove raw file
            if not self.keep_files:
                row.path2file_local.unlink()
    
        #### todo: concatenate 
        # if this is actually desired I would think this should be done seperately, not as part of this package
        # try:
        #     ds = _xr.open_mfdataset(p2flpgrp.path2file_tmp)

        #     #### save final product
        #     ds.to_netcdf(p2flp)
        
        #     #### remove all tmp files
        #     if not keep_tmp_files:
        #         for dt, row in p2flpgrp.iterrows():
        #             try:
        #                 row.path2file_tmp.unlink()
        #             except FileNotFoundError:
        #                 pass
        # except:
        #     print('something went wrong with the concatenation. The file will not be removed')
        