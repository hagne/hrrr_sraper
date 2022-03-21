#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:22:02 2022

@author: hagen
"""

import configparser
import ast
import importlib.resources as pkg_resources
import pandas as pd

def get_all_variable_names(file_type = 'HRRRv4_2d'):
    if file_type == 'HRRRv4_2d':
        df = pd.read_excel(pkg_resources.open_binary('hrrr_scraper.extra', 'hrrr_2d_grb_info_matched.xlsx'))
        
    df = df[~df['netcdf variable name'].isna()]
    txt = '\n'.join([f"{row['netcdf variable name']: <12} # {row['long name']}" for idx, row in df.iterrows()])
    return txt

def create_config(version = None):
    """
    This function creates a configparser instance whith the relavend field to 
    run a model extraction
    
    To save the return config:
        with open('example.ini', 'w') as configfile:
            config.write(configfile)
            
    Parameters
    ----------
    version : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    config : TYPE
        DESCRIPTION.

    """
    config = configparser.ConfigParser(allow_no_value=True, )
    config.optionxform = str
    ######################################
    sect_fileio = 'file_io'
    config[sect_fileio] = {}
    config[sect_fileio]['input_file_type'] = 'HRRRv4_3d'
    config[sect_fileio]['path2input'] = 'prog/hrrr_sraper/examples/operationalize/ncep_hrrr_3d.2133506000000.grib2'
    config[sect_fileio]['path2outputdir'] = 'prog/hrrr_sraper/examples/operationalize/'
    config[sect_fileio]['outputnameformat'] = 'hrrr_surfrad_{datetime}.nc'
    
    ######################################
    sect_locations = 'locations'
    config[sect_locations] = {}
    config[sect_locations]['# These are the locations for which model values are extracted.'] = None
    config[sect_locations]['# Example:'] = None
    config[sect_locations]["# TBL = 'name': 'Table Mountain (CO)', 'lat': 1, 'lon': 2"] = None
    # config[sect_locations]['# '] = None
    # config[sect_locations]['# '] = None
    # config[sect_locations]['# '] = None
    # config[sect_locations]['# '] = None
    config[sect_locations]['TBL'] = "'name': 'Table Mountain (CO)', 'lat': 1, 'lon': 2"
    
    
    ########################################
    sect_variables = 'variables'
    config[sect_variables] = {}
    config[sect_variables]['# Variables to be extracted. Comment out (add a # to the beginning of the line) variables that are not needed.'] = None
    config[sect_variables]['# For more info on variable see https://drive.google.com/drive/u/0/folders/1dUn_o9Nxoga9lKvGx7i_898_t8siOMJB'] = None
    config[sect_variables]['CH4_hf #Methane from Human fats'] = None
    
    return config

def read_config(path2config, parse = True):
    config = configparser.ConfigParser(allow_no_value=True,)
    config.read(path2config)
    if not parse:
        return config