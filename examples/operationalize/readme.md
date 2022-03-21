Files
============
development.ipynb : develop the proceedure that goes into the script

Todo
===========
* dave creates a test scenario
* add 3d file capability
    * does the excel file have any netcdf variables?
    * add the HRRRv4_3d option and make it run
    
* adapted code to run Rap
* some of the projected variables are still float64
* there are retrieved parameters that need to be taken care of
* take care of the minimum distance programatically. That refers to, when a point is considered outside of the grid... when dist is larger then grid resoltion -> invalid. Models have differen resolution. is there a key that mention something that allows us to infer the resultion and therefor set the min value?
* 

Questions
=========
* right now it is processing every file in the folder, regardless if the output exists or not. Should we change that to processing only "new" files
* Along the way I do produce netcdfs from the grib. do we want that feature as an option ... convert to netcdf ... no site-projection

Strategy of final script
=========================
inputs:
- config file
- folder with grib files
- output folder

test scenarios:
operational: 
verbose:
test: process only one file
skip save: 
bypass grib file type detection
pybass external params ... us internal ones


The saving issue
=================
I can't open the netcdfs generated in the py38test environment

netcdflibraries in 

py38test:
libnetcdf                 4.8.1           nompi_hb3fd0d9_101    conda-forge
netcdf4                   1.5.8           nompi_py38h2823cc8_101    conda-forge

py38
libnetcdf                 4.7.4           nompi_h56d31a8_107    conda-forge
netcdf4                   1.5.6           nompi_py38hf887595_102    conda-forge
