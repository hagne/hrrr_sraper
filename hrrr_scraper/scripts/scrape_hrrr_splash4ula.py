#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:10:44 2020

@author: hagen
"""

from hrrr_scraper import hrrr_lab
import hrrr_scraper.aws
import traceback
import sys
import pandas as pd
import time
import datetime
import atmPy.data_archives.noaa_gml
from atmPy.general import measurement_site as ms
import psutil
import numpy as np
import json
import pathlib as pl
import smtplib
from email.mime.text import MIMEText
import os
import argparse

import productomator.lab as prolab


import warnings
warnings.filterwarnings("ignore")

pid = os.getpid()
#print(f'pid of current process: {pid}')
version = '2.1.0'




def _main_gml(dryrun = False, first = True, verbose = True,
          no_of_cpu = 1,
          p2f_shortlog = '/home/grad/htelg/script_inits_logs/scrapehrrr_splash3ola.log',
          path2raw = '/home/grad/htelg/tmp/hrrr_tmp_splash4ola/',
          path2projected_individual = '/home/grad/htelg/tmp/hrrr_tmp_inter_splash4ola/',
          path2projected_final = '/nfs/stu3data2/Model_data/HRRR/HRRRvAWS_splash_subset4ola/',
          name_pattern =  'splash_subset4ola_{date}.nc',
          start = '2024-10-14',  
          end = '2024-10-16',
          max_forcast_interval=18,
            # ftp_server = 'ftp.ncep.noaa.gov',
            # ftp_path2files = '/pub/data/nccf/com/hrrr/prod',
            ):
    
    #### execute the program
    #### -------
    # messages = ['run started {}\n========='.format(pd.Timestamp(datetime.datetime.now()))]
    # messages.append(f'scrape_hrrr_conus_nat version {version}')
   
    # errors = []
    # abort = False
    # exit_status = 'fail'
    # loglist = check_log(path2log=p2f_shortlog, verbose = verbose)['loglist']
    # verbose = True
    reporter = prolab.Reporter('scrape_hrrr_gml', 
                                   log_folder='/home/grad/htelg/.processlogs', 
                                   reporting_frequency=(3,'h'))
    # if verbose:
    #     print('Begin processing')
    #     print('=================')
    try:   
        if isinstance(no_of_cpu, type(None)):
            running_processes = [p for p in psutil.process_iter() if p.name() == "scrape_hrrr"]
            running_processes = [p for p in running_processes if p.pid != pid]
            no_of_processes_running = len(running_processes)
            if verbose:
                print(f'number of hrrr_scrape processes running: {no_of_processes_running}')
            mem = psutil.virtual_memory()
            mem_keep = 2 # GiB; assure this much of mememory stay available
            mem_per_process = 6 # GiB, this is the amount of memory that is needed for each process
            mem_avalable = mem.available * 1e-9
            no_of_cpu = int(np.floor((mem_avalable-mem_keep) / mem_per_process))
            # no_of_cpu = round((100 - mem.percent - 15)/20)
            if verbose:
                print(f'numper of cpus: {no_of_cpu}')
            # messages.append(f'numper of cpus: {no_of_cpu}')
        if no_of_cpu < 1:
            # messages.append('aborted since insufficient memory')
            # no_of_files_generated = 0
            # no_of_files_generated_bdl = 0
            abort = True
            if verbose:
                print('aborted,insufficient memoryg')
            exit_status = 'low_memory'
            
        # elif no_of_processes_running > 0:
            
        #     messages.append('aborted since processess still running')
        #     for p in running_processes:
        #         txt = f'status: {p.status()}'
        #         messages.append(txt)
        #         print(txt)
                
        #     mon_proc = monitor_processes(verbose = verbose)
        #     messages.append(mon_proc.mean().__str__())
            
            
        #     no_of_files_generated = 0
        #     no_of_files_generated_bdl = 0
        #     abort = True
        #     exit_status = 'process_running'
        #     if verbose:
        #         print('aborted, process still running')
        else:
            if verbose:
                print('get sites')
            gml_sites = atmPy.data_archives.noaa_gml.get_all_sites()
            # extra from Betsy
            gml_sites.add_station(ms.Station(abbreviation= 'GBN',name = 'Great Basin NP', state='NV', lat=39.005147, lon= -114.215994, alt = 2061))
            gml_sites.add_station(ms.Station(abbreviation= 'AUS',name = 'Austin', state = 'NV', lat= 39.503006, lon= -117.081512, alt =  1915))
            gml_sites.add_station(ms.Station(abbreviation= 'WMS',name = 'White Mtn Summit', state = 'CA', lat=37.634093, lon=-118.255688, alt=4343))
            gml_sites.add_station(ms.Station(abbreviation= 'BCO',name = 'Barcroft Obs', state = 'CA', lat=37.58925, lon=-118.238703, alt=3889))
            
            #### Sail-Splash starting: 20211028
            # test if new stations exist
            new_stations = [dict(abbreviation= 'SS_AP',     name = 'Avery Picnic (PSL’s ASFS sled)',        state = 'CO', lat=38.97262, lon=-106.99717, alt=0),
                            dict(abbreviation= 'SS_AMF2',   name = 'DOE AMF2',                              state = 'CO', lat=38.9564,  lon=-106.98693, alt=0),
                            dict(abbreviation= 'SS_KP',     name = 'Kettle Ponds',                          state = 'CO', lat=38.94177, lon=-106.97308, alt=0),
                            dict(abbreviation= 'SS_KPA',    name = 'Kettle Ponds Annex (PSL’s ASFS sled)',  state = 'CO', lat=38.93939, lon=-106.96985, alt=0),
                            dict(abbreviation= 'SS_BC',     name = 'Brush Creek',                           state = 'CO', lat=38.85915, lon=-106.9209,  alt=0),
                            dict(abbreviation= 'SS_RJ',     name = 'Roaring Judy',                          state = 'CO', lat=38.71688, lon=-106.85304, alt=0),
                            ]
                            
            for ns in new_stations:
                assert(ns['abbreviation'] not in [s.abb for s in gml_sites.stations._stations_list]), f"Station with abbriviation {ns['abbreviation']} already exists in gml_sites"
                gml_sites.add_station(ms.Station(**ns))
                             

            #### process 
            if verbose:
                print('start processing over at hrrr_lab')
            # pp =  hrrr_lab.ProjectorProject(gml_sites, 
            #                                  path2raw = path2raw,
            #                                  path2projected_individual = path2projected_individual,
            #                                  path2projected_final = path2projected_final,
            #                                  ftp_server = ftp_server,
            #                                  ftp_path2files = ftp_path2files,
            #                                  max_forcast_interval= 18,
            #                                  verbose = verbose
            #                                  )
            hsd = hrrr_scraper.aws.HrrrScraperAWSDaily(sites = gml_sites,
                                                       path2out = path2projected_final,#'/nfs/stu3data2/Model_data/HRRR/HRRRv4_conus_projected/',
                                                 path2temp_raw = path2raw, #'/home/grad/htelg/tmp/hrrr_tmp/',
                                                 path2temp_projections = path2projected_individual, #'/home/grad/htelg/tmp/hrrr_tmp_inter/',
                                                 name_pattern =  name_pattern,
                                                 start = '2024-10-14',  
                                                 end = '2024-10-16',
                                                 max_forcast_interval=18,
                                                 reporter = reporter)

            
            if verbose:
                print('generate workplan', end = ' ... ', flush=True)
                hsd.workplan
                print('done')
            if first:
                hsd.workplan = hsd.workplan.iloc[[0], :]
            no_of_files_generated = hsd.workplan.shape[0]
            if verbose:
                print(f'number of files to process: {no_of_files_generated}')
            if dryrun:
                return hsd.workplan
                
            if no_of_files_generated == 0:
                exit_status = 'no_files_on_server'
            else:
                # print(f'no_of_cpu: {no_of_cpu}, {type(no_of_cpu)}')
                hsd.process(no_of_cpu=no_of_cpu, verbose = verbose)
                
                # out_bdl = scrapelab.concat2daily_files()
                # workplan_bdl = out['concat'].workplan
                # no_of_files_generated_bdl = workplan_bdl.shape[0]
                # for idx,row in workplan.iterrow():
            #     messages.append(f'{row.path2tempfile} -> {row.path2file}')
                # exit_status = 'clean'
            reporter.wrapup()
            if verbose:
                print('Done!!')
    except:
        error, error_txt, trace = sys.exc_info()
        tm = ['{}: {}'.format(error.__name__, error_txt.args[0])] + traceback.format_tb(trace)
        txt = '\n'.join(tm)
        print(txt)
        # messages.append(txt)
        # errors.append(txt)
        # no_of_files_generated = 0
        # no_of_files_generated_bdl = 0
    
    if verbose:
        print('++++++++++++++++++++++++++')
    return
    # messages.append('============\nrun finished {}\n\n'.format(pd.Timestamp(datetime.datetime.now())))
    
    ### generate log text
    # message_txt = '\n'.join(messages)
    
    
    #### save log
    ##### shortlog
    
    # if verbose:
    #     print('logging')
    #     print('=========')
    # try:
    #     log = {}
    #     log['exit_status'] = exit_status
    #     log['timestamp'] = pd.Timestamp.now().__str__()
    #     # loglist = []
    #     if exit_status == 'process_running':    
    #         df = mon_proc.mean()
    #         df = df.loc[[idx for idx in df.index if 'cpu' in idx]]
    #         log['cpuavg'] = df.mean()#'{:0.2f}'.format(df.mean())
    #         log['cpumax'] = df.max()#'{:0.2f}'.format(df.max())
    #         loglist.append(log)
            
    #     with open(p2f_shortlog, 'w') as f:
    #         # loglist.to_csv(f)
    #         json.dump(loglist, f)
    #     if verbose:
    #         print('Done')
    # except:
    #     errors.append('logging failed!?!?')
    #     if verbose:
    #         print('writing log failed')
    # if verbose:
    #     print('++++++++++++++')
    
    # if len(errors) !=0:
    #     error_txt = '\n\n======================\nERRORS\n=======================\n'
    #     error_txt += '\n=======================================\n=========================================='.join(errors)
    #     message_txt += error_txt
    # #### send email with results
    # if verbose:
    #     print('Sending email')
    #     print('==============')
    # try:
    #     # import smtplib
    
    #     # # Import the email modules we'll need
    #     # from email.mime.text import MIMEText
    
    #     # Open a plain text file for reading.  For this example, assume that
    #     # the text file contains only ASCII characters.
    #     # with open(textfile) as fp:
    #     #     # Create a text/plain message
    #     msg = MIMEText(message_txt)
    
    #     # me == the sender's email address
    #     # you == the recipient's email address
    #     address  = 'hagen.telg@noaa.gov'
    #     if abort:
    #         passed = 'Aborted'
    #     elif exit_status == 'no_files_on_server':
    #         passed = 'Abnormal (no files on server)'
    #     elif len(errors) == 0:
    #         passed = f'Clean ({no_of_files_generated}/{no_of_files_generated_bdl} files)'
    #     else:
    #         passed = 'Errors ({})'.format(len(errors))
    #     msg['Subject'] = 'scrape_hrrr_smoke run - {} - {}'.format(passed, pd.Timestamp(datetime.datetime.now()))
    #     msg['From'] = address
    #     msg['To'] = address
    
    #     # Send the message via our own SMTP server.
    #     s = smtplib.SMTP('localhost')
    #     s.send_message(msg)
    #     s.quit()
    #     if verbose:
    #         print('Done')
    # except:
    #     if verbose:
    #         print('sending email failed')
    
    # #### new logging 
    # #### TODO: evantuall this should replace the above
    # # retrieval = type('scrape_hrrr', (), {})
    # # retrieval.no_processed_success = no_of_files_generated
    # # retrieval.no_processed_error = len(errors)
    # # retrieval.no_processed_warning = 0
    
    # if verbose:
    #     print('++++++++++++++++')
    
    # if verbose:
    #     print('End of scirpt!')
    
    # return retrieval
    
# if __name__ == '__main__':
def main():
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('-d', '--dry', help='testing: will stop before execution.',  action='store_true')
    parser.add_argument('-f', '--first', help='testing: process only the very first enty.',  action='store_true')
    parser.add_argument('-v', '--verbose', help='verbose',  action='store_true')
    args = parser.parse_args()
    print(f'executing scrape_hrrr version {version}.')
    end = pd.to_datetime(pd.Timestamp.now().date() + pd.to_timedelta(1, 'd'))
    start = pd.to_datetime('20240101')
    _main_gml(dryrun = args.dry, first = args.first, verbose = args.verbose, start = start, end = end)
    # auto = prolab.Automation(retrieval, product_name='scrape_hrrr')
    # auto.log()
    
if __name__ == '__main__':
    main()