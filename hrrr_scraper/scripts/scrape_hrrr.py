#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I think this should be deprecated!!!  

Created on Wed Nov 11 17:10:44 2020

@author: hagen
"""

from hrrr_scraper import hrrr_lab
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
version = '2.0.2'


### helper

def monitor_processes(name = 'scrape_hrrr', duration = 300, verbose = False):
    process_name = name
    mon_p = [p for p in psutil.process_iter() if p.name() == process_name]
    #### remove the process of this run
    mon_p = [p for p in mon_p if p.pid != pid]
    
    # pid = [p.pid for p in mon_p]

    mon_proc = pd.DataFrame()

    i = 0
    start = datetime.datetime.now()
    while 1:
        now = pd.to_datetime(datetime.datetime.now())
        dt = now-start
        if dt.seconds > duration:
            break
        data = {}
        for e,p in enumerate(mon_p):
            data[f'mem{e}'] = p.memory_percent()
            data[f'cpu{e}'] = p.cpu_percent()
        data
        newrow = pd.DataFrame(data, index = [now])
        # mon_proc = mon_proc.append(newrow)
        mon_proc = pd.concat([mon_proc, newrow])
        if verbose:
            if i == 0:
                print(',\t'.join([f'{col}' for col in newrow.columns]))
            if np.mod(i,300) == 0:
                print(',\t'.join([f'{i:0.2f}' for i in newrow.values[0]]))

        time.sleep(1)
        i+=1
    return mon_proc

def killallhrrrprocesses(test = False):
    if test:
        print('killed')
    else:
        process_name = "scrape_hrrr"
        pmatch = [p for p in psutil.process_iter() if process_name == p.name()]
        #### remove the process of this run
        pmatch = [p for p in pmatch if p.pid != pid]
        
        if len(pmatch) > 0:
            print(f'matching processes: {len(pmatch)}')
            for pm in pmatch:   
                print(pm)
                print('======sdf===========')
                print(f'name: {pm.name()}', end = '\t') 
                print(f'pid: {pm.pid}')
                print('=================')
                print(f'cpu use %: {pm.cpu_percent()}')
                print(f'mem use %: {pm.memory_percent():0.0f}')
                print(f'status%: {pm.status()}')
                pm.kill()
        else:
            print('no match found')

def check_log(path2log = 'my_dict.json',
              maxlength = 50,
              minlength = 4,
              mincpumax = 5,
              verbose = False):
    if verbose:
        print('checklog\n==================')
        
    out = {}
    p2f_shortlog = pl.Path(path2log)
    
    if not p2f_shortlog.is_file():
        if verbose:
            print('no log file, return empty loglist')
            print('++++++++++++++++++++++++')
        out['loglist'] = [{'exit_status':'newlog'}]
        return out
    
    with open(p2f_shortlog) as f:
        loglist = json.load(f)

    # cut to max length
    if len(loglist)>maxlength:
        loglist = loglist[-maxlength:]
        if verbose:
            print('list was cut')

      
    if len(loglist) >= minlength:

        # if last is False, check if the others are false too and then kill processes
        logl = loglist[-minlength:]
        if logl[-1]['exit_status'] == 'clean':
            if verbose:
                print('clean')
        elif 'killed' in [log['exit_status'] for log in logl]:
            if verbose:
                print('killed in logs')
        elif 'error' in [log['exit_status'] for log in logl]:
            if verbose:
                print('error in logs')
        elif 'newlog' in [log['exit_status'] for log in logl]:
            if verbose:
                print('new log file in logs')
        elif 'no_files_on_server'  in [log['exit_status'] for log in logl]:
            if verbose:
                print('No files on server in in last logs')
        else:
            if verbose:
                print('Last is unclean')
                print(np.array([log['cpumax'] for log in logl]))
            if (np.array([log['cpumax'] for log in logl]) > mincpumax).any():
                if verbose:
                    # tt = ', '.join([f'{log['cpumax']:0.2f}' for log in logl])
                    tt = ', '.join([str(round(log['cpumax']))  for log in logl])
                    print(f'At least one has cpumax above threshold (thresh:{mincpumax}, is: {tt})')
            else:
                if verbose:
                    print('killing all scrape_hrrr processes')
                killallhrrrprocesses()
                loglist.append(dict(exit_status = 'killed',
                                    logtime = pd.Timestamp.now().__str__()))
                
                msg = MIMEText('All scrape_hrrr processes killed ... pretty sure they are zombies')

                # me == the sender's email address
                # you == the recipient's email address
                address  = 'hagen.telg@noaa.gov'
                passed = 'Killed'
                msg['Subject'] = 'scrape_hrrr_smoke run - {} - {}'.format(passed, pd.Timestamp(datetime.datetime.now()))
                msg['From'] = address
                msg['To'] = address

                # Send the message via our own SMTP server.
                s = smtplib.SMTP('localhost')
                s.send_message(msg)
                s.quit()
    else:
        if verbose:
            print('list shorter then minlength')
            
    out['loglist'] = loglist
    if verbose:
        print('++++++++++++++++++++++++')
    return out

def _main(dryrun = False, first = False, verbose = False,
          p2f_shortlog = '/home/grad/htelg/script_inits_logs/scrapehrrr.log',
            path2raw = '/home/grad/htelg/tmp/hrrr_tmp/',
            path2projected_individual = '/home/grad/htelg/tmp/hrrr_tmp_inter/',
            path2projected_final = '/nfs/stu3data2/Model_data/HRRR/HRRRv4_conus_projected/',
            ftp_server = 'ftp.ncep.noaa.gov',
            ftp_path2files = '/pub/data/nccf/com/hrrr/prod',
            ):
    
    #### execute the program
    #### -------
    messages = ['run started {}\n========='.format(pd.Timestamp(datetime.datetime.now()))]
    messages.append(f'scrape_hrrr_conus_nat version {version}')
   
    errors = []
    abort = False
    exit_status = 'fail'
    loglist = check_log(path2log=p2f_shortlog, verbose = verbose)['loglist']
    # verbose = True
    
    if verbose:
        print('Begin processing')
        print('=================')
    try:   
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
        messages.append(f'numper of cpus: {no_of_cpu}')
        if no_of_cpu < 1:
            messages.append('aborted since insufficient memory')
            no_of_files_generated = 0
            no_of_files_generated_bdl = 0
            abort = True
            if verbose:
                print('aborted,insufficient memoryg')
            exit_status = 'low_memory'
        elif no_of_processes_running > 0:
            
            messages.append('aborted since processess still running')
            for p in running_processes:
                txt = f'status: {p.status()}'
                messages.append(txt)
                print(txt)
                
            mon_proc = monitor_processes(verbose = verbose)
            messages.append(mon_proc.mean().__str__())
            
            
            no_of_files_generated = 0
            no_of_files_generated_bdl = 0
            abort = True
            exit_status = 'process_running'
            if verbose:
                print('aborted, process still running')
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
            pp =  hrrr_lab.ProjectorProject(gml_sites, 
                                             path2raw = path2raw,
                                             path2projected_individual = path2projected_individual,
                                             path2projected_final = path2projected_final,
                                             ftp_server = ftp_server,
                                             ftp_path2files = ftp_path2files,
                                             max_forcast_interval= 18,
                                             verbose = verbose
                                             )
            if verbose:
                print('generate workplan', end = ' ... ', flush=True)
                pp.workplan
                print('done')
            if first:
                pp.workplan = pp.workplan.iloc[[0], :]
            no_of_files_generated = pp.workplan.shape[0]
            if verbose:
                print(f'number of files to process: {no_of_files_generated}')
            if dryrun:
                return pp.workplan
                
            if no_of_files_generated == 0:
                exit_status = 'no_files_on_server'
            else:
                print(f'no_of_cpu: {no_of_cpu}, {type(no_of_cpu)}')
                out = pp.process(no_of_cpu=no_of_cpu, verbose = verbose)
                
                # out_bdl = scrapelab.concat2daily_files()
                workplan_bdl = out['concat'].workplan
                no_of_files_generated_bdl = workplan_bdl.shape[0]
                # for idx,row in workplan.iterrow():
            #     messages.append(f'{row.path2tempfile} -> {row.path2file}')
                exit_status = 'clean'
                
            if verbose:
                print('Done!!')
    except:
        error, error_txt, trace = sys.exc_info()
        tm = ['{}: {}'.format(error.__name__, error_txt.args[0])] + traceback.format_tb(trace)
        txt = '\n'.join(tm)
        print(txt)
        messages.append(txt)
        errors.append(txt)
        no_of_files_generated = 0
        no_of_files_generated_bdl = 0
    
    if verbose:
        print('++++++++++++++++++++++++++')
        
    messages.append('============\nrun finished {}\n\n'.format(pd.Timestamp(datetime.datetime.now())))
    
    ### generate log text
    message_txt = '\n'.join(messages)
    
    
    #### save log
    ##### shortlog
    
    if verbose:
        print('logging')
        print('=========')
    try:
        log = {}
        log['exit_status'] = exit_status
        log['timestamp'] = pd.Timestamp.now().__str__()
        # loglist = []
        if exit_status == 'process_running':    
            df = mon_proc.mean()
            df = df.loc[[idx for idx in df.index if 'cpu' in idx]]
            log['cpuavg'] = df.mean()#'{:0.2f}'.format(df.mean())
            log['cpumax'] = df.max()#'{:0.2f}'.format(df.max())
            loglist.append(log)
            
        with open(p2f_shortlog, 'w') as f:
            # loglist.to_csv(f)
            json.dump(loglist, f)
        if verbose:
            print('Done')
    except:
        errors.append('logging failed!?!?')
        if verbose:
            print('writing log failed')
    if verbose:
        print('++++++++++++++')
    
    if len(errors) !=0:
        error_txt = '\n\n======================\nERRORS\n=======================\n'
        error_txt += '\n=======================================\n=========================================='.join(errors)
        message_txt += error_txt
    #### send email with results
    if verbose:
        print('Sending email')
        print('==============')
    try:
        # import smtplib
    
        # # Import the email modules we'll need
        # from email.mime.text import MIMEText
    
        # Open a plain text file for reading.  For this example, assume that
        # the text file contains only ASCII characters.
        # with open(textfile) as fp:
        #     # Create a text/plain message
        msg = MIMEText(message_txt)
    
        # me == the sender's email address
        # you == the recipient's email address
        address  = 'hagen.telg@noaa.gov'
        if abort:
            passed = 'Aborted'
        elif exit_status == 'no_files_on_server':
            passed = 'Abnormal (no files on server)'
        elif len(errors) == 0:
            passed = f'Clean ({no_of_files_generated}/{no_of_files_generated_bdl} files)'
        else:
            passed = 'Errors ({})'.format(len(errors))
        msg['Subject'] = 'scrape_hrrr_smoke run - {} - {}'.format(passed, pd.Timestamp(datetime.datetime.now()))
        msg['From'] = address
        msg['To'] = address
    
        # Send the message via our own SMTP server.
        s = smtplib.SMTP('localhost')
        s.send_message(msg)
        s.quit()
        if verbose:
            print('Done')
    except:
        if verbose:
            print('sending email failed')
    
    #### new logging 
    #### TODO: evantuall this should replace the above
    retrieval = type('scrape_hrrr', (), {})
    retrieval.no_processed_success = no_of_files_generated
    retrieval.no_processed_error = len(errors)
    retrieval.no_processed_warning = 0
    
    if verbose:
        print('++++++++++++++++')
    
    if verbose:
        print('End of scirpt!')
    
    return retrieval
    
# if __name__ == '__main__':
def main():
    parser = argparse.ArgumentParser(description = 'Say hello')
    parser.add_argument('-d', '--dry', help='testing: will stop before execution.',  action='store_true')
    parser.add_argument('-f', '--first', help='testing: process only the very first enty.',  action='store_true')
    parser.add_argument('-v', '--verbose', help='verbose',  action='store_true')
    args = parser.parse_args()

    retrieval = _main(dryrun = args.dry, first = args.first, verbose = args.verbose)
    auto = prolab.Automation(retrieval, product_name='scrape_hrrr')
    auto.log()
    
if __name__ == '__main__':
    main()