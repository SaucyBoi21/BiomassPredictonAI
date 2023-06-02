# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:10:35 2023
Example script to connect to a campbell datalogger: Using datalogger URL;
@author: jcard
"""

from pycampbellcr1000 import CR1000
import datetime
import pandas as pd
import csv


# Greenhouse 13: 'tcp:172.18.185.8:6785' (copy whole url including )
# Greenhouse 12: ''
#start/end format date: (year,month,day,hour,minute,second)
# add dates like this: (2023,4,3) april 3rd,2023



def import_daily(file_name,date1,date2):
    device = CR1000.from_url('tcp:172.18.185.8:6785')
    # or with Serial connection
    ##device = CR1000.from_url('serial:/dev/ttyUSB0:38400')
    
    device.gettime()
    
    # This will display all data tables availale in your dataloggger
    device.list_tables()
    
    #Set start and stop vaariables according to the time frame you want to extract information.
    start = datetime.datetime(*date1) 
    stop = datetime.datetime(*date2) 
    
    data = device.get_data('DailyIndoors',start,stop)
    keys = data[0].keys()
    
    # Give a name to the csv generated. 
    with open('Indoors_GH13_now.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
        
    df = pd.read_csv('Indoors_GH13_now.csv')
        
    df= df.drop(df.columns[[1,2,3,4]],axis = 1)
    df = df.rename(columns={'Datetime': 'TIMESTAMP'})
    #df.info()
    #print(df[:5])
    
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df = df.rename(columns={df.columns[0]: df.columns[0],
                            **{col: col[2:-1] for col in df.columns[1:]}})
    return df





