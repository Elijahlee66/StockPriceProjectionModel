# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:41:13 2020

@author: elijah
"""
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time


api_key = '52IJ61RCT32DB1VZ'
# Using full to have all intraday data
ts = TimeSeries(key = api_key, output_format = 'pandas')
data, meta_data = ts.get_intraday(symbol = 'MSFT', interval = '1min', 
                                  outputsize = 'full')
# Using date as index
ts2 = TimeSeries(key = api_key, output_format = 'pandas', indexing_type='date')
data2,meta_data = ts2.get_intraday(symbol= 'JNJ' , interval = '5min',
                                   outputsize = 'compact')
# Using number as index
ts3 = TimeSeries(key = api_key, output_format = 'pandas', indexing_type='integer')
data3,meta_data = ts3.get_intraday(symbol= 'JNJ' , interval = '5min',
                                   outputsize = 'compact')
# pprint gives more formatting choice when printing
from pprint import pprint
pprint(data3.head(2))

# Technical indicators Sample
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt

# time_period is for number of days
ti = TechIndicators(key = api_key, output_format = 'pandas')
data4,meta_data = ti.get_bbands(symbol = 'BABA',interval = '60min',
    time_period = 24)
data4.plot()
plt.title('BBBands Indicator for BABA Stock (60min)')
plt.show()

# Sector Performance example
from alpha_vantage.sectorperformance import SectorPerformances
import matplotlib.pyplot as plt

sp = SectorPerformances(key = api_key, output_format = 'pandas', 
                        indexing_type = 'date')
data5, mega_data = sp.get_sector()
data5.columns
data5['Rank E: Month Performance'].plot(kind = 'bar')
plt.title('Sector Monthly Performance',fontsize=26)
plt.tight_layout()
plt.show()
