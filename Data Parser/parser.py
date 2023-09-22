# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:41:13 2023

@author: chris
"""

import json
import numpy as np
import pandas as pd

#open the json as read only
with open('schedule.json', 'r') as f:
    data = json.load(f) # loads the data into python
    df = pd.json_normalize(data, record_path=['teams'], errors = 'ignore') #converts input into dataframe.
    #note that schedule is an array
    
    #sorts schedule into a dataframe
    games_df = pd.concat([pd.json_normalize(team['schedule'], meta=['index','timestamp','dayOfWeek','location', 'opponent', 'outcome','pointsScored','pointsAllowed'], errors = 'ignore') for team in data['teams']])
    
    
print(df.info()) #prints out info for the entire dataframe
print(games_df.info()) # prints out info for the team schedule
#replace info() with head() to see the data.


f.close()