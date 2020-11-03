#!/usr/bin/env python3

# Generate datasets for ANDI challenge
# JS, 16.08.2020
# 
# The following code generates data for teaching multiple classifiers on time series with fixed lengths.

import os
from andi import andi_datasets as AD


# set parameters
datafolder = "MyData"
subfolders = [ '10','50','100','150','200','300','400','500','900' ]
ntraj = 30000


# create data folder
if not os.path.exists(datafolder):
    os.makedirs(datafolder)

# generate data
os.chdir(datafolder)
for folder in subfolders:
    if not os.path.exists(folder):
        os.makedirs(folder)
    os.chdir(folder)
    tmax = int(folder)
    print("Trajectory length: ",tmax)
    X1, Y1, X2, Y2, X3, Y3 = AD().andi_dataset(N=ntraj, tasks=2, min_T=tmax, max_T=tmax+1, save_dataset=True)
    os.chdir('..')

os.chdir('..')    
print(os.getcwd())    

