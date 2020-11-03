#!/usr/bin/env python3

# Clean datasets 
# JS, 1.09.2020
# 
# Some of the trajectories generated with the ANDI function
# contained overflows. This program removes those trajectories
# from the dataset.
#
# It seems that all overflows generated the `e16` suffix in the numbers.
#
# Important note:
# * Remove corresponding lines from the `ref` files!

import os

datafolder = 'MyData'
subfolders = [ '10','50','100','150','200','300','400','500','900' ]

os.chdir(datafolder)

for folder in subfolders:
    os.chdir(folder)
    print(os.getcwd())
    is_skipped = False
    counter = 0
    with open('task2.txt') as s1, open('ref2.txt') as s2, open('task2new.txt','w') as t1, open('ref2new.txt','w') as t2:
        for line1,line2 in zip(s1,s2):
            if 'e+16' in line1:
                is_skipped = True
                counter = counter + 1
                continue
            t1.write(line1)
            t2.write(line2)  
        
    if is_skipped:
        os.remove('task2.txt')
        os.rename('task2new.txt','task2.txt')
        os.remove('ref2.txt')
        os.rename('ref2new.txt','ref2.txt')
    else:
        os.remove('task2new.txt')
        os.remove('ref2new.txt')
    print('%d lines skipped'%(counter))        
            
    os.chdir('..')
    
os.chdir('..')
print(os.getcwd())    
    

