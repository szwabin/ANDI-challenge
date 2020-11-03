#!/usr/bin/env python3

# Train MrSEQL classifier for 3D case
# JS, 2.10.2020
#
# Usage:
#    ./task2-3D.py N
#
# where N = {10,50,100,150,200,300,400,500,900} is the length of the trajectories in the dataset.

import sys
import numpy as np
import pandas as pd
from andi import andi_datasets as AD

from sktime.classification.shapelet_based import MrSEQLClassifier
from sklearn.model_selection import train_test_split

import joblib

# load the training data
N = int(sys.argv[1])
training_dataset = 'MyData/'+str(N)+'/'
X1, Y1, X2, Y2, X3, Y3 = AD().andi_dataset(load_dataset=True,tasks=2,path_datasets=training_dataset)
X2_3d = X2[2]
Y2_3d = Y2[2]


# convert data to the format required by sktime
# (pandas dataframe, time evolution of each coordinate as a single cell)
traj_x_list = []
traj_y_list = []
traj_z_list = []

for i in X2_3d:
    traj_x_list.append(pd.Series(i[:N]))
    traj_y_list.append(pd.Series(i[N:2*N]))
    traj_z_list.append(pd.Series(i[2*N:]))

x_data = {}
x_data['dim_0'] = pd.Series(traj_x_list)
x_data['dim_1'] = pd.Series(traj_y_list)
x_data['dim_2'] = pd.Series(traj_z_list)

X3 = pd.DataFrame(x_data)
y3 = pd.Series(Y2_3d)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X3, y3)
# check if balanced
labels, counts = np.unique(y_train, return_counts=True)
print(labels, counts)

# classify with MrSEQL
print(" Classifying with MrSEQL...")
clf = MrSEQLClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

fname = 'mrseql3D_'+str(N)+'.clf'
_ = joblib.dump(clf,fname,compress=9)
print("...done!")

