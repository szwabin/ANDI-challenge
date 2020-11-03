#!/usr/bin/env python3

# Train MrSEQL classifier for 2D case
# JS, 2.10.2020
#
# Usage:
#    ./task2-2D.py N
#
# where N = {10,50,100,150,200,300,400,500,900} is the length of the trajectories in the dataset.

import sys
import numpy as np
import pandas as pd
from andi import andi_datasets as AD

from sklearn.pipeline import Pipeline
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sklearn.model_selection import train_test_split

import joblib


# load the training data
N = int(sys.argv[1])
training_dataset = 'MyData/'+str(N)+'/'
X1, Y1, X2, Y2, X3, Y3 = AD().andi_dataset(load_dataset=True,tasks=2,path_datasets=training_dataset)
X2_2d = X2[1]
Y2_2d = Y2[1]

# convert data to the format required by sktime
# (pandas dataframe, time evolution of each coordinate as a single cell)
traj_x_list = []
traj_y_list = []

for i in X2_2d:
    traj_x_list.append(pd.Series(i[:N]))
    traj_y_list.append(pd.Series(i[N:]))

x_data = {}
x_data['dim_0'] = pd.Series(traj_x_list)
x_data['dim_1'] = pd.Series(traj_y_list)
X2 = pd.DataFrame(x_data)
y2 = pd.Series(Y2_2d)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X2, y2)
# check if balanced
labels, counts = np.unique(y_train, return_counts=True)
print(labels, counts)



# classify with MrSEQL
print(" Classifying with MrSEQL...")
clf = MrSEQLClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

fname = 'mrseql2D_'+str(N)+'.clf'
_ = joblib.dump(clf,fname,compress=9)
print("...done!")
