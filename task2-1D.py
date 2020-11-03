#!/usr/bin/env python3

# Train RISE classifier for 1D case
# JS, 2.10.2020
#
# Usage:
#    ./task2-1D.py N
#
# where N = {10,50,100,150,200,300,400,500,900} is the length of the trajectories in the dataset.

import sys
import numpy as np
import pandas as pd
from andi import andi_datasets as AD

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.series_as_features.compose import FeatureUnion
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.segment import     RandomIntervalSegmenter
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from sktime.transformers.series_as_features.reduce import Tabularizer

import joblib

# function definitions (required by the RISE classifier)
def ar_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    lags = np.minimum(len(x) - 1, maxlag) // 2
    model = AutoReg(endog=x, trend="n", lags=lags,old_names=False)
    return model.fit().params.ravel()

def acf_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    nlags = np.minimum(len(x) - 1, maxlag)
    return acf(x, nlags=nlags, fft=True).ravel()

def powerspectrum(x):
    x = np.asarray(x).ravel()
    fft = np.fft.fft(x)
    ps = fft.real * fft.real + fft.imag * fft.imag
    return ps[:ps.shape[0] // 2].ravel()


# load the training data
N = int(sys.argv[1])
training_dataset = 'MyData/'+str(N)+'/'
X1, Y1, X2, Y2, X3, Y3 = AD().andi_dataset(load_dataset=True,tasks=2,path_datasets=training_dataset)
X2_1d = X2[0]
Y2_1d = Y2[0]


# convert data to the format required by sktime
# (pandas dataframe, time evolution of each coordinate as a single cell)
traj_list = []

for i in X2_1d:
    traj_list.append(pd.Series(i))

x_data = {}
x_data['dim_0'] = pd.Series(traj_list)
X1 = pd.DataFrame(x_data)
y1 = pd.Series(Y2_1d)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X1, y1)
# check if balanced
labels, counts = np.unique(y_train, return_counts=True)
print(labels, counts)


# classify with RISE
print(" Classifying with RISE...")
# build the pipeline
steps = [
    ('segment', RandomIntervalSegmenter(n_intervals=1, min_length=5)),
    ('transform', FeatureUnion([
        ('ar', RowTransformer(FunctionTransformer(func=ar_coefs, validate=False))),
        ('acf', RowTransformer(FunctionTransformer(func=acf_coefs, validate=False))),
        ('ps', RowTransformer(FunctionTransformer(func=powerspectrum, validate=False)))
    ])),
    ('tabularise', Tabularizer()),
    ('clf', DecisionTreeClassifier())
]
rise_tree = Pipeline(steps)
# perform the classification
rise = TimeSeriesForestClassifier(estimator=rise_tree, n_estimators=100)
rise.fit(X_train, y_train)
print(rise.score(X_test, y_test))

fname = 'rise_'+str(N)+'.clf'
_ = joblib.dump(rise,fname,compress=9)
print("...done!")