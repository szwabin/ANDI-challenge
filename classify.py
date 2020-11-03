#!/usr/bin/env python3

# Scoring of the classifiers on challenge data set
# Janusz Szwabiński, 28.10.2020

import numpy as np
import pandas as pd
import joblib
import glob
import csv

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.series_as_features.compose import FeatureUnion
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.segment import     RandomIntervalSegmenter
from sktime.utils.time_series import time_series_slope
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
from sktime.transformers.series_as_features.summarize import RandomIntervalFeatureExtractor
from sktime.transformers.series_as_features.reduce import Tabularizer


from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.transformers.series_as_features.compose import ColumnConcatenator


# Function definitions
# To be able to load the classifiers, we have to re-define all the object that were used while saving them!
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

# helper functions
def find_length(ntraj,dim):
    "Adjust the length of the trajectory to the optimal classifier"
    return lengths[dim][np.digitize(ntraj,lengths[dim][1:])] #w 2. argumencie digitize nie ma 10!!!

def rotate(l, n):
    "Rotate elements in a list"
    return l[-n:] + l[:-n]


# load data for scoring
scoring_dataset = 'challenge_for_scoring/'
trajs_from_files = csv.reader(open(scoring_dataset+'task2.txt','r'), delimiter=';', 
                                        lineterminator='\n',quoting=csv.QUOTE_NONNUMERIC)
validation = [[],[],[]]
for trajs in enumerate(trajs_from_files):
    validation[int(trajs[1][0])-1].append(trajs[1][1:])

X1val , X2val, X3val = validation

# load classifiers
clf_dir = 'MyData/'
clfs = {}
clfs[1] = {}
clfs[2] = {}
clfs[3] = {}

#1D
tpl = "rise1D_*.clf"
for c in glob.glob(clf_dir+tpl):
    print(c)
    l = c.split("_")[-1].split('.')[0]
    l = int(l)
    clf = joblib.load(c)
    clfs[1][l] = clf 

#2D
tpl2D = "mrseql2D_*.clf"
for c in glob.glob(clf_dir+tpl2D):
    print(c)
    l = c.split("_")[-1].split('.')[0]
    l = int(l)
    clf = joblib.load(c)
    clfs[2][l] = clf 

#3D
tpl3D = "mrseql3D_*.clf"
for c in glob.glob(clf_dir+tpl3D):
    print(c)
    l = c.split("_")[-1].split('.')[0]
    l = int(l)
    clf = joblib.load(c)
    clfs[3][l] = clf 


# Classify the challenge data (may take several hours).
# Each trajectory is cut to match the length of a classifier
# and converted to the format required by sktime.

#lengths of available classifiers (some are missing due to limited resources)
lengths = {1 : [10,50,100,150,200,300,400,500], 
           2: [10,50,100,150,200,300,400,500],
           3: [10,50,100,150,200,300]}

results = []
counter = 0
for traj in X1val:
    ltraj = find_length(len(traj),1)
    x_data = {}
    x_data['dim_0'] = pd.Series([traj[:ltraj]])
    X = pd.DataFrame(x_data)
    ypred = clfs[1][ltraj].predict(X)
    results.append((1,ypred[0]))
    counter = counter + 1
    if counter%100 == 0:
        print('1D',counter)


counter = 0
for traj in X2val:
    N = len(traj)//2
    traj_x = traj[:N]
    traj_y = traj[N:]
    ltraj = find_length(N,2)
    x_data = {}
    x_data['dim_0'] = pd.Series([traj_x[:ltraj]])
    x_data['dim_1'] = pd.Series([traj_y[:ltraj]])
    X = pd.DataFrame(x_data)
    ypred = clfs[2][ltraj].predict(X)
    results.append((2,ypred[0]))
    counter = counter + 1
    if counter%100 == 0:
        print('2D',counter)

counter = 0
for traj in X3val:
    N = len(traj)//3
    traj_x = traj[:N]
    traj_y = traj[N:2*N]
    traj_z = traj[2*N:]
    ltraj = find_length(N,3)
    x_data = {}
    x_data['dim_0'] = pd.Series([traj_x[:ltraj]])
    x_data['dim_1'] = pd.Series([traj_y[:ltraj]])
    x_data['dim_2'] = pd.Series([traj_z[:ltraj]])
    X = pd.DataFrame(x_data)
    ypred = clfs[3][ltraj].predict(X)
    results.append((3,ypred[0]))
    counter = counter + 1
    if counter%100 == 0:
        print(counter)


# save results (in ANDI format)
template = [' 1;', ' 0;', ' 0;', ' 0;', ' 0;']
with open("task2.txt",'w') as file:
    for d,r in results:
        txt = str(d)+';'+''.join(rotate(template,int(r)))
        txt = txt[:-1]+'\n' #remove the last ';' before saving
        file.write(txt)
