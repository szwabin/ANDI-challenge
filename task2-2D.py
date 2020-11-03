#!/usr/bin/env python3


#wymagane biblioteki
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from andi import andi_datasets as AD

from sklearn.pipeline import Pipeline
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.datasets import load_basic_motions
from sktime.transformers.series_as_features.compose import ColumnConcatenator
from sklearn.model_selection import train_test_split


import joblib


# odczyt danych

N = int(sys.argv[1])
training_dataset = str(N)+'/'
X1, Y1, X2, Y2, X3, Y3 = AD().andi_dataset(load_dataset=True,tasks=2,path_datasets=training_dataset)
X2_2d = X2[1]
Y2_2d = Y2[1]


# Przygotowanie danych
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

# Dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X2, y2)
print("Dane treningowe:")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
labels, counts = np.unique(y_train, return_counts=True)
print(labels, counts)


# Time series concatenation
print("Time series concatenation")
steps = [
    ('concatenate', ColumnConcatenator()),
    ('classify', TimeSeriesForestClassifier(n_estimators=100))]
clf = Pipeline(steps)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Acc.: ", acc)

fname = 'tsf2D_'+str(N)+'.clf'
_ = joblib.dump(clf,fname,compress=9)


# Column ensembling
print("Column ensembling with TSF")
clf = ColumnEnsembleClassifier(estimators=[
    ("TSF0", TimeSeriesForestClassifier(n_estimators=100), [0]),
    ("TSF1", TimeSeriesForestClassifier(n_estimators=100), [1]),
    #("BOSSEnsemble3", BOSSEnsemble(max_ensemble_size=5), [3]),
])
clf.fit(X_train, y_train)
#clf.score(X_test, y_test)
acc = clf.score(X_test, y_test)
print("Acc.: ", acc)

fname = 'ce2D_'+str(N)+'.clf'
_ = joblib.dump(clf,fname,compress=9)


print("Column ensembling with BOSS")
clf = ColumnEnsembleClassifier(estimators=[
    ("BOSSEnsemble0", BOSSEnsemble(max_ensemble_size=5), [0]),
    ("BOSSEnsemble1", BOSSEnsemble(max_ensemble_size=5), [1]),
])
clf.fit(X_train, y_train)
#clf.score(X_test, y_test)
acc = clf.score(X_test, y_test)
print("Acc.: ", acc)

fname = 'boss2D_'+str(N)+'.clf'
_ = joblib.dump(clf,fname,compress=9)

# Bespoke classification algorithms
print("MrSEQL")
clf = MrSEQLClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

fname = 'mrseql2D_'+str(N)+'.clf'
_ = joblib.dump(clf,fname,compress=9)

