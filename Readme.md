# ANDI Challenge contribution
## Janusz Szwabiński
### 3.11.2020

This repository contains the necessary codes written in Python 3 to train the classifiers for Task 2 of the ANDI challenge (see https://competitions.codalab.org/competitions/23601). 

Disclaimer:
The codes are provided as they are. They constitute a quick and dirty solution rather than a well designed application. Thus, use them at your own risk :).

## 1. Required modules 

1. `andi_datasets` for data generation.
2. `numpy` and `pandas` for data handling.
3. `sklearn` for basic ML functionalities.
3. `sktime` for time series classification.
4. `joblib` for storing the classifiers.

## 2. Classification algorithms

### 2.1 Basic assumption

Since most of the classification algoritms for time series working with the raw data require trajectories of the same lengths both for training and classification, we decided to go for the following approach:

1. We prepared 9 different training datasets. Each of these sets contains trajectories of a fixed length $X\in\{10,50,100,150,200,300,400,500,900\}$.
2. For each length, separate classifiers were trained for 1D, 2D and 3D subtasks.
3. In the classification phase:
	* a new trajectory was first cut to match the largest possible length used in the training,
	* a corresponding classifier was chosen to predict its motion type.
	
### 2.2 Algorithms
The major goal for the challenge was to apply algorithms to SPT data, which are taylor-made for time series classification. We obtained the best results with the following methods:

* Random Interval Spectral Ensemble (RISE) in 1D [1], which makes use of several series-to-series feature extraction transformers, including:
	* Fitted auto-regressive coefficients,
	* Estimated autocorrelation coefficients,
	* Power spectrum coefficients.

* MrSEQL in 2D and 3D [2]:
	- converts the numeric time series vector into strings to create multiple symbolic
representations of the time series. The symbolic representations are then used as input for  a sequence learning algorithm, to select the most discriminative subsequence features for training a classifier using logistic regression.

We used the implementations of the algorithms provided by the `sktime` module. In both cases, default parameters provided the best results. 
	
## 3. Usage

1. Download the whole repository and extract it in a directory of your choice.
2. Download the challenge dataset and put it to the same directory.
3. Use the `generate_dataset.py` file to generate training data. Trajectories of a fixed length `X` will be stored in `MyData/X` subfolder of the working directory.
4. Use the `clean_dataset.py` file to remove trajectories containing overflows.
5. Use the `task2-*D.py` files to train the classifiers in 1, 2 and 3 dimensions.
6. Use the `classify.py` code to perform the classification of the challenge dataset.

#### Important note
If you want to use our classifiers, download them and put to `MyData` subfolder of your working directory.

## References

[1] Jason Lines, Sarah Taylor, and Anthony Bagnall. 2018. Time Series Classification with HIVE-COTE: The Hierarchical Vote Collective of Transformation-Based Ensembles. ACM Trans. Knowl. Discov. Data. 12, 5, Article 52 (July 2018), 35 pages.

[2] T. L. Nguyen, S. Gsponer, I. Ilie, M. O'reilly and G. Ifrim Interpretable Time Series Classification using Linear Models and Multi-resolution Multi-domain Symbolic Representations in Data Mining and Knowledge Discovery (DMKD), May 2019, https://doi.org/10.1007/s10618-019-00633-3
