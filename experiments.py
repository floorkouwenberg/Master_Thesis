# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 10:17:59 2023

@author: Floor
"""


import numpy as np 
from skactiveml.utils import MISSING_LABEL
from activeLearning import activeLearning
from skactiveml.classifier import SklearnClassifier

'''
#Arguments
df = dataframe
label = string that specifies the column that contains the label
folds = kFold object
query_strategy = query strategy object as defined by skactiveml
classifier = classifier as function handle
sample_indeces = list of indeces per fold and bias ratio that indicate the initial
                 labeled set
bias_ratios = list of values that indicate the bias ratios
n_cycles = number of cycles to run the AL experiments for

#Function
Function that drives AL experiments for a give number folds and bias ratios. 

#Output
queried_indeces_per_fold = a list containing the indeces that have been queried
learning_curve_per_fold =  accuracy obtained  on the train set
predictions_per_fold = predictions as made on the test set
scores_per_fold = scores obtained by classifing the test set
'''


def runExperiments(df, label, folds, query_strategy, classifier, sample_indeces, bias_ratios, n_cycles): 
    queried_indeces_per_fold = [] 
    learning_curve_per_fold = []
    predictions_per_fold = []
    scores_per_fold = []
    
    print("Using query strategy: ", query_strategy.__name__)
    
    for i, (train_index, test_index) in enumerate(folds.split(df)):
        
        print("\n Fold: ", i)
        queried_indeces_per_ratio = []
        learning_curve_per_ratio = []
        classifiers_per_ratio = []
        predictions_per_ratio = []
        scores_per_ratio = [] 

        train_data = df.loc[train_index] 
        test_data = df.loc[test_index]
             

         
        for j, ratio in enumerate(bias_ratios): 
             print("With ratio: ", ratio)
             
             #Create biased sample
             indeces_labeled_set = sample_indeces[i][j]
             
             #Create training data for active learning
             y_train_true = train_data[label]
             x_train = train_data.drop(label, axis=1)
             y_train = y_train_true.copy()
             y_train.loc[~y_train.index.isin(indeces_labeled_set)] = MISSING_LABEL
             x_test = test_data.drop(label, axis =1)
             
             
             #Initialize and do active learning
             clf = SklearnClassifier(classifier, classes=np.unique(y_train_true))
             
             if query_strategy.__name__ == "MSAL": 
                 print("Initialize Query Strategy")
                 qs = query_strategy(x_train)
             else: 
                 qs = query_strategy
             
             print("Start Active Learning Sequence")
             queried_indeces_per_cylce, learning_curve, predictions, scores = activeLearning(qs, x_train, y_train, y_train_true, n_cycles, clf, x_test) 
             scores_per_ratio.append(scores)
             queried_indeces_per_ratio.append(queried_indeces_per_cylce)
             learning_curve_per_ratio.append(learning_curve)
             predictions_per_ratio.append(predictions)
             
        queried_indeces_per_fold.append(queried_indeces_per_ratio)
        learning_curve_per_fold.append(learning_curve_per_ratio)
        scores_per_fold.append(scores_per_ratio)
        predictions_per_fold.append(predictions_per_ratio)
         
    print(" ")
    return queried_indeces_per_fold, learning_curve_per_fold, predictions_per_fold, scores_per_fold
     