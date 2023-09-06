# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:01:51 2023

@author: Floor

"""
import pandas as pd
import numpy as np
from visualize import tsneEmbed
from sklearn.model_selection import KFold
from skactiveml.pool import RandomSampling, UncertaintySampling, ProbabilisticAL
from sklearn.naive_bayes import GaussianNB
from biasedSample import createAllBiasedSamples
from experiments import runExperiments
import pickle
from activeLearning import MSAL

### BEANS
#Load data and create tsne embedding
beans = pd.read_csv("data\\Dry_Bean_Dataset.csv")
beans.reset_index(drop = True, inplace = True)
beans["Label"] = beans["Class"].map({"SEKER":1, "BARBUNYA":0, "BOMBAY":0, "CALI":0, "HOROZ":1, "SIRA":0, "DERMARSON":0})
label = "Label"
beans.dropna(inplace=True)
beans.drop(labels=["Class"], axis = 1, inplace=True)
beans_embedded = tsneEmbed(beans, label, 30)

#Create 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)
kf.get_n_splits(beans)

#Create for each fold, for each ratio a biased sample to start AL with. 
indeces, ratios, clusters = createAllBiasedSamples(beans, kf, label, 40, 2)

#Run all AL experiments
beans_us_gnb_i, beans_us_gnb_lc, beans_us_gnb_pred, beans_us_gnb_scores = runExperiments(beans, label, kf, UncertaintySampling, GaussianNB(), indeces, ratios, 200)
beans_rs_gnb_i, beans_rs_gnb_lc, beans_rs_gnb_pred, beans_rs_gnb_scores = runExperiments(beans, label, kf, RandomSampling, GaussianNB(), indeces, ratios, 200)
beans_pal_gnb_i, beans_pal_gnb_lc, beans_pal_gnb_pred, beans_pal_gnb_scores = runExperiments(beans, label, kf, ProbabilisticAL, GaussianNB(), indeces, ratios, 200)
beans_msal_gnb_i, beans_msal_gnb_lc, beans_msal_gnb_pred, beans_msal_gnb_scores = runExperiments(beans, label, kf, MSAL, GaussianNB(), indeces, ratios, 200)


with open('beans.pkl', 'wb') as f: 
    pickle.dump([beans_embedded, label, kf, indeces, ratios, clusters],f)

with open('beans_us.pkl', 'wb') as f: 
    pickle.dump([beans_us_gnb_i, beans_us_gnb_lc, beans_us_gnb_pred, beans_us_gnb_scores],f)

with open('beans_rs.pkl', 'wb') as f: 
    pickle.dump([beans_rs_gnb_i, beans_rs_gnb_lc, beans_rs_gnb_pred, beans_rs_gnb_scores],f)

with open('beans_pal.pkl', 'wb') as f: 
    pickle.dump([beans_pal_gnb_i, beans_pal_gnb_lc, beans_pal_gnb_pred, beans_pal_gnb_scores],f)

with open('beans_msal.pkl', 'wb') as f: 
    pickle.dump([beans_msal_gnb_i, beans_msal_gnb_lc, beans_msal_gnb_pred, beans_msal_gnb_scores],f)

