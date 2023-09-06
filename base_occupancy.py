"""
Created on Thu Jun 22 12:00:18 2023

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

### occupancy
occupancy = pd.read_csv('data/datatraining.txt', sep=',', index_col=0)
occupancy_2 = pd.read_csv('data/datatest2.txt', sep=',', index_col = 0)
occupancy = pd.concat([occupancy, occupancy_2])
label = "Occupancy"
occupancy.dropna(inplace=True)
occupancy = occupancy.sample(10000)
occupancy.reset_index(inplace = True)
occupancy.drop(['date', 'index'], axis=1, inplace=True)
occupancy_embedded = tsneEmbed(occupancy, label, 30)


#Create 5-fold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)
kf.get_n_splits(occupancy)

#Create for each fold, for each ratio a biased sample to start AL with. 
indeces, ratios, clusters = createAllBiasedSamples(occupancy, kf, label, 40, 2)

#Run all AL experiments
occupancy_us_gnb_i, occupancy_us_gnb_lc, occupancy_us_gnb_pred, occupancy_us_gnb_scores = runExperiments(occupancy, label, kf, UncertaintySampling, GaussianNB(), indeces, ratios, 200)
occupancy_rs_gnb_i, occupancy_rs_gnb_lc, occupancy_rs_gnb_pred, occupancy_rs_gnb_scores = runExperiments(occupancy, label, kf, RandomSampling, GaussianNB(), indeces, ratios, 200)
occupancy_pal_gnb_i, occupancy_pal_gnb_lc, occupancy_pal_gnb_pred, occupancy_pal_gnb_scores = runExperiments(occupancy, label, kf, ProbabilisticAL, GaussianNB(), indeces, ratios, 200)
occupancy_msal_gnb_i, occupancy_msal_gnb_lc, occupancy_msal_gnb_pred, occupancy_msal_gnb_scores = runExperiments(occupancy, label, kf, MSAL, GaussianNB(), indeces, ratios, 200)


with open('occupancy_final.pkl', 'wb') as f: 
    pickle.dump([occupancy, occupancy_embedded, label, kf, indeces, ratios, clusters],f)

with open('occupancy_us_final.pkl', 'wb') as f: 
    pickle.dump([occupancy_us_gnb_i, occupancy_us_gnb_lc, occupancy_us_gnb_pred, occupancy_us_gnb_scores],f)

with open('occupancy_rs_final.pkl', 'wb') as f: 
    pickle.dump([occupancy_rs_gnb_i, occupancy_rs_gnb_lc, occupancy_rs_gnb_pred, occupancy_rs_gnb_scores],f)

with open('occupancy_pal_final.pkl', 'wb') as f: 
    pickle.dump([occupancy_pal_gnb_i, occupancy_pal_gnb_lc, occupancy_pal_gnb_pred, occupancy_pal_gnb_scores],f)

with open('occupancy_msal_final.pkl', 'wb') as f: 
    pickle.dump([occupancy_msal_gnb_i, occupancy_msal_gnb_lc, occupancy_msal_gnb_pred, occupancy_msal_gnb_scores],f)
