# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:35:30 2023

@author: Floor
"""

from sklearn.mixture import GaussianMixture
import numpy as np
from math import floor

'''
#Arguments 
data = dataframe 
label = string that is the name of the column that contains the label
numberofclusters = integer that determines the number of clusters are being made

#Function 
Uses a Gaussian Mixture Model to cluster the data

#Output 
clusters = a list of lenght data that describes to which cluster each instance belongs
'''
def createClusters(data, label, numberofclusters): 
    
    gm = GaussianMixture(n_components=numberofclusters, random_state=0).fit(list(np.array(data.drop(label, axis =1))))
    clusters = gm.predict(list(np.array(data.drop(label, axis =1))))
      
    return clusters

'''
#Arguments 
data = dataframe
samplesize = integer that describes the size of the sample 
clusters = a list of length data that describes the cluster each instance belongs to
ratio = the ratio to which each cluster should be respresented

#Function 
Samples from data and can be used to create a biased sample by changing the ratio
of the clusters compared to the actual ratio

#Output
A list of size samplesize containing the index values of the instances selected
'''
def createBiasedSample(data, samplesize, clusters, ratio):    
    unique, counts = np.unique(clusters, return_counts=True)
    
    if ratio < 0: 
        return data.loc[clusters == unique[np.argmin(counts)]].sample(samplesize).index
    if ratio > 1: 
        return data.loc[clusters == unique[np.argmax(counts)]].sample(samplesize).index
    
    x1 = data.loc[clusters == unique[np.argmin(counts)]].sample(floor(samplesize*ratio)).index
    x0 = data.loc[clusters == unique[np.argmax(counts)]].sample(samplesize - floor(samplesize*ratio)).index
    
    return x1.append(x0)

'''
#Arguments 
data = dataframe
clusters = a list of length data that describes the cluster each instance belongs to

#Function 
Calulates the bias ratios 

#Output
Returns a list of lenght 6 that gives the ratios of the two clusters
'''
def calculateBiasRatios(data, clusters): 
    unique, counts = np.unique(clusters, return_counts=True)
    print(unique, counts)
    unbiased_ratio = min(counts)/(max(counts)+min(counts)) 
    return np.array([-1, 2, 1.5, 1, 0.5, 0])*unbiased_ratio


'''
#Arguments 
df = dataframe
folds = folds object that can be used to split the df
label = string that describes the label column
n = size of the sample 
n_clusters = the number of clusters that should be used

#Function 
Creates for each fold and each bias ratio a sample

#Output 
sample_indeces_per_fold = A list of length folds that contains a list of length 
                          6 that contains a list of length n
ratios = a list of length 6 that contains the ratios
'''
def createAllBiasedSamples(df, folds, label, n, n_clusters): 
    sample_indeces_per_fold = []
    clusters = []
    
    #Create a clustering to use for a biased sample
    clusters = createClusters(df, label, 2)

    #Calculate the ratios necessery for clustering
    ratios = calculateBiasRatios(df, clusters)
    print(ratios)
    
    for i, (train_index, test_index) in enumerate(folds.split(df)):
        sample_indeces_per_ratio = []
        train_data = df.iloc[train_index]         
        train_clusters = clusters[train_index]
        
        for ratio in ratios:
            indeces_sampled_data = createBiasedSample(train_data, n, train_clusters, ratio)
            sample_indeces_per_ratio.append(indeces_sampled_data)
        sample_indeces_per_fold.append(sample_indeces_per_ratio)
        
    return sample_indeces_per_fold, ratios, clusters
             