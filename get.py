# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:52:24 2023

@author: Floor
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Gets the train indeces of each fold
def getFolds(kf, df): 
    folds = []
    for i, (train_index, test_index) in enumerate(kf.split(df)):
         folds.append(train_index)
    return folds

#Gets the test indeces of each fold
def getTestIndex(kf, df): 
    testset = []
    for i, (train_index, test_index) in enumerate(kf.split(df)):
         testset.append(test_index)
    return testset

#Gets the average learning curve over all folds
def getAveragesPerFold(lc, cycles): 
    average_lc_per_ratio = [[] for _ in range(len(lc[0]))]
    
    for cycle in range(cycles): 
        for i, ratio in enumerate(lc[0]):
            average_over_folds = np.zeros(5)
            for j, fold in enumerate(lc): 
                average_over_folds[j] = lc[j][i][cycle]
            average_lc_per_ratio[i].append(average_over_folds.mean())
                
    return average_lc_per_ratio

#Shortens the amount of cycles in the results
def getLessCycles(final_indeces, length):
    shorter_list = final_indeces.copy()
    for fold in shorter_list: 
        for ratio in fold: 
            del ratio[length:]
    return shorter_list

#Gets the learning curve of the test set, based on the prediction of the test set
def getLcTestSet(kf, df, label, predictions_test_set): 
    test_index = getTestIndex(kf, df)
    
    lc_per_fold = []    
    for i, fold in enumerate(predictions_test_set): 
        print("Fold", i)
        y_true = df[label].loc[test_index[i]]
        lc_per_ratio = []
        for ratio in fold: 
            lc = []
            for cycle in ratio: 
                lc.append(accuracy_score(y_true, cycle))
            lc_per_ratio.append(lc)
        lc_per_fold.append(lc_per_ratio)
    return lc_per_fold

#Gets the learning curve per clusters
def getLcPerClusterTest(kf, df, label, predictions_test_set, clusters): 
    test_index = getTestIndex(kf, df)
    
    lc0_per_fold = []
    lc1_per_fold = []
    for i, fold in enumerate(predictions_test_set): 
        lc0_per_ratio = []
        lc1_per_ratio = []
        for ratio in fold: 
            lc0 = []
            lc1 = []
            for cycle in ratio: 
                cluster0_true = df[label].loc[test_index[i]].loc[clusters[test_index[i]] == 0]
                cluster1_true = df[label].loc[test_index[i]].loc[clusters[test_index[i]] == 1]
                cluster0_pred = cycle[clusters[test_index[i]] == 0]
                cluster1_pred = cycle[clusters[test_index[i]] == 1]
                lc0.append(accuracy_score(cluster0_true, cluster0_pred))
                lc1.append(accuracy_score(cluster1_true, cluster1_pred))
            lc0_per_ratio.append(lc0)
            lc1_per_ratio.append(lc1)
        lc0_per_fold.append(lc0_per_ratio)
        lc1_per_fold.append(lc1_per_ratio)
    return lc0_per_fold, lc1_per_fold

#Gets the F1 score per cluster
def getF1PerClusterTest(kf, df, label, predictions_test_set, clusters): 
    test_index = getTestIndex(kf, df)
    
    lc0_per_fold = []
    lc1_per_fold = []
    for i, fold in enumerate(predictions_test_set): 
        lc0_per_ratio = []
        lc1_per_ratio = []
        for ratio in fold: 
            lc0 = []
            lc1 = []
            for cycle in ratio: 
                cluster0_true = df[label].iloc[test_index[i]].loc[clusters[test_index[i]] == 0]
                cluster1_true = df[label].iloc[test_index[i]].loc[clusters[test_index[i]] == 1]
                cluster0_pred = cycle[clusters[test_index[i]] == 0]
                cluster1_pred = cycle[clusters[test_index[i]] == 1]
                lc0.append(f1_score(cluster0_true, cluster0_pred))
                lc1.append(f1_score(cluster1_true, cluster1_pred))
            lc0_per_ratio.append(lc0)
            lc1_per_ratio.append(lc1)
        lc0_per_fold.append(lc0_per_ratio)
        lc1_per_fold.append(lc1_per_ratio)
    return lc0_per_fold, lc1_per_fold

#Gets the PR AUC per cluster
def getPRAUCPerCluster(kf, df, label, scores_test_set, clusters): 
    test_index = getTestIndex(kf, df)
    
    prauc0_per_fold = []
    prauc1_per_fold = []
    for i, fold in enumerate(scores_test_set): 
        prauc0_per_ratio = []
        prauc1_per_ratio = []
        for ratio in fold: 
            prauc0 = []
            prauc1 = []
            for cycle in ratio: 
                cluster0_true = df[label].loc[test_index[i]].loc[clusters[test_index[i]] == 0]
                cluster1_true = df[label].loc[test_index[i]].loc[clusters[test_index[i]] == 1]
                cluster0_score = cycle[clusters[test_index[i]] == 0][:,1]
                cluster1_score = cycle[clusters[test_index[i]] == 1][:,1]
                precision0, recall0, _ = precision_recall_curve(cluster0_true, cluster0_score)
                precision1, recall1, _ = precision_recall_curve(cluster1_true, cluster1_score)
                auc_precision_recall0 = auc(recall0, precision0)
                auc_precision_recall1 = auc(recall1, precision1)
                prauc0.append(auc_precision_recall0)
                prauc1.append(auc_precision_recall1)
            prauc0_per_ratio.append(prauc0)
            prauc1_per_ratio.append(prauc1)
        prauc0_per_fold.append(prauc0_per_ratio)
        prauc1_per_fold.append(prauc1_per_ratio)
    return prauc0_per_fold, prauc1_per_fold

#Gets the PRAUC                
def getPRAUC(kf, df, label, scores_test_set): 
    test_index = getTestIndex(kf, df)
    
    prauc_per_fold = []
    for i, fold in enumerate(scores_test_set): 
        prauc_per_ratio = []
        for ratio in fold: 
            prauc = []
            for cycle in ratio: 
                true = df[label].loc[test_index[i]]
                score = cycle[:,1]
                precision, recall, _ = precision_recall_curve(true, score)
                auc_precision_recall = auc(recall, precision)
                prauc.append(auc_precision_recall)
            prauc_per_ratio.append(prauc)
        prauc_per_fold.append(prauc_per_ratio)
    return prauc_per_fold

#Gets the F1 score for the test set
def getF1Test(kf, df, label, predictions_test_set): 
    test_index = getTestIndex(kf, df)
    lc_per_fold = []
    for i, fold in enumerate(predictions_test_set): 
        lc_per_ratio = []
        for ratio in fold: 
            lc = []
            for cycle in ratio: 
                true = df[label].iloc[test_index[i]]
                lc.append(f1_score(true, cycle))    
            lc_per_ratio.append(lc)
        lc_per_fold.append(lc_per_ratio)
    return lc_per_fold
                
#Gets the ratio of each cluster in the queried indeces
def getRatioQueriedIndecesClusters(clusters, indeces_per_fold): 
    n_ratios = len(indeces_per_fold[0])
    ratios_per_bias_ratio = np.zeros(6)
    cluster_min = np.argmin(np.unique(clusters, return_counts=True)[1])
    for i in range(n_ratios):
        for j in range(len(indeces_per_fold)):
            values, counts = np.unique(clusters[indeces_per_fold[j][i]], return_counts = True)
            if len(counts) != 2: 
                if values[0] == cluster_min:         
                    ratios_per_bias_ratio[i]+= 1/5
                else: 
                    ratios_per_bias_ratio[i]+= 0
            else:
                ratios_per_bias_ratio[i]+= counts[cluster_min]/5/200
                    
    
    return ratios_per_bias_ratio

#Gets the ratio of labels in the queried indeces
def getRatioQueriedIndecesLabels(label, df, indeces_per_fold): 
    n_ratios = len(indeces_per_fold[0])
    ratios_per_bias_ratio = np.zeros(n_ratios)
    for i in range(n_ratios):
        for j in range(len(indeces_per_fold)):
            counts = df[label].loc[indeces_per_fold[j][i]].value_counts()[1.0]
            ratios_per_bias_ratio[i]+= counts/5/200
    
    return ratios_per_bias_ratio

#Gets the accuracy of a classifier trained to distinguish between an unbiased
#test set and a final labeled set
def getMetricsUnbiasedVSAL(df, kf, final_sets, label, sample = False, feature_imp = False): 
    accuracy_per_ratio =  [[] for _ in range(len(final_sets[0]))]
    f1_per_ratio = [[] for _ in range(len(final_sets[0]))]
    test_index = getTestIndex(kf, df)
    feature_importances = []
    for i, fold in enumerate(final_sets):
        print("Fold: ", i)
        for j, ratio in enumerate(fold): 
            actively_chosen = df.loc[ratio]
            if sample :
                unbiased = df.loc[test_index[i]].sample(len(actively_chosen), random_state = 0)
            else: 
                unbiased = df.loc[test_index[i]] 
            X_train, X_test, y_train, y_test = train_test_split(pd.concat([actively_chosen, unbiased]).drop(label, axis = 1), [0]*len(actively_chosen)+[1]*len(unbiased) , test_size=0.33, random_state=13)
            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            feature_importances.append(clf.feature_importances_)
            y_pred = clf.predict(X_test)
            accuracy_per_ratio[j].append(accuracy_score(y_test, y_pred))
            f1_per_ratio[j].append(f1_score(y_test, y_pred))
            
    if feature_imp:
        return accuracy_per_ratio, f1_per_ratio, feature_importances
    else: 
        return accuracy_per_ratio, f1_per_ratio


    