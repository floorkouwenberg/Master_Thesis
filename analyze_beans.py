# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:46:25 2023

@author: Floor
"""

import pickle
from visualize import *
from get import getFolds, getLcTestSet, getAveragesPerFold, getTestIndex
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, precision_recall_curve, auc

path = 'results\\gemini_results\\'

with open(path + 'beans_final.pkl', 'rb') as f: 
    [beans, beans_embedded, label, kf, indeces, ratios, clusters] = pickle.load(f)

with open(path + 'beans_us_final.pkl', 'rb') as f: 
    [beans_us_gnb_i, beans_us_gnb_lc, beans_us_gnb_pred, beans_us_gnb_score] = pickle.load(f)

with open(path + 'beans_rs_final.pkl', 'rb') as f: 
    [beans_rs_gnb_i, beans_rs_gnb_lc, beans_rs_gnb_pred, beans_rs_gnb_score] = pickle.load(f)

with open(path + 'beans_pal_final.pkl', 'rb') as f: 
    [beans_pal_gnb_i, beans_pal_gnb_lc, beans_pal_gnb_pred, beans_pal_gnb_score] = pickle.load(f)

with open(path + 'beans_msal_final.pkl', 'rb') as f: 
    [beans_msal_gnb_i, beans_msal_gnb_lc, beans_msal_gnb_pred, beans_msal_gnb_score] = pickle.load(f)

cycles = 200
dot_size = 1
name = "//final_beans//beans"
folds = getFolds(kf, beans_embedded)
ratios[0] = 1.0

tsneClusterPlot(beans_embedded, clusters, name, dot_size)
tsneLabelPlot(beans_embedded, beans[label], name, dot_size)
tsneIntialSetPlot(beans_embedded, clusters, indeces, folds, ratios, name)

test_index = getTestIndex(kf, beans)
for fold in test_index:
    classifier_all = GaussianNB()
    classifier_all.fit(beans.drop(fold).drop(label, axis = 1), beans.drop(fold)[label])
    scores = classifier_all.predict_proba(beans.loc[fold].drop(label, axis=1))
    precision, recall, _ = precision_recall_curve(beans.loc[fold][label], scores[:,1])
    auc_precision_recall = auc(recall, precision)
    print(classification_report(beans.loc[fold][label], classifier_all.predict(beans.loc[fold].drop(label, axis =1))))
    print(auc_precision_recall)
    
### Probabalistic Active Learning
pal_lc_test = getLcTestSet(kf, beans, label, beans_pal_gnb_pred)
pal_lc_test_avg = getAveragesPerFold(pal_lc_test, cycles)
tsneFinalSetPlot(beans_embedded, clusters, indeces, beans_pal_gnb_i, folds, ratios, name+"_pal", ['black', 'lightcoral'], "PAL", "Reds")
testSetMissclassificationPlot(beans, kf, beans_embedded, beans_pal_gnb_pred, label, ratios, name+"_pal", "PAL")

### MSAL
msal_lc_test = getLcTestSet(kf, beans, label, beans_msal_gnb_pred)
msal_lc_test_avg = getAveragesPerFold(msal_lc_test, cycles)
tsneFinalSetPlot(beans_embedded, clusters, indeces, beans_msal_gnb_i, folds, ratios, name+"_msal", ['black', 'moccasin'], "MSAL", "Oranges")
testSetMissclassificationPlot(beans, kf, beans_embedded, beans_msal_gnb_pred, label, ratios, name+"_msal", "MSAL")


### US
us_lc_test = getLcTestSet(kf, beans, label, beans_us_gnb_pred)
us_lc_test_avg = getAveragesPerFold(us_lc_test, cycles)
tsneFinalSetPlot(beans_embedded, clusters, indeces, beans_us_gnb_i, folds, ratios, name+"_us", ['black', 'lightgreen'], "US", "Greens")
testSetMissclassificationPlot(beans, kf, beans_embedded, beans_us_gnb_pred, label, ratios, name+"_us", "US")

### RS
rs_lc_test = getLcTestSet(kf, beans, label, beans_rs_gnb_pred)
rs_lc_test_avg = getAveragesPerFold(rs_lc_test, cycles)
tsneFinalSetPlot(beans_embedded, clusters, indeces, beans_rs_gnb_i, folds, ratios, name+"_rs", ['black', 'cornflowerblue'], "RS", "Blues")
testSetMissclassificationPlot(beans, kf, beans_embedded, beans_rs_gnb_pred, label, ratios, name+"_rs", "RS")


learningCurvePerBiasDegreePlot([rs_lc_test_avg, us_lc_test_avg, pal_lc_test_avg, msal_lc_test_avg], ['RS', "US", "PAL", "MSAL"], ratios, name)
learningCurvePerALStrategy([rs_lc_test_avg, us_lc_test_avg, pal_lc_test_avg, msal_lc_test_avg], ['RS', "US", "PAL", "MSAL"], ratios, name, ['blue', 'green', 'red', 'orange'])
PRAUCPerALStrategy([beans_rs_gnb_score, beans_us_gnb_score, beans_pal_gnb_score, beans_msal_gnb_score], ['RS', "US", "PAL", "MSAL"], ratios, name, kf, beans, label)
F1ScorePerALStrategy([beans_rs_gnb_pred, beans_us_gnb_pred, beans_pal_gnb_pred, beans_msal_gnb_pred], ['RS', "US", "PAL", "MSAL"], ratios, name, kf, beans, label)

performanceDifferenceClustersPlotAccuracy([beans_rs_gnb_pred, beans_us_gnb_pred, beans_pal_gnb_pred, beans_msal_gnb_pred], kf, beans, label, clusters, ['RS', "US", "PAL", "MSAL"], ratios, name)
performanceDifferenceClustersPlotF1Score([beans_rs_gnb_pred, beans_us_gnb_pred, beans_pal_gnb_pred, beans_msal_gnb_pred], kf, beans, label, clusters, ['RS', "US", "PAL", "MSAL"], ratios, name)
performanceDifferenceClustersPlotPRAUC([beans_rs_gnb_score, beans_us_gnb_score, beans_pal_gnb_score, beans_msal_gnb_score], kf, beans, label, clusters, ['RS', "US", "PAL", "MSAL"], ratios, name)

ratioClustersQueriedIndecesBarPlot(indeces, clusters, [beans_rs_gnb_i, beans_us_gnb_i, beans_msal_gnb_i, beans_pal_gnb_i], ["RS", "US", "MSAL", "PAL"], ['blue', 'green', 'orange', 'red'], ratios, name)
ratioLabelsQueriedIndecesBarPlot(indeces, beans, label, [beans_rs_gnb_i, beans_us_gnb_i, beans_msal_gnb_i, beans_pal_gnb_i], ["RS", "US", "MSAL", "PAL"], ['blue', 'green', 'orange', 'red'], ratios, name)

plotMeansMetrics(beans, kf, [beans_msal_gnb_i, beans_pal_gnb_i, beans_us_gnb_i, beans_rs_gnb_i], label, ratios, ["MSAL", "PAL", "US", "RS"], ['orange', 'red', 'green', 'blue'], name)
plotMeansMetrics(beans, kf, [beans_msal_gnb_i, beans_pal_gnb_i, beans_us_gnb_i, beans_rs_gnb_i], label, ratios, ["MSAL", "PAL", "US", "RS"], ['orange', 'red', 'green', 'blue'], name, True)


for i in range(6): 
    ridgePlotColumn("AspectRation", beans, [beans_msal_gnb_i, beans_pal_gnb_i, beans_us_gnb_i, beans_rs_gnb_i],indeces, name, ["MSAL", "PAL", "US", "RS"], ["orange", "red", "green", 'blue'], i, ratios, 0.8, 2.6)
    ridgePlotClassificationErrorColumn("AspectRation", beans, test_index, [beans_msal_gnb_pred, beans_pal_gnb_pred, beans_us_gnb_pred, beans_rs_gnb_pred], ["MSAL", "PAL", "US", "RS"], name, i, ratios, label, 199, 0.8, 2.6)
ridgePlotLabel(beans, "AspectRation", label, name, 0.8, 2.6)    
    





