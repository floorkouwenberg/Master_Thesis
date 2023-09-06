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

with open(path + 'dielectron_final.pkl', 'rb') as f: 
    [dielectron, dielectron_embedded, label, kf, indeces, ratios, clusters] = pickle.load(f)

with open(path + 'dielectron_us_final.pkl', 'rb') as f: 
    [dielectron_us_gnb_i, dielectron_us_gnb_lc, dielectron_us_gnb_pred, dielectron_us_gnb_score] = pickle.load(f)

with open(path + 'dielectron_rs_final.pkl', 'rb') as f: 
    [dielectron_rs_gnb_i, dielectron_rs_gnb_lc, dielectron_rs_gnb_pred, dielectron_rs_gnb_score] = pickle.load(f)

with open(path + 'dielectron_pal_final.pkl', 'rb') as f: 
    [dielectron_pal_gnb_i, dielectron_pal_gnb_lc, dielectron_pal_gnb_pred, dielectron_pal_gnb_score] = pickle.load(f)

with open(path + 'dielectron_msal_final.pkl', 'rb') as f: 
    [dielectron_msal_gnb_i, dielectron_msal_gnb_lc, dielectron_msal_gnb_pred, dielectron_msal_gnb_score] = pickle.load(f)

cycles = 200
dot_size = 1
name = "//final_electron//electron"
folds = getFolds(kf, dielectron_embedded)
ratios[0] = 1.0

tsneClusterPlot(dielectron_embedded, clusters, name, dot_size)
tsneLabelPlot(dielectron_embedded, dielectron[label], name, dot_size)
tsneIntialSetPlot(dielectron_embedded, clusters, indeces, folds, ratios, name)

test_index = getTestIndex(kf, dielectron)
for fold in test_index:
    classifier_all = GaussianNB()
    classifier_all.fit(dielectron.drop(fold).drop(label, axis = 1), dielectron.drop(fold)[label])
    scores = classifier_all.predict_proba(dielectron.loc[fold].drop(label, axis=1))
    precision, recall, _ = precision_recall_curve(dielectron.loc[fold][label], scores[:,1])
    auc_precision_recall = auc(recall, precision)
    print(classification_report(dielectron.loc[fold][label], classifier_all.predict(dielectron.loc[fold].drop(label, axis =1))))
    print(auc_precision_recall)
    
### Probabalistic Active Learning
pal_lc_test = getLcTestSet(kf, dielectron, label, dielectron_pal_gnb_pred)
pal_lc_test_avg = getAveragesPerFold(pal_lc_test, cycles)
tsneFinalSetPlot(dielectron_embedded, clusters, indeces, dielectron_pal_gnb_i, folds, ratios, name+"_pal", ['black', 'lightcoral'], "PAL", "Reds")
testSetMissclassificationPlot(dielectron, kf, dielectron_embedded, dielectron_pal_gnb_pred, label, ratios, name+"_pal", "PAL")

### MSAL
msal_lc_test = getLcTestSet(kf, dielectron, label, dielectron_msal_gnb_pred)
msal_lc_test_avg = getAveragesPerFold(msal_lc_test, cycles)
tsneFinalSetPlot(dielectron_embedded, clusters, indeces, dielectron_msal_gnb_i, folds, ratios, name+"_msal", ['black', 'moccasin'], "MSAL", "Oranges")
testSetMissclassificationPlot(dielectron, kf, dielectron_embedded, dielectron_msal_gnb_pred, label, ratios, name+"_msal", "MSAL")


### US
us_lc_test = getLcTestSet(kf, dielectron, label, dielectron_us_gnb_pred)
us_lc_test_avg = getAveragesPerFold(us_lc_test, cycles)
tsneFinalSetPlot(dielectron_embedded, clusters, indeces, dielectron_us_gnb_i, folds, ratios, name+"_us", ['black', 'lightgreen'], "US", "Greens")
testSetMissclassificationPlot(dielectron, kf, dielectron_embedded, dielectron_us_gnb_pred, label, ratios, name+"_us", "US")

### RS
rs_lc_test = getLcTestSet(kf, dielectron, label, dielectron_rs_gnb_pred)
rs_lc_test_avg = getAveragesPerFold(rs_lc_test, cycles)
tsneFinalSetPlot(dielectron_embedded, clusters, indeces, dielectron_rs_gnb_i, folds, ratios, name+"_rs", ['black', 'cornflowerblue'], "RS", "Blues")
testSetMissclassificationPlot(dielectron, kf, dielectron_embedded, dielectron_rs_gnb_pred, label, ratios, name+"_rs", "RS")


learningCurvePerBiasDegreePlot([rs_lc_test_avg, us_lc_test_avg, pal_lc_test_avg, msal_lc_test_avg], ['RS', "US", "PAL", "MSAL"], ratios, name)
learningCurvePerALStrategy([rs_lc_test_avg, us_lc_test_avg, pal_lc_test_avg, msal_lc_test_avg], ['RS', "US", "PAL", "MSAL"], ratios, name, ['blue', 'green', 'red', 'orange'])
PRAUCPerALStrategy([dielectron_rs_gnb_score, dielectron_us_gnb_score, dielectron_pal_gnb_score, dielectron_msal_gnb_score], ['RS', "US", "PAL", "MSAL"], ratios, name, kf, dielectron, label)
F1ScorePerALStrategy([dielectron_rs_gnb_pred, dielectron_us_gnb_pred, dielectron_pal_gnb_pred, dielectron_msal_gnb_pred], ['RS', "US", "PAL", "MSAL"], ratios, name, kf, dielectron, label)

performanceDifferenceClustersPlotAccuracy([dielectron_rs_gnb_pred, dielectron_us_gnb_pred, dielectron_pal_gnb_pred, dielectron_msal_gnb_pred], kf, dielectron, label, clusters, ['RS', "US", "PAL", "MSAL"], ratios, name)
performanceDifferenceClustersPlotF1Score([dielectron_rs_gnb_pred, dielectron_us_gnb_pred, dielectron_pal_gnb_pred, dielectron_msal_gnb_pred], kf, dielectron, label, clusters, ['RS', "US", "PAL", "MSAL"], ratios, name)
performanceDifferenceClustersPlotPRAUC([dielectron_rs_gnb_score, dielectron_us_gnb_score, dielectron_pal_gnb_score, dielectron_msal_gnb_score], kf, dielectron, label, clusters, ['RS', "US", "PAL", "MSAL"], ratios, name)

ratioClustersQueriedIndecesBarPlot(indeces, clusters, [dielectron_rs_gnb_i, dielectron_us_gnb_i, dielectron_msal_gnb_i, dielectron_pal_gnb_i], ["RS", "US", "MSAL", "PAL"], ['blue', 'green', 'orange', 'red'], ratios, name)
ratioLabelsQueriedIndecesBarPlot(indeces, dielectron, label, [dielectron_rs_gnb_i, dielectron_us_gnb_i, dielectron_msal_gnb_i, dielectron_pal_gnb_i], ["RS", "US", "MSAL", "PAL"], ['blue', 'green', 'orange', 'red'], ratios, name)

plotMeansMetrics(dielectron, kf, [dielectron_msal_gnb_i, dielectron_pal_gnb_i, dielectron_us_gnb_i, dielectron_rs_gnb_i], label, ratios, ["MSAL", "PAL", "US", "RS"], ['orange', 'red', 'green', 'blue'], name)
plotMeansMetrics(dielectron, kf, [dielectron_msal_gnb_i, dielectron_pal_gnb_i, dielectron_us_gnb_i, dielectron_rs_gnb_i], label, ratios, ["MSAL", "PAL", "US", "RS"], ['orange', 'red', 'green', 'blue'], name, True)

for i in range(6): 
    ridgePlotColumn("Event", dielectron, [dielectron_msal_gnb_i, dielectron_pal_gnb_i, dielectron_us_gnb_i, dielectron_rs_gnb_i],indeces, name, ["MSAL", "PAL", "US", "RS"], ["orange", "red", "green", 'blue'], i, ratios)
    ridgePlotClassificationErrorColumn("Event", dielectron, test_index, [dielectron_msal_gnb_pred, dielectron_pal_gnb_pred, dielectron_us_gnb_pred, dielectron_rs_gnb_pred], ["MSAL", "PAL", "US", "RS"], name, i, ratios, label, 199)
ridgePlotLabel(dielectron, "Event", label, name)    





