# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:31:46 2023

@author: Floor
"""

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

with open(path + 'occupancy_final.pkl', 'rb') as f: 
    [occupancy, occupancy_embedded, label, kf, indeces, ratios, clusters] = pickle.load(f)

with open(path + 'occupancy_us_final.pkl', 'rb') as f: 
    [occupancy_us_gnb_i, occupancy_us_gnb_lc, occupancy_us_gnb_pred, occupancy_us_gnb_score] = pickle.load(f)

with open(path + 'occupancy_rs_final.pkl', 'rb') as f: 
    [occupancy_rs_gnb_i, occupancy_rs_gnb_lc, occupancy_rs_gnb_pred, occupancy_rs_gnb_score] = pickle.load(f)

with open(path + 'occupancy_pal_final.pkl', 'rb') as f: 
    [occupancy_pal_gnb_i, occupancy_pal_gnb_lc, occupancy_pal_gnb_pred, occupancy_pal_gnb_score] = pickle.load(f)

with open(path + 'occupancy_msal_final.pkl', 'rb') as f: 
    [occupancy_msal_gnb_i, occupancy_msal_gnb_lc, occupancy_msal_gnb_pred, occupancy_msal_gnb_score] = pickle.load(f)

cycles = 200
dot_size = 1
name = "//test_occupancy//occupancy"
folds = getFolds(kf, occupancy_embedded)
ratios[0] = 1.0

tsneClusterPlot(occupancy_embedded, clusters, name, dot_size)
tsneLabelPlot(occupancy_embedded, occupancy[label], name, dot_size)
tsneIntialSetPlot(occupancy_embedded, clusters, indeces, folds, ratios, name)

test_index = getTestIndex(kf, occupancy)

for fold in test_index:
    classifier_all = GaussianNB()
    classifier_all.fit(occupancy.drop(fold).drop(label, axis = 1), occupancy.drop(fold)[label])
    scores = classifier_all.predict_proba(occupancy.loc[fold].drop(label, axis=1))
    precision, recall, _ = precision_recall_curve(occupancy.loc[fold][label], scores[:,1])
    auc_precision_recall = auc(recall, precision)
    print(classification_report(occupancy.loc[fold][label], classifier_all.predict(occupancy.loc[fold].drop(label, axis =1))))
    print(auc_precision_recall)

### Probabalistic Active Learning
pal_lc_test = getLcTestSet(kf, occupancy, label, occupancy_pal_gnb_pred)
pal_lc_test_avg = getAveragesPerFold(pal_lc_test, cycles)
tsneFinalSetPlot(occupancy_embedded, clusters, indeces, occupancy_pal_gnb_i, folds, ratios, name+"_pal", ['black', 'lightcoral'], "PAL", "Reds")
testSetMissclassificationPlot(occupancy, kf, occupancy_embedded, occupancy_pal_gnb_pred, label, ratios, name+"_pal", "PAL")

### MSAL
msal_lc_test = getLcTestSet(kf, occupancy, label, occupancy_msal_gnb_pred)
msal_lc_test_avg = getAveragesPerFold(msal_lc_test, cycles)
tsneFinalSetPlot(occupancy_embedded, clusters, indeces, occupancy_msal_gnb_i, folds, ratios, name+"_msal", ['black', 'moccasin'], "MSAL", "Oranges")
testSetMissclassificationPlot(occupancy, kf, occupancy_embedded, occupancy_msal_gnb_pred, label, ratios, name+"_msal", "MSAL")


### US
us_lc_test = getLcTestSet(kf, occupancy, label, occupancy_us_gnb_pred)
us_lc_test_avg = getAveragesPerFold(us_lc_test, cycles)
tsneFinalSetPlot(occupancy_embedded, clusters, indeces, occupancy_us_gnb_i, folds, ratios, name+"_us", ['black', 'lightgreen'], "US", "Greens")
testSetMissclassificationPlot(occupancy, kf, occupancy_embedded, occupancy_us_gnb_pred, label, ratios, name+"_us", "US")

### RS
rs_lc_test = getLcTestSet(kf, occupancy, label, occupancy_rs_gnb_pred)
rs_lc_test_avg = getAveragesPerFold(rs_lc_test, cycles)
tsneFinalSetPlot(occupancy_embedded, clusters, indeces, occupancy_rs_gnb_i, folds, ratios, name+"_rs", ['black', 'cornflowerblue'], "RS", "Blues")
testSetMissclassificationPlot(occupancy, kf, occupancy_embedded, occupancy_rs_gnb_pred, label, ratios, name+"_rs", "RS")


learningCurvePerBiasDegreePlot([rs_lc_test_avg, us_lc_test_avg, pal_lc_test_avg, msal_lc_test_avg], ['RS', "US", "PAL", "MSAL"], ratios, name)
learningCurvePerALStrategy([rs_lc_test_avg, us_lc_test_avg, pal_lc_test_avg, msal_lc_test_avg], ['RS', "US", "PAL", "MSAL"], ratios, name, ['blue', 'green', 'red', 'orange'])
PRAUCPerALStrategy([occupancy_rs_gnb_score, occupancy_us_gnb_score, occupancy_pal_gnb_score, occupancy_msal_gnb_score], ['RS', "US", "PAL", "MSAL"], ratios, name, kf, occupancy, label)
F1ScorePerALStrategy([occupancy_rs_gnb_pred, occupancy_us_gnb_pred, occupancy_pal_gnb_pred, occupancy_msal_gnb_pred], ['RS', "US", "PAL", "MSAL"], ratios, name, kf, occupancy, label)

performanceDifferenceClustersPlotAccuracy([occupancy_rs_gnb_pred, occupancy_us_gnb_pred, occupancy_pal_gnb_pred, occupancy_msal_gnb_pred], kf, occupancy, label, clusters, ['RS', "US", "PAL", "MSAL"], ratios, name)
performanceDifferenceClustersPlotF1Score([occupancy_rs_gnb_pred, occupancy_us_gnb_pred, occupancy_pal_gnb_pred, occupancy_msal_gnb_pred], kf, occupancy, label, clusters, ['RS', "US", "PAL", "MSAL"], ratios, name)
performanceDifferenceClustersPlotPRAUC([occupancy_rs_gnb_score, occupancy_us_gnb_score, occupancy_pal_gnb_score, occupancy_msal_gnb_score], kf, occupancy, label, clusters, ['RS', "US", "PAL", "MSAL"], ratios, name)

ratioClustersQueriedIndecesBarPlot(indeces, clusters, [occupancy_rs_gnb_i, occupancy_us_gnb_i, occupancy_msal_gnb_i, occupancy_pal_gnb_i], ["RS", "US", "MSAL", "PAL"], ['blue', 'green', 'orange', 'red'], ratios, name)
ratioLabelsQueriedIndecesBarPlot(indeces, occupancy, label, [occupancy_rs_gnb_i, occupancy_us_gnb_i, occupancy_msal_gnb_i, occupancy_pal_gnb_i], ["RS", "US", "MSAL", "PAL"], ['blue', 'green', 'orange', 'red'], ratios, name)

plotMeansMetrics(occupancy, kf, [occupancy_msal_gnb_i, occupancy_pal_gnb_i, occupancy_us_gnb_i, occupancy_rs_gnb_i], label, ratios, ["MSAL", "PAL", "US", "RS"], ['orange', 'red', 'green', 'blue'], name)
plotMeansMetrics(occupancy, kf, [occupancy_msal_gnb_i, occupancy_pal_gnb_i, occupancy_us_gnb_i, occupancy_rs_gnb_i], label, ratios, ["MSAL", "PAL", "US", "RS"], ['orange', 'red', 'green', 'blue'], name, True)


for i in range(6): 
    ridgePlotColumn("Light", occupancy, [occupancy_msal_gnb_i, occupancy_pal_gnb_i, occupancy_us_gnb_i, occupancy_rs_gnb_i],indeces, name, ["MSAL", "PAL", "US", "RS"], ["orange", "red", "green", 'blue'], i, ratios, 0, 1500)
    ridgePlotClassificationErrorColumn("Light", occupancy, test_index, [occupancy_msal_gnb_pred, occupancy_pal_gnb_pred, occupancy_us_gnb_pred, occupancy_rs_gnb_pred], ["MSAL", "PAL", "US", "RS"], name, i, ratios, label, 199, 0, 1500)
ridgePlotLabel(occupancy, "Light", label, name, 0, 1500)    
    




