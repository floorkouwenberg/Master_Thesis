# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:01:51 2023

@author: Floor
"""

from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from get import *
import math
from joypy import joyplot
from sklearn.neighbors import NearestNeighbors


#Creates the Tsne embedding of the data
def tsneEmbed(data, label, perplex):
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', verbose=True, perplexity=perplex, random_state=0).fit_transform(data.drop(label, axis=1))
    df = pd.DataFrame()
    df["y"] = data[label]
    df["comp-1"] = X_embedded[:,0]
    df["comp-2"] = X_embedded[:,1]
       
    return df  

#Creates a cluster plot    
def tsneClusterPlot(x_embedded, clusters, name, size): 
    colormap = np.array(['darkviolet', 'violet', 'lightblue', 'red', 'green', 'orange', 'yellow', 'darkblue'])
    #fig, ax = plt.subplots()
    plt.rcParams["figure.figsize"] = (8,6)
    
    plt.scatter(x_embedded["comp-1"], x_embedded['comp-2'], s=size, c = colormap[clusters])
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.title("Cluster Distribution")
    plt.legend()
    plt.savefig("results/" + name + "ClusterDistribution.pdf", bbox_inches='tight')
    plt.show()
    
#Creates a label plot    
def tsneLabelPlot(x_embedded, y, name, size): 
    colormap = np.array(['darkblue', 'lightblue'])
    plt.rcParams["figure.figsize"] = (8,6)
    plt.scatter(x_embedded["comp-1"], x_embedded['comp-2'], s=size, c = colormap[y.to_numpy(dtype = 'int32')])
    plt.tick_params(left = False, right = False , labelleft = False, labelbottom = False, bottom = False)
    plt.title("Class Distribution")
    plt.legend()
    plt.savefig("results/" + name + "LabelDistribution.pdf", bbox_inches='tight')
    plt.show()
 
    
#Plots the instances that were used as initial labled sets. The two clusters 
#are also shown    
def tsneIntialSetPlot(x_embedded, clusters, indeces, folds, ratios, name): 
    colormap = np.array(['darkgray', 'lightgray'])
    embedded_data = x_embedded.loc[folds[0]]
    embedded_data["colors"] =  clusters[folds[0]]
    plt.rcParams["figure.figsize"] = (8,6)
    fig, axs = plt.subplots(2, 3)
    setcolor = "black"
    
    for i, ax in enumerate(fig.axes): 
        ax.scatter(embedded_data["comp-1"], embedded_data['comp-2'], s=1, c = colormap[embedded_data['colors']])
        ax.scatter(embedded_data.loc[indeces[0][i]]['comp-1'], embedded_data.loc[indeces[0][i]]['comp-2'], s=2, c = setcolor)                                          
        ax.set_title('Bias ratio: ' + str(ratios[i].round(2)))
    
    for ax in axs:
        for a in ax: 
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
    
    
    fig.suptitle("Initial Labeled Set")            
    fig.tight_layout()
    plt.savefig("results/" + name + "InitialLabledSet.pdf")
    
    plt.show()

#Plots the final chosen indeces after active learning for the first fold and
#plots the initial indeces and the clusters as well    
def tsneFinalSetPlot(x_embedded, clusters, initial_indeces, final_indeces, folds, ratios, name, colours, figure_name, colourmap=None):
    colormap = np.array(['darkgray', 'lightgray'])
    embedded_data = x_embedded.loc[folds[0]]
    embedded_data["colors"] =  clusters[folds[0]]
    plt.rcParams["figure.figsize"] = (14,3)
    fig, axs = plt.subplots(1, 6)
    
    
    for i, ax in enumerate(fig.axes): 
        ax.scatter(embedded_data["comp-1"], embedded_data['comp-2'], s=1, c = colormap[embedded_data['colors']])
        ax.scatter(embedded_data.loc[initial_indeces[0][i]]['comp-1'], embedded_data.loc[initial_indeces[0][i]]['comp-2'], s=7, c = colours[0])                                          
        
        if not colourmap: 
            ax.scatter(embedded_data.loc[final_indeces[0][i]]['comp-1'], embedded_data.loc[final_indeces[0][i]]['comp-2'], s=4, c = colours[1])                                          
        else: 
            plot = ax.scatter(embedded_data.loc[final_indeces[0][i]]['comp-1'], embedded_data.loc[final_indeces[0][i]]['comp-2'], s=4, c = range(200), cmap = truncateColormap(colourmap, 0.25, 0.95).reversed())                                          
        
        
        ax.set_title('Bias ratio: ' + str(ratios[i].round(2)))
    
    #for ax in axs:
    for a in axs: 
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    
    
    fig.suptitle("Final Labeled Set using " + figure_name)     
    fig.tight_layout()
    fig.colorbar(plot, ax=axs.ravel().tolist())
    plt.savefig("results/" + name + "FinalLabledSet.pdf", bbox_inches='tight')
    
    plt.show()

#Plots the learnig curve per bais degree, so so there is a plot for each query
#strategy that includes 6 lines. One for each bias degree.     
def learningCurvePerBiasDegreePlot(list_of_average_predictions_results, names_strategy, ratios, name):

    plt.rcParams["figure.figsize"] = (8,6)    
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8, 6))
    cycles = len(list_of_average_predictions_results[0][0])
    
    for i, ax in enumerate(fig.axes): 

        for k, bias_level in enumerate(list_of_average_predictions_results[i]):
            ax.plot(list(range(1, cycles+1)), bias_level, label = ratios[k].round(2))
            ax.title.set_text(names_strategy[i])
    
    fig.suptitle("Learning Curves per Query Method" )
    fig.supxlabel("Cycle")
    fig.supylabel("Average Accuracy")
    fig.tight_layout()
    
    plt.legend(title = "Bias Degree", loc = "center right", bbox_to_anchor=(1.4, 1.0))
    plt.savefig("results/" + name + "LearningCurveBiasDegree.pdf", bbox_inches='tight')
    
    plt.show()
    
#Plots the learning curve per query strategy. So there are six plots, each with
#four lines representing each query strategy
def learningCurvePerALStrategy(list_of_average_prediction_results, names_of_strategy, ratios, name, colours): 
    plt.rcParams["figure.figsize"] = (8,6)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    colours = ['blue', 'green', 'red', 'orange']
    cycles = len(list_of_average_prediction_results[0][0])
    for i, ax in enumerate(fig.axes): 
        for j, AL_Strategy in enumerate(list_of_average_prediction_results): 
            ax.plot(list(range(1, cycles+1)), AL_Strategy[i], label = names_of_strategy[j], c=colours[j])
            
        ax.set_title('Bias ratio: ' + str(ratios[i].round(2)))
        
    fig.supxlabel("Cycle")
    fig.supylabel("Average Accuracy")
    fig.suptitle("Learning Curve per Bias Ratio")
    fig.tight_layout()
    
    plt.legend(title = "Query Method", loc = "center right", bbox_to_anchor=(1.8, 1.2))
    
    plt.savefig("results/" + name + "LearningCurveALStrategy.pdf", bbox_inches='tight')
        
    plt.show()

#Plots the PR AUC for each bias ratio and AL strategy
def PRAUCPerALStrategy(list_of_scores, names_of_strategy, ratios, name, kf, df, label): 
    
    average_prauc_list = []
    for score in list_of_scores: 
        print("Calculating PRAUC")
        prauc_per_fold = getPRAUC(kf, df, label, score)
        average_prauc = getAveragesPerFold(prauc_per_fold, 200)
        average_prauc_list.append(average_prauc)
        
    plt.rcParams["figure.figsize"] = (8,6)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    colours = ['blue', 'green', 'red', 'orange']
    cycles = len(average_prauc_list[0][0])
    
    for i, ax in enumerate(fig.axes): 
        for j, AL_Strategy in enumerate(average_prauc_list): 
            ax.plot(list(range(1, cycles+1)), AL_Strategy[i], label = names_of_strategy[j], c=colours[j])
            
        ax.set_title('Bias ratio: ' + str(ratios[i].round(2)))
        
    fig.supxlabel("Cycle")
    fig.supylabel("Average PR AUC ")
    
    fig.suptitle("PR AUC per Bias Ratio")
    fig.tight_layout()
    
    plt.legend(title = "Query Method", loc = "center right", bbox_to_anchor=(1.8, 1.2))
    
    plt.savefig("results/" + name + "PRAUCALStrategy.pdf", bbox_inches='tight')
        
    plt.show()


#Plots the F1 score for each bias ratio and AL strategy    
def F1ScorePerALStrategy(list_of_predictions, names_of_strategy, ratios, name, kf, df, label): 
    
    average_f1_score_list = []
    for pred in list_of_predictions: 
        print("Calculating F1-score")
        f1_score_per_fold = getF1Test(kf, df, label, pred)
        average_f1_score = getAveragesPerFold(f1_score_per_fold, 200)
        average_f1_score_list.append(average_f1_score)
        
    plt.rcParams["figure.figsize"] = (8,6)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    colours = ['blue', 'green', 'red', 'orange']
    cycles = len(average_f1_score_list[0][0])
    
    for i, ax in enumerate(fig.axes): 
        for j, AL_Strategy in enumerate(average_f1_score_list): 
            ax.plot(list(range(1, cycles+1)), AL_Strategy[i], label = names_of_strategy[j], c=colours[j])
            
        ax.set_title('Bias ratio: ' + str(ratios[i].round(2)))
        
    fig.supxlabel("Cycle")
    fig.supylabel("Average F1-score")
    
    fig.suptitle("F1-score per Bias Ratio")
    fig.tight_layout()
    
    plt.legend(title = "Query Method", loc = "center right", bbox_to_anchor=(1.8, 1.2))
    
    plt.savefig("results/" + name + "F1ScoreALStrategy.pdf", bbox_inches='tight')
        
    plt.show()

#Plots the accuracy (learing curve) for each cluster. There are 
#six plots, one for each bias ratio each containing 8 lines, one for each cluster
#query startegy combo
def performanceDifferenceClustersPlotAccuracy(list_of_predictions_test_set, kf, df, label, clusters, names_of_strategy, ratios, name):
    average_predictions = []
    plt.rcParams["figure.figsize"] = (8,6)
    print("Finding results per cluster...")
    for predictions in list_of_predictions_test_set: 
        lc0, lc1 = getLcPerClusterTest(kf, df, label, predictions, clusters)
        lc0_average = getAveragesPerFold(lc0, 200)
        lc1_average = getAveragesPerFold(lc1, 200)
        average_predictions.append([lc0_average, lc1_average])
    

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    cycles = 200
    colours = ['darkblue', 'lightblue', 'darkgreen', 'lightgreen', 'darkred', 'lightcoral', 'darkorange', 'bisque']
    colour_i = 0
    
    print("Creating Plots...")
    for i, ax in enumerate(fig.axes): 
        for j, AL_Strategy in enumerate(average_predictions): 
            ax.plot(list(range(1, cycles+1)), AL_Strategy[0][i], label = names_of_strategy[j] + '0', c = colours[colour_i])
            ax.plot(list(range(1, cycles+1)), AL_Strategy[1][i], label = names_of_strategy[j] + '1', c = colours[colour_i + 1])
            colour_i += 2
        ax.set_title('Bias ratio: ' + str(ratios[i].round(2)))
        colour_i = 0
        
    fig.supxlabel("Cycle")
    fig.supylabel("Average Accuracy")
    
    fig.suptitle("Learning Curves for both Clusters per Bias Ratio")
    fig.tight_layout()
    
    plt.legend(title = "Query Method \nand Cluster", loc = "center right", bbox_to_anchor=(1.8, 1.2))
    
    plt.savefig("results/" + name + "ClusterDifferenceLearningCurve.pdf", bbox_inches='tight')
        
    plt.show()

#Plots the F1-score for each cluster. There are 
#six plots, one for each bias ratio each containing 8 lines, one for each cluster
#query startegy combo
def performanceDifferenceClustersPlotF1Score(list_of_predictions_test_set, kf, df, label, clusters, names_of_strategy, ratios, name):
    average_predictions = []
    plt.rcParams["figure.figsize"] = (8,6)
    print("Finding results per cluster...")
    for predictions in list_of_predictions_test_set: 
        lc0, lc1 = getF1PerClusterTest(kf, df, label, predictions, clusters)
        lc0_average = getAveragesPerFold(lc0, 200)
        lc1_average = getAveragesPerFold(lc1, 200)
        average_predictions.append([lc0_average, lc1_average])
    

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    cycles = 200
    colours = ['cornflowerblue', 'darkblue', 'lightgreen', 'darkgreen', 'lightcoral', 'darkred', 'moccasin', 'darkorange']
    colour_i = 0
    
    print("Creating Plots...")
    for i, ax in enumerate(fig.axes): 
        for j, AL_Strategy in enumerate(average_predictions): 
            ax.plot(list(range(1, cycles+1)), AL_Strategy[0][i], label = names_of_strategy[j] + '0', c = colours[colour_i])
            ax.plot(list(range(1, cycles+1)), AL_Strategy[1][i], label = names_of_strategy[j] + '1', c = colours[colour_i + 1])
            colour_i += 2
        ax.set_title('Bias ratio: ' + str(ratios[i].round(2)))
        colour_i = 0
        
    fig.supxlabel("Cycle")
    fig.supylabel("Average F1-score")
    
    fig.suptitle("F1-score for both Clusters per Bias Ratio")
    fig.tight_layout()
    
    plt.legend(title = "Query Method \nand Cluster", loc = "center right", bbox_to_anchor=(1.8, 1.2))
    
    plt.savefig("results/" + name + "ClusterDifferenceF1Score.pdf", bbox_inches='tight')
        
    plt.show()
    
#Plots the PR AUC for each cluster. There are 
#six plots, one for each bias ratio each containing 8 lines, one for each cluster
#query startegy combo    
def performanceDifferenceClustersPlotPRAUC(list_of_scores_test_set, kf, df, label, clusters, names_of_strategy, ratios, name):
    average_areas = []
    plt.rcParams["figure.figsize"] = (8,6)
    print("Finding results per cluster...")
    for scores in list_of_scores_test_set: 
        lc0, lc1 = getPRAUCPerCluster(kf, df, label, scores, clusters)
        prauc0_average = getAveragesPerFold(lc0, 200)
        prauc1_average = getAveragesPerFold(lc1, 200)
        average_areas.append([prauc0_average, prauc1_average])
    

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    cycles = 200
    colours = ['cornflowerblue', 'darkblue', 'lightgreen', 'darkgreen', 'lightcoral', 'darkred', 'moccasin', 'darkorange']
    colour_i = 0
    
    print("Creating Plots...")
    for i, ax in enumerate(fig.axes): 
        for j, AL_Strategy in enumerate(average_areas): 
            ax.plot(list(range(1, cycles+1)), AL_Strategy[0][i], label = names_of_strategy[j] + '0', c = colours[colour_i])
            ax.plot(list(range(1, cycles+1)), AL_Strategy[1][i], label = names_of_strategy[j] + '1', c = colours[colour_i + 1])
            colour_i += 2
        if ratios[i] < 0: 
            ax.set_title('Bias ratio: 1.0')
        else: 
            ax.set_title('Bias ratio: ' + str(ratios[i].round(2)))
        colour_i = 0
        
    fig.supxlabel("Cycle")
    fig.supylabel("Average PR AUC")
    
    fig.suptitle("PR AUC for both Clusters per Bias Ratio")
    fig.tight_layout()
    
    plt.legend(title = "Query Method \nand Cluster", loc = "center right", bbox_to_anchor=(1.8, 1.2))
    
    plt.savefig("results/" + name + "ClusterDifferencePRAUC.pdf", bbox_inches='tight')
        
    plt.show()

#Plots the missclassifcation error in the test set using the t-sne embedding    
def testSetMissclassificationPlot(df, kf, x_embedded, predictions, label, ratios, name, figure_name, averaged=False): 
    colormap = truncateColormap('brg_r', 0, 0.5)#cmap_map(lambda x: x/2 + 0.5, truncateColormap('brg_r', 0, 0.5))
    test_index = getTestIndex(kf, df)
    embedded_data = x_embedded.loc[test_index[0]]
    plt.rcParams["figure.figsize"] = (14,3)
    fig, axs = plt.subplots(1, 6)
    
    
    for i, ax in enumerate(fig.axes): 
        if averaged: 
            colorsss = neighbourColours(df, x_embedded, test_index, predictions, i, label)
        else: 
            colorsss = abs(np.sum(predictions[0][i], axis=0)/200 - df.loc[test_index[0]][label])
        plot = ax.scatter(embedded_data["comp-1"], embedded_data['comp-2'], s=1,c = colorsss, cmap = colormap)
 
        
        ax.set_title('Bias ratio: ' + str(ratios[i].round(2)))
    
    #for ax in axs:
    for a in axs: 
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
    
    
    fig.suptitle("Average Missclassification Error over 200 Cycles using " + figure_name) 
    fig.tight_layout()
    fig.colorbar(plot, ax=axs.ravel().tolist())

    
    plt.savefig("results/" + name + "_missclassification.pdf", bbox_inches='tight')
    
    plt.show()
        
#Shortens a color map    
def truncateColormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap  

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

#Shortens a reversed color map
def truncateColormapRerversed(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).r
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap 

#Plots the ratio of clusters in the queried instances per bias ratio and AL strategy
def ratioClustersQueriedIndecesBarPlot(indeces, clusters, list_of_queried_instances, list_of_names, list_of_colours, ratios, name): 
    plt.rcParams["figure.figsize"] = (8,6)
    ratios_queried_instances = []
    for item in list_of_queried_instances: 
        ratios_queried_instances.append(getRatioQueriedIndecesClusters(clusters, item))
    
    # create data
    x = np.arange(len(ratios))
    width = 0.1
    offset = [-0.1, 0, 0.1, 0.2]
    ratios[0] = 1
    
    plt.bar(x-0.2, ratios, width, color = 'violet')
    # plot data in grouped manner of bar type
    for i, ratioos in enumerate(ratios_queried_instances): 
        plt.bar(x+offset[i], ratioos, width, color=list_of_colours[i])
        

    plt.xlabel("Bias Ratios")
    plt.xticks(x, [str(i) for i in ratios.round(2)])
    plt.ylabel("Cluster Ratio")
    plt.legend(['Bias Ratio'] + list_of_names)
    plt.suptitle("Cluster Ratio in the Final Labeled Set") 
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    
    plt.savefig("results/" + name + "_ratiosClusterBarPlot.pdf", bbox_inches='tight')
    
    plt.show()


#Plots the ratio of labels in the queried instances per bias ratio and AL strategy    
def ratioLabelsQueriedIndecesBarPlot(indeces, df, label, list_of_queried_instances, list_of_names, list_of_colours, ratios, name): 
    plt.rcParams["figure.figsize"] = (8,6)
    ratios_queried_instances = []
    for item in list_of_queried_instances: 
        ratios_queried_instances.append(getRatioQueriedIndecesLabels(label, df, item))
    
    # create data
    x = np.arange(len(ratios))
    width = 0.1
    offset = [-0.1, 0, 0.1, 0.2]
    
    initial_ratio = np.zeros(6)
    for i, fold in enumerate(indeces): 
        for j, ratio in enumerate(fold): 
            initial_ratio[j] += df[label].loc[ratio].value_counts()[1.0]/5/40
    
    plt.bar(x-0.2, initial_ratio, width, color ='lightblue')
    # plot data in grouped manner of bar type
    for i, ratioos in enumerate(ratios_queried_instances): 
        plt.bar(x+offset[i], ratioos, width, color=list_of_colours[i])
        

    plt.xlabel("Bias Ratios")
    plt.xticks(x, [str(i) for i in ratios.round(2)])
    plt.ylabel("Label Ratio")
    plt.ylim([0.0, 1.0])
    plt.legend(['Ratio in Intial Set'] + list_of_names)
    plt.suptitle("Label Ratio in the Final Labeled Set") 
    plt.tight_layout()
    
    plt.savefig("results/" + name + "_ratiosLabelBarPlot.pdf", bbox_inches='tight')
    
    plt.show()

#Plots the mean of the accuracy obtained from distinguishing between an
#unbiased test set and a final labeled set    
def plotMeansMetrics(df, kf, list_of_final_sets, label, ratios, list_of_names, list_of_colours, name, sample=False): 
    plt.figure(figsize =(8, 6))
    
    for i, final_set in enumerate(list_of_final_sets): 
       metric, _ = getMetricsUnbiasedVSAL(df, kf, final_set, label, sample) 
       metric_mean = [np.mean(ratio) for ratio in metric]
       plt.scatter(range(len(ratios)), metric_mean, c=list_of_colours[i])
       
    plt.title('Distinguishability of Test Set and Final AL set')
    plt.xlabel("Bias Ratios")
    plt.xticks([0, 1, 2, 3, 4, 5], [str(i) for i in ratios.round(2)])
    plt.ylabel("Average Accuracy")
    plt.legend(list_of_names, title = "Query Method")
    plt.tight_layout()
    
    if sample: 
        plt.savefig("results/" + name + "_classifierAccuracyWithSample.pdf", bbox_inches='tight')
    else: 
        plt.savefig("results/" + name + "_classifierAccuracy.pdf", bbox_inches='tight')
    
    plt.show()

#Creates a ridgeplot based on the indeces queried per active learning strategy
#The density is shown as it changes troughout 200 cycles.    
def ridgePlotColumn(column_name, df, list_of_indeces, initial_indeces, name, list_of_names, list_of_colours, ratio_level, ratios, limitsa = None, limitsb = None): 
    n_initial_indeces = len(initial_indeces[0][0])
    
    ridgePlot = df[[column_name]][0:(200+n_initial_indeces)]
    
    for item in list_of_names: 
        ridgePlot[name] = None
    ridgePlot["Cycle"] = None
    
    
    
    for i, row in ridgePlot.iterrows(): 
        if i < n_initial_indeces: 
            ridgePlot.at[i,'Cycle'] = 0
            for item in list_of_names: 
              ridgePlot.at[i,item] = df[column_name].loc[initial_indeces[0][ratio_level][i]]  
        else:
            ridgePlot.at[i,'Cycle'] = math.trunc((i-n_initial_indeces)/50)*50 + 50
            
            for j, item in enumerate(list_of_names):   
                ridgePlot.at[i, item] = df[column_name].loc[list_of_indeces[j][0][ratio_level][i-n_initial_indeces]]
    
    ridgePlotBig = ridgePlot.copy()
    for period in [0, 50, 100, 150]: 
        for destination in [50, 100, 150, 200]: 
            if destination > period: 
                ridgePlotBig = pd.concat([ridgePlot[ridgePlot["Cycle"] == period].assign(Cycle = destination), ridgePlotBig])
    ridgePlot = ridgePlotBig
    
    for item in list_of_names: 
        ridgePlot[item] = ridgePlot[item].astype('float')
        
    plt.figure()        
    fig, axs = joyplot(
        fade = True,
        data=ridgePlot[list_of_names + ["Cycle"]], 
        by='Cycle',
        legend = True,
        fill = False,
        ylim='own',
        color = list_of_colours,
        figsize=(8, 6)
    )
    if limitsa: 
        plt.xlim(limitsa, limitsb)
        axs[-1].set_xticks(range(7))
        axs[-1].set_xticklabels(np.linspace(limitsa, limitsb, 7))
        
    fig.supylabel("Cycle \n", x=0, y=0.5)
    plt.title('Bias Ratio: ' + str(ratios[ratio_level].round(2)), fontsize=20)
    plt.xlabel(column_name)
    
    plt.savefig("results/" + name + column_name + str(ratio_level)+ "_ridgelinePlot.pdf", bbox_inches='tight')
    plt.show()
       
#Gives a ridge line plot of the label division in a given column
def ridgePlotLabel(df, column_name, label, name, limitsa=None, limitsb=None): 
    fig, axs = joyplot(
        fade = False,
        data=df[[column_name, label]], 
        by=label,
        fill = False,
        figsize=(8, 6),
        overlap = 8,
        color = ["darkblue", 'lightblue'],
        title = "Class Distribution"
    )
    
    if limitsa: 
        plt.xlim(limitsa, limitsb)
        axs[-1].set_xticks(range(7))
        axs[-1].set_xticklabels(np.linspace(limitsa, limitsb, 7))
        
    plt.title("Class Distribution", fontsize=20)
    plt.xlabel(column_name)
    
    plt.savefig("results/" + name + column_name + "_ridgelineLabelPlot.pdf", bbox_inches='tight')
    plt.show()

#Creates a ridgeplot based on the indeces queried per active learning strategy
#The density of classifiying instances (in-)correctly is shown per AL strategy 
def ridgePlotClassificationErrorColumn(column_name, df, test_index, list_of_predictions, AL_list, name, ratio_level, ratios, label, cycle, limitsa = None, limitsb = None): 
    n_test = len(test_index[0])
    ridgePlot = df[[column_name]][0:n_test*4]
    
    for item in ["Correct", "Incorrect", "AL_type"]: 
        ridgePlot[item] = np.nan

    for i in range(n_test): 
        for j, AL in enumerate(AL_list): 
            
            ridgePlot.at[i+(j*n_test),"AL_type"] = AL
            if list_of_predictions[j][0][ratio_level][cycle][i] == df.loc[test_index[0][i]][label]:                
                ridgePlot.at[i+(j*n_test),"Correct"] = df[column_name].loc[test_index[0][i]]  
            else: 
                ridgePlot.at[i+(j*n_test),"Incorrect"] = df[column_name].loc[test_index[0][i]]
        
    plt.figure()        
    fig, axs = joyplot(
        fade = False,
        data=ridgePlot[["Correct", "Incorrect", "AL_type"]], 
        by='AL_type',
        legend = True,
        fill = False,
        color = ["green", "red"],
        figsize=(8, 6)
    )
    
    if limitsa: 
        plt.xlim(limitsa, limitsb)
        plt.xlim(limitsa, limitsb)
        axs[-1].set_xticks(range(7))
        axs[-1].set_xticklabels(np.linspace(limitsa, limitsb, 7))
        
    plt.xlabel(column_name)
    fig.supylabel("Query Method \n", x=0, y=0.5)
    plt.title('Bias Ratio: ' + str(ratios[ratio_level].round(2)), fontsize=20)
    
    plt.savefig("results/" + name + column_name + str(ratio_level)+ "_ridgelineClassificationPlot.pdf", bbox_inches='tight')
    
    plt.show()

