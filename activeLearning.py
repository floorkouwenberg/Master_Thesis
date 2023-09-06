# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:23:47 2023

@author: Floor
"""

from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances

'''
#Arguments
qs = Query strategy as name not function handle
x_train = dataframe used for training
y_train = initial labled set, unlabeled instances should have a nan value
y_train_true = all labels of the training set, this is used as oracle
n_cycles = number of cycles in the Active Learning cycles
clf = classifier as function handle
x_test = the test set of the fold

#Function
This function runs the entire active learning process and uses the sk-activeml 
package.
 
#Output
queried_indeces_per_cylce = A list containing the index values of the queried 
                            instances, this has the length of n_cycles
learning_curve = The learning curve of of the classifier on the trianing data. 
                 This is a list of length n_cycles containing accuracy scores
predictions = A list containing the predictions made by the classifier on the test set. 
              So this is a list of size n_cycles where each entry is a list of 
              length x_test
scores = the predicted probability for each test instance. 
'''
def activeLearning(qs, x_train, y_train, y_train_true, n_cycles, clf, x_test): 
    queried_indeces_per_cylce = []
    learning_curve = []
    predictions = []
    scores = []
        
    for c in range(n_cycles):
        print('#', end='') 
        
        #Each query strategy needs different things so:
        if qs.__name__ == 'RandomSampling': 
            query_idx = qs().query(X=x_train, y=y_train)[0]
            query_idx = x_train.iloc[query_idx].name
        elif qs.__name__ == 'MSAL': 
            query_idx = qs.query(x_train, y_train)[0]
        elif qs.__name__ == "ProbabilisticAL": 
            #PAL is running with rbf kernels and has a bandwith of gamma which is dependant on the dataset
            query_idx = qs(metric = "rbf", metric_dict = {'gamma': 0}).query(X=x_train, y=y_train, clf = clf)[0]
            query_idx = x_train.iloc[query_idx].name
        else: 
            query_idx = qs().query(X=x_train, y=y_train, clf = clf)[0]
            query_idx = x_train.iloc[query_idx].name
        
        y_train.loc[query_idx] = y_train_true.loc[query_idx]
        queried_indeces_per_cylce.append(x_train.loc[query_idx].name)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_train)
        learning_curve.append(accuracy_score(y_train_true, y_pred))
        predictions.append(clf.predict(x_test))
        scores.append(clf.predict_proba(x_test))
    
    print('\n ', end=' ')
    return queried_indeces_per_cylce, learning_curve, predictions, scores


'''
A class that implements Multi Standard Optimization Active Learning. As 
introducec by Wang, Zhang and Min in Active Learning Through Multi-Standard Optimization (2019)
'''
class MSAL: 
    
    def __init__(self, x_train):
        self.distance_matrix = self.calculateDistanceMatrix(x_train)
        self.dc = 0 #Dependant on the dataset
        self.threshold = self.calculateThreshold(x_train)
        self.summy = np.e**-(self.distance_matrix**2/(2*self.dc**2))
        self.lastQueriedIndex = None 
        self.__name__ = 'MSAL'
        
        
    def query(self, x_train, y_train): 
        "Returns the best instance to be queried"
        unlabeled_indeces = y_train[y_train.isna()].index.values
        
        informativeness = self.msalInformativeness(x_train, y_train)
        representativeness = self.msalRepresentativeness(x_train, y_train)
        
        sort = np.argsort(informativeness*representativeness)
        
        answer_not_found = True
        i = 1    
        while answer_not_found: 
            if i > len(unlabeled_indeces): 
                print("No value meets the threshold, so will pick the hihgest number")
                i = 1
                break
            
            if self.msalDifferenceEnough(x_train, x_train.loc[unlabeled_indeces[sort[-i]]]): 
                answer_not_found = False
            else: 
                i += 1
                
        self.lastQueriedIndex = unlabeled_indeces[sort[-i]]
        return [unlabeled_indeces[sort[-i]]] 
    
    
    def msalDifferenceEnough(self, x_train, instance):      
        "Checks whether the difference between two instances meets the threshold"
        if not self.lastQueriedIndex: 
            return True 
        else:
            return (euclidean_distances(x_train.loc[self.lastQueriedIndex].values.reshape(1,-1), instance.values.reshape(1,-1))[0][0]>self.threshold)
            
  
    def msalInformativeness(self, x_train, y_train): 
        "Returns the informativeness of each instance in the unlabeled train set"
        labeled_indeces = np.where(~np.isnan(y_train.to_numpy()))
        y_L = y_train.to_numpy()[labeled_indeces]
        x_L = x_train.to_numpy()[labeled_indeces]
        x_U = x_train.to_numpy()[np.where(np.isnan(y_train))]
        informativeness = np.zeros(len(x_U))
        
        if len(np.unique(y_L))<2: 
            return None
        else: 
            #Uses a soft-max regession to find the entropy of each instance
            logreg = LogisticRegression(max_iter = 1000, multi_class = 'multinomial')
            logreg.fit(x_L, y_L)
            probabilities = logreg.predict_proba(x_U)
            for i in range(len(informativeness)): 
                informativeness[i] = entropy(probabilities[i,:])
                
        return informativeness


    def msalRepresentativeness(self, x_train, y_train): 
        "Returns the representativeness of the unlabeled train set using gaussian kernel densitiy estimation"
        x_U = x_train.to_numpy()[np.where(np.isnan(y_train))]
        kde = KernelDensity(kernel='gaussian', bandwidth=self.dc).fit(x_U)
        
        #.fit returns the loglikelihoods of the instances, so have to convert to likelihood
        representativeness = np.e**kde.score_samples(x_U)
            
        return representativeness
                   
            
    def calculateThreshold(self, x_train): 
        "Calculates the threshold during setup"
        gamma = 0.1 #Can be changed
        n = len(x_train)
        summed_distances = self.distance_matrix.sum(axis=1)
        beta = gamma/n * max(summed_distances)
        
        return beta


    def calculateDistanceMatrix(self, x_train): 
        "Calculates the distance matrix during setup"
        x_train = x_train.to_numpy()
        distance_matrix = euclidean_distances(x_train, x_train)
        return distance_matrix
        