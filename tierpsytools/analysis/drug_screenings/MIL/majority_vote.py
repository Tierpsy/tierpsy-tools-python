#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:01:33 2020

@author: em812
"""
import numpy as np

def get_two_most_likely_accuracy(ytest, ytest_pred_two):

    check=[]
    for i,y in enumerate(ytest):
        if y in ytest_pred_two[i]:
            check.append(True)
    acc = np.sum(check)/len(check)
    return acc

def get_two_most_likely(Xtest, estimator):

    ytest_pred_two = np.empty([Xtest.shape[0],2])

    ytest_pred_proba = estimator.predict_proba(Xtest)
    indx = np.flip(np.argsort(ytest_pred_proba,axis=1),axis=1)
    classes = estimator.classes_
    for cmpd in range(Xtest.shape[0]):
        ytest_pred_two[cmpd] = classes[indx[cmpd]][0:2]

    return ytest_pred_two

def get_seen_compounds(Xtest, ytest, ytrain):
    """
    Keep only test set cpmpounds that belong to MOAs that were seen in ytrain
    """
    import pandas as pd
    if isinstance(ytest,list):
        ytest=np.array(ytest)

    if isinstance(Xtest,pd.DataFrame):
        Xtest=Xtest.values
    seen = [i for i,y in enumerate(ytest) if y in ytrain]
    ytest = ytest[seen]
    Xtest = Xtest[seen,:]

    return Xtest,ytest,seen

def majority_vote_CV(
        feat, moa_group, drug_names, estimator, splitter, scale_function=None):
    from tierpsytools.analysis.classification_tools import get_fscore
    from scaling_class import scalingClass
    from sklearn.metrics import accuracy_score
    ## Majority vote
    #---------------
    # Make feature matrix with bags of sample points per compound
    X=[]
    y=[]
    groups=[]
    for (cmpd_name,cmpd_feat) in feat.groupby(by=drug_names):
        #print(cmpd_name,cmpd_feat.shape[0])
            X.append(cmpd_feat.values)
            y.append(moa_group[drug_names==cmpd_name])
            groups.append(drug_names[drug_names==cmpd_name])
    X = np.array(X)
    y=np.array(y)

    y_pred = np.array([[np.nan for i in range(y[j].shape[0])] for j in range(y.shape[0])])
    for train_index, test_index in splitter.split(X, y):
        X_train = np.concatenate(X[train_index])
        X_test = np.concatenate(X[test_index])
        y_train = np.concatenate(y[train_index])
        #y_test = np.concatenate(y[test_index])

        # Normalize
        scaler = scalingClass(function=scale_function)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train classifier
        estimator.fit(X_train,y_train)

        # Predict
        y_pred[np.squeeze(test_index)] = estimator.predict(X_test)

    y = np.concatenate(y)
    y_pred = np.concatenate(y_pred)
    groups = np.concatenate(groups)

    y_maj = get_majority_vote(y_pred,groups)
    unq_groups,unq_index = np.unique(groups,return_index=True)
    y_unq = y[unq_index]
    y_maj = y_maj[unq_index]
    maj_accuracy = accuracy_score(y_unq,y_maj)
    accuracy = accuracy_score(y,y_pred)
    f_score = get_fscore(y, y_pred, return_precision_recall=False)
    maj_f_score = get_fscore(y_unq, y_maj, return_precision_recall=False)

    return maj_accuracy,maj_f_score,y_maj,y_unq,accuracy,f_score,y_pred

def get_majority_vote(y_pred,groups,probas=None,labels=None):
    """
    Estimate the classification accuracy when groups of data points are classified together based on a majority vote.
    param:
        y_true: the true class labels of the samples (array size n_samples)
        y_pred: the predicted class labels from the classfier (array size n_samples)
        groups: an array defining the groups of data points (array size n_samples)
    """
    from collections import Counter

    y_maj = np.empty_like(y_pred)

    for grp in np.unique(groups):

        c = Counter(y_pred[groups==grp])

        #value,count = c.most_common()[0]
        counts=np.array([c.most_common()[i][1] for i in range(len(c.most_common()))])

        if len(counts[counts==counts[0]])==1:
            value,count = c.most_common()[0]
            y_maj[groups==grp] = value
        # if more than one labels have the same number of votes
        else:   #len(c.most_common())>1 and c.most_common()[0][1]==c.most_common()[1][1]:
            if probas is None:
                value,count = c.most_common()[0]
                y_maj[groups==grp] = value
                print('Warning: the samples of compound {} are classified in more than one classes with the same frequency.'.format(grp))
            else:
                assert len(labels)==probas.shape[1]
                values = np.array([c.most_common()[i][0] for i in range(len(c.most_common()))])
                equal_classes = values[counts==counts[0]]
                probas_of_equal_classes = []
                for iclass in equal_classes:
                    probas_of_equal_classes.append(np.mean(probas[groups==grp,labels==iclass]))
                most_likely_class = equal_classes[np.argmax(probas_of_equal_classes)]
                y_maj[groups==grp] = most_likely_class
    return y_maj

def get_two_most_likely_majority_vote(y_pred,groups):
    """
    Estimate the classification accuracy when groups of data points are classified together based on a majority vote.
    param:
        y_true: the true class labels of the samples (array size n_samples)
        y_pred: the predicted class labels from the classfier (array size n_samples)
        groups: an array defining the groups of data points (array size n_samples)
    """
    from collections import Counter

    y_maj = np.empty((y_pred.shape[0],2))

    for grp in np.unique(groups):

        c = Counter(y_pred[groups==grp])

        if len(c.most_common())>1:
            for rnk in [0,1]:
                value,count = c.most_common()[rnk]
                y_maj[groups==grp,rnk] = value
        else:
            value,count = c.most_common()[0]
            y_maj[groups==grp,:] = [value,value]

        # if more than one labels have the same number of votes
        if len(c.most_common())>1 and c.most_common()[0][1]==c.most_common()[1][1]:
            print('Warning: the samples of compound {} are classified in more than one classes with the same frequency.'.format(grp))

    return y_maj
