#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:39:29 2019

@author: em812
"""

def k_significant_feat(feat,y_class,k=5,featNames=None,plotgroups=None,grouplabels=None,method='f_classif',figsize=None,title=None,savefig=None,close_after_plotting=False):
    """
    Finds the k most significant features in the feature matrix, based on the classification in y_class
    and plots their distribution.
    Uses univariate feature selection.
    param:
        featMat: feature matrix - np array-like
        y_class: vector with the class of each sample
    return:
        k most significant features
        save plot
    """
    from tierpsytools.feature_processing.scaling_class import scalingClass
    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import SelectKBest,chi2,f_classif
    import matplotlib.pyplot as plt
    
    if isinstance(feat,np.ndarray):
        feat = pd.DataFrame(feat,columns=featNames)
           
    # Find most significant features
    if method=='f_classif':
        scaler = scalingClass(scaling='standardize')
        feat_scaled = scaler.fit_transform(feat)
        skb = SelectKBest(score_func=f_classif,k=k)
    elif method=='chi2':
        scaler = scalingClass(scaling='minmax_scale')
        feat_scaled = scaler.fit_transform(feat)
        skb = SelectKBest(score_func=chi2,k=k)
    
    scores = skb.fit(feat_scaled,y_class).scores_
    
    # Re-order feature matrix based on feature significance
    feat = feat.iloc[:,np.argsort(scores)]
    
    # Plot a boxplot for each feature, showing its distribution in each class
    for ft in feat.columns[:k]:
        plt.figure(figsize=figsize)
        plt.title(ft)
        plt.boxplot([feat.loc[y_class==cl,ft] for cl in np.unique(y_class)])
        if savefig:
            plt.savefig(savefig)
        if close_after_plotting:
            plt.close()

    return feat.columns[:k]
        
   

def top_feat_in_PCs(X,pc=1,k='auto'):
    """
    Runs PCA and gives the top k contributing features for the specified principal component (by default the first component).
    
    """
    import pandas as pd
    from sklearn.decomposition import PCA
    
    if isinstance(X,np.ndarray):
        X = pd.DataFrame(X)
    Xscaled = X.loc[:,X.std()!=0].copy()
    Xscaled = ( Xscaled - Xscaled.mean() ) / Xscaled.std()
    
    pca = PCA(n_components=pc)
    pca.fit(Xscaled)
    
    ## pca.components_ --> each row contains the feature coefficients for one component
    component = pca.components_[pc-1,:]
    
    sortid = np.flip(np.argsort(np.abs(component)))
    
    if k=='auto':
        from kneed import KneeLocator
        kn = KneeLocator(np.arange(sortid.shape[0]), np.abs(component[sortid]), curve='convex', direction='decreasing')
        k = kn.knee

    k_feat = list(Xscaled.columns[sortid[:k]])
    
    return k_feat,component[sortid[:k]]

    
if __name__=="__main__":    
    
    from sklearn.decomposition import PCA
    import numpy as np
    
    X = np.random.rand(20,5)
    
    Xscal = (X-np.mean(X,axis=0))/np.std(X,axis=0)
    
    pca = PCA(n_components=4)
    pca.fit(Xscal)
        
    top_feat,coefficients = top_feat_in_PCs(X,pc=1,k=3)
    
    print(pca.components_[0,:])
    
