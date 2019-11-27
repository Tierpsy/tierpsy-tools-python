#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:15:22 2019

@author: em812
"""

def scale_feat(features,scaling='standardize',axis=None,norm=None):
    
    from tierpsytools.feature_filtering.scaling_class import scalingClass
    
    scaler = scalingClass(function=scaling,axis=axis,norm=norm)
    
    return scaler.fit_transform(features)
