#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
#from pandas import *
import xgboost as xgb
import pdb
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from FeatureEngineering import feature_processing
from sklearn import linear_model

trainX_file_paths = ['../data/train/12/12_data.csv','../data/train/23/23_data.csv','../data/train/29/29_data.csv']
trainY_file_paths = ['../data/train/12/12_failureInfo.csv','../data/train/23/23_failureInfo.csv','../data/train/29/29_failureInfo.csv']

trainX_concate1 = np.array([])
trainY_concate1 = np.array([])

trainX_concate2 = np.array([])
trainY_concate2 = np.array([])

trainXs, trainYs = [], []

for k in xrange(3):
    print 'Processing the %d th training dataset...' %(k)
    trainX_df = pd.read_csv(trainX_file_paths[k])
    trainY_df = pd.read_csv(trainY_file_paths[k])
    
    timeStamps = []
    for item in trainX_df.time:
        timeStamps.append(pd.Timestamp(item))

    trainX_df['time'] = timeStamps

    t3 = pd.Timestamp(trainY_df.iat[0,4])
    t2 = pd.Timestamp(trainY_df.iat[0,3])
    t1 = pd.Timestamp(trainY_df.iat[0,2])
    t0 = pd.Timestamp(trainY_df.iat[0,1])

    # labeling the timestamps
    positive_st, positive_ed, highRisk_st, highRisk_ed = -1, -1, -1, -1

    for i in xrange(trainX_df.shape[0]):
        if trainX_df.iat[i,0] >= t3:
            positive_st = i
            break

    for i in xrange(positive_st,trainX_df.shape[0]):
        if trainX_df.iat[i,0] >= t2:
            highRisk_st = i
            break

    for i in xrange(highRisk_st,trainX_df.shape[0]):
        if trainX_df.iat[i,0] >= t1:
            highRisk_ed = i-1
            break

    for i in xrange(highRisk_ed,trainX_df.shape[0]):
        if trainX_df.iat[i,0] >= t0:
            positive_ed = i
            break
   
    print 'positive_st,highRisk_st,highRisk_ed,positive_ed',positive_st,highRisk_st,highRisk_ed,positive_ed

    if positive_st != -1:
        
        phase1 = np.linspace(0,0.5*100,positive_st)
        phase2 = np.array([0.6*100 for i in xrange(highRisk_st-positive_st)])
        phase3 = np.array([0.8*100 for i in xrange(highRisk_ed-highRisk_st)])
        phase4 = np.linspace(0.8*100,100,trainX_df.shape[0]-highRisk_ed)
        
       
        '''
        phase1 = np.array([0.2*100 for i in xrange(positive_st)])
        phase2 = np.array([0.5*100 for i in xrange(highRisk_st-positive_st)])
        phase3 = np.array([0.8*100 for i in xrange(highRisk_ed-highRisk_st)])
        phase4 = np.array([1*100 for i in xrange(trainX_df.shape[0]-highRisk_ed)])
        '''

        trainY = np.concatenate((phase1,phase2,phase3,phase4), axis=0)
    else:
        trainY = np.array([10 for i in xrange(trainX_df.shape[0])])
    '''
    trainY = []
    for i in xrange(trainX_df.shape[0]):
        if i >= positive_st and i <= positive_ed:
            #if i >= highRisk_st:
            #    trainY.append(2)
            #else:
                trainY.append(1)
        else:
            trainY.append(0)
    trainY = np.array(trainY)
    '''
   
    # Feature engineering
    trainX = feature_processing(trainX_df)

    trainXs.append(trainX)
    trainYs.append(trainY)
    
    #print trainX,trainY
    #print trainX.shape, trainY.shape

    '''
    if k == 0:
        trainX_concate, trainY_concate = trainX, trainY
    else:
        trainX_concate = np.concatenate((trainX_concate,trainX), axis=0)
        trainY_concate = np.concatenate((trainY_concate,trainY), axis=0)
    print trainX_concate.shape, trainY_concate.shape
    '''

trainX_concate1 = np.concatenate((trainXs[0],trainXs[1]), axis=0)
trainY_concate1 = np.concatenate((trainYs[0],trainYs[1]), axis=0)


trainX_concate2 = np.concatenate((trainXs[0],trainXs[2]), axis=0)
trainY_concate2 = np.concatenate((trainYs[0],trainYs[2]), axis=0)

reg1 = linear_model.RidgeCV(alphas=np.linspace(0.1,0.9,9),cv=5)
#reg1 = linear_model.RidgeCV(alphas=[0.9],cv=5)
reg1.fit(trainX_concate1,trainY_concate1)
print 'reg1 best alpha:', reg1.alpha_
print reg1.score(trainX_concate1,trainY_concate1)

reg2 = linear_model.RidgeCV(alphas=np.linspace(0.1,0.9,9),cv=5)
#reg2 = linear_model.RidgeCV(alphas=[0.9],cv=5)
reg2.fit(trainX_concate2,trainY_concate2)
print 'reg2 best alpha:', reg2.alpha_

print reg2.score(trainX_concate2,trainY_concate2)
