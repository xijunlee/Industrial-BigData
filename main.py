#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
#from pandas import *
import xgboost as xgb
import pdb

trainX_file_paths = ['../data/train/12/12_data.csv','../data/train/23/23_data.csv','../data/train/29/29_data.csv']
trainY_file_paths = ['../data/train/12/12_failureInfo.csv','../data/train/23/23_failureInfo.csv','../data/train/29/29_failureInfo.csv']

testX_file_path = '../data/test/26/26_data.csv'

testX_pd = pd.read_csv(testX_file_path)

trainX_concate = np.array([])
trainY_concate = np.array([])

for k in xrange(3):
    print 'processing the %d th training dataset...' %(k)
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

    trainY = []
    for i in xrange(trainX_df.shape[0]):
        if i >= positive_st and i <= positive_ed:
            trainY.append(1)
        else:
            trainY.append(0)
    pdb.set_trace()
    trainY = np.array(trainY)

    trainX_df = trainX_df.drop(['time'],axis=1)
    trainX = trainX_df.values
    if k == 0:
        trainX_concate, trainY_concate = trainX, trainY
    else:
        trainX_concate = np.concatenate((trainX_concate,trainX), axis=0)
        trainY_concate = np.concatenate((trainY_concate,trainY), axis=0)
    print trainX_concate.shape, trainY_concate.shape


'''
dtrain = xgb.DMatrix(trainX_concate,label=trainY_concate)
max_depth = 5
eta = 1
nEstimators = 300

# cross validation
param = {'max_depth':max_depth, 'eta':eta, 'silent':1, 'objective':'binary:logistic','n_estimators':nEstimators }
num_boost_round = 10
n_fold = 5
res = xgb.cv(param,dtrain,num_boost_round,n_fold,metrics={'error'},seed=0,
            callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                      xgb.callback.early_stop(3)])
print res
'''

clf = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 500,
 max_depth= 5,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(trainX_concate, trainY_concate)

testX = testX_pd.drop(['time'], axis=1).values

predictions = clf.predict(testX)

# Generate Submission File 
StackingSubmission = pd.DataFrame({'time': testX_pd['time']],
                            'predictions': predictions})
StackingSubmission.to_csv("submission.csv", index=False)
print 'results saved to .csv successfully!'
