#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
#from pandas import *
import xgboost as xgb
import pdb
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

trainX_file_paths = ['../data/train/12/12_data.csv','../data/train/23/23_data.csv','../data/train/29/29_data.csv']
trainY_file_paths = ['../data/train/12/12_failureInfo.csv','../data/train/23/23_failureInfo.csv','../data/train/29/29_failureInfo.csv']

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
    
    if positive_st != -1:
        
        '''
        phase1 = np.linspace(0,0.5*100,positive_st)
        phase2 = np.array([0.5*100 for i in xrange(highRisk_st-positive_st)])
        phase3 = np.array([0.8*100 for i in xrange(highRisk_ed-highRisk_st)])
        phase4 = np.linspace(0.8*100,1*100,trainX_df.shape[0]-highRisk_ed)
        '''
        
        phase1 = np.array([0.2*100 for i in xrange(positive_st)])
        phase2 = np.array([0.5*100 for i in xrange(highRisk_st-positive_st)])
        phase3 = np.array([0.8*100 for i in xrange(highRisk_ed-highRisk_st)])
        phase4 = np.array([1*100 for i in xrange(trainX_df.shape[0]-highRisk_ed)])
        

        trainY = np.concatenate((phase1,phase2,phase3,phase4), axis=0)
    else:
        trainY = np.array([40 for i in xrange(trainX_df.shape[0])])
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
    
    trainX_df = trainX_df.drop(['time'],axis=1)
    trainX = trainX_df.values
    trainX = normalize(trainX,norm='l2',axis=1)
    
    print trainX,trainY
    print trainX.shape, trainY.shape

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

print 'training xgbclassfier ...'

nComponent = 15
pca = PCA(n_components=nComponent)
pca.fit(trainX_concate)
trainX_concate = pca.transform(trainX_concate)

clf = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 500,
 max_depth= 5,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(trainX_concate, trainY_concate)

testX_file_paths = ['../data/test/26/26_data.csv','../data/test/33/33_data.csv']


for k in xrange(len(testX_file_paths)):
    testX_pd = pd.read_csv(testX_file_paths[k])

    testX = testX_pd.drop(['time'], axis=1).values

    print 'predicting the %d th test dataset ...'%k
    
    pca = PCA(n_components=nComponent)
    pca.fit(testX)
    testX = pca.transform(testX)
    predictions = clf.predict(testX)

    print 'generating submission file of the %d test dataset ...'%k
    # Generate Submission File 
    st, ed = -1, -1
    for i in xrange(len(predictions)):
        if predictions[i] >= 50.0:
            st = i
            break
    for i in xrange(len(predictions)-1,-1,-1):
        if predictions[i] <= 80.0:
            ed = i
            break
    t2, t1 = [st+1], [ed+1]
    mid_result = pd.DataFrame({'predictions':predictions})
    mid_result.to_csv('test_'+str(k)+'mid_result.csv',index=True)
    sub = pd.DataFrame({'t1':t1,'t2':t2})
    save_path = 'test_'+str(k)+'_submission.csv'
    sub.to_csv(save_path, index=False)
    print save_path + 'result has been saved successfully!'
