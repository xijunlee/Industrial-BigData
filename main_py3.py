#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
#from pandas import *
#import xgboost as xgb
import pdb
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from FeatureEngineering import feature_processing
from sklearn import linear_model

from keras.models import Sequential
from keras.layers import Dense, Activation

trainX_file_paths = ['../data/train/12/12_data.csv','../data/train/23/23_data.csv','../data/train/29/29_data.csv']
trainY_file_paths = ['../data/train/12/12_failureInfo.csv','../data/train/23/23_failureInfo.csv','../data/train/29/29_failureInfo.csv']

trainX_concate1 = np.array([])
trainY_concate1 = np.array([])

trainX_concate2 = np.array([])
trainY_concate2 = np.array([])

trainXs, trainYs = [], []

for k in range(3):
    print ('Processing the %d th training dataset...'%(k))
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

    for i in range(trainX_df.shape[0]):
        if trainX_df.iat[i,0] >= t3:
            positive_st = i
            break

    for i in range(positive_st,trainX_df.shape[0]):
        if trainX_df.iat[i,0] >= t2:
            highRisk_st = i
            break

    for i in range(highRisk_st,trainX_df.shape[0]):
        if trainX_df.iat[i,0] >= t1:
            highRisk_ed = i-1
            break

    for i in range(highRisk_ed,trainX_df.shape[0]):
        if trainX_df.iat[i,0] >= t0:
            positive_ed = i
            break
   
    print ('positive_st,highRisk_st,highRisk_ed,positive_ed',positive_st,highRisk_st,highRisk_ed,positive_ed)

    if positive_st != -1:
        
        phase1 = np.linspace(0,0.3*100,positive_st)
        #phase2 = np.array([0.6*100 for i in range(highRisk_st-positive_st)])
        phase2 = np.linspace(0.6*100, 0.8*100, highRisk_st-positive_st)
        phase3 = np.array([0.8*100 for i in range(highRisk_ed-highRisk_st)])
        phase4 = np.linspace(0.8*100,100,trainX_df.shape[0]-highRisk_ed)
        
       
        '''
        phase1 = np.array([0.2*100 for i in range(positive_st)])
        phase2 = np.array([0.5*100 for i in range(highRisk_st-positive_st)])
        phase3 = np.array([0.8*100 for i in range(highRisk_ed-highRisk_st)])
        phase4 = np.array([1*100 for i in range(trainX_df.shape[0]-highRisk_ed)])
        '''

        trainY = np.concatenate((phase1,phase2,phase3,phase4), axis=0)
    else:
        trainY = np.array([10 for i in range(trainX_df.shape[0])])
    '''
    trainY = []
    for i in range(trainX_df.shape[0]):
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

inputDim = trainX_concate.shape[1]

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

'''
print 'PCA processing ...'
nComponent = 10
pca = PCA(n_components=nComponent)
pca.fit(trainX_concate1)
trainX_concate1 = pca.transform(trainX_concate1)
pca.fit(trainX_concate2)
trainX_concate2 = pca.transform(trainX_concate2)
'''

'''
print 'Training 1st xgbclassfier ...'
clf1 = xgb.XGBClassifier(
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
 scale_pos_weight=1).fit(trainX_concate1, trainY_concate1)

print 'Training 2nd xgbclassfier ...'
clf2 = xgb.XGBClassifier(
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
 scale_pos_weight=1).fit(trainX_concate2, trainY_concate2)
'''

'''
print 'Training 1st linear ridge regression clf...'
clf1 = linear_model.Ridge(alpha=.9).fit(trainX_concate1,trainY_concate1)

print 'Training 2nd linear ridge regression clf...'
clf2 = linear_model.Ridge(alpha=.1).fit(trainX_concate2,trainY_concate2)
'''

nn_model1 = Sequential([
    Dense(10,input_dim = inputDim),
    Activation('relu'),
    Dense(8),
    Activation('relu'),
    Dense(3),
    Activation('relu')
])
nn_model1.summary()
nn_model1.compile(opimizer='rmsprop', loss='mse')

print ('Training 1st neural network ...')
nn_model1.fit(trainX_concate1,trainY_concate1,epochs=10,batch_size=32,verbose=1)

nn_model2 = Sequential([
    Dense(10,input_dim = inputDim),
    Activation('relu'),
    Dense(8),
    Activation('relu'),
    Dense(3),
    Activation('relu')
])
nn_model2.summary()
nn_model2.compile(opimizer='rmsprop', loss='mse')

print ('Training 2nd neural network ...')
nn_model2.fit(trainX_concate2,trainY_concate2,epochs=10,batch_size=32,verbose=1)

testX_file_paths = ['../data/test/26/26_data.csv','../data/test/33/33_data.csv']


print ('The 1st classifier predicting ...')
for k in range(len(trainX_file_paths)):
    testX_pd = pd.read_csv(trainX_file_paths[k])

    # Feature processing
    testX = feature_processing(testX_pd)

    print ('Predicting the %d th test dataset ...'%k)
    
    '''
    pca = PCA(n_components=nComponent)
    pca.fit(testX)
    testX = pca.transform(testX)
    '''
    #predictions = clf1.predict(testX)
    predictions = nn_model1.predict(testX)

    print ('Generating submission file of the %d test dataset ...'%k)
    # Generate Submission File 
    
    st, ed = -1, -1
    for i in range(len(predictions)):
        if predictions[i] >= 50.0:
            st = i
            break
    for i in range(len(predictions)-1,-1,-1):
        if predictions[i] <= 80.0:
            ed = i
            break
    t2, t1 = [st+1], [ed+1]
    
    '''
    for i in range(len(predictions)):
        if predictions[i]<80.0:
            predictions[i] = 0
    '''

    '''
    f = [0 for i in range(len(predictions))]
    f[0] += predictions[0]
    for i in range(1,len(predictions)):
        f[i] = f[i-1] + predictions[i]

    L, op, t1, t2 = len(predictions), -1, -1, -1
    for i in range(L):
        for j in range(i+1,L):
            l = j -i
            c = (f[j] - f[i])*1.0
            rho = c/l
from keras.layers.core as core
            l1 = l*100.0/L
            if op < rho + l1:
                t1, t2 = j, i

    t1, t2 = [t1], [t2]
    '''
    mid_result = pd.DataFrame({'predictions':predictions})
    mid_result.to_csv('train_'+str(k)+'mid_result1.csv',index=True)
    sub = pd.DataFrame({'t1':t1,'t2':t2})
    save_path = 'train_'+str(k)+'_submission1.csv'
    #sub.to_csv(save_path, index=False)
    print (save_path + ' result 1 has been saved successfully!')


print ('The 2nd classifier predicting ...')
for k in range(len(trainX_file_paths)):
    testX_pd = pd.read_csv(trainX_file_paths[k])

    # Feature processing
    testX = feature_processing(testX_pd)

    print ('Predicting the %d th test dataset ...'%k)
    
    '''
    pca = PCA(n_components=nComponent)
    pca.fit(testX)
    testX = pca.transform(testX)
    '''
    #predictions = clf2.predict(testX)
    predictions = nn_model2.predict(testX)

    print ('Generating submission file of the %d test dataset ...'%k)
    # Generate Submission File 
    
    st, ed = -1, -1
    for i in range(len(predictions)):
        if predictions[i] >= 50.0:
            st = i
            break
    for i in range(len(predictions)-1,-1,-1):
        if predictions[i] <= 80.0:
            ed = i
            break
    t2, t1 = [st+1], [ed+1]
    
    ''' 
    for i in range(len(predictions)):
        if predictions[i]<80.0:
            predictions[i] = 0
    '''

    '''
    f = [0 for i in range(len(predictions))]
    f[0] += predictions[0]
    for i in range(1,len(predictions)):
        f[i] = f[i-1] + predictions[i]

    L, op, t1, t2 = len(predictions), -1, -1, -1
    for i in range(L):
        for j in range(i+1,L):
            l = j -i
            c = (f[j] - f[i])*1.0
            rho = c/l
            l1 = l*100.0/L
            if op < rho + l1:
                t1, t2 = j, i

    t1, t2 = [t1], [t2]
    '''
    mid_result = pd.DataFrame({'predictions':predictions})
    mid_result.to_csv('train_'+str(k)+'mid_result2.csv',index=True)
    sub = pd.DataFrame({'t1':t1,'t2':t2})
    save_path = 'train_'+str(k)+'_submission2.csv'
    #sub.to_csv(save_path, index=False)
    print (save_path + ' result 2 has been saved successfully!')
