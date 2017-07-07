#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np
#from pandas import *
import pdb

from sklearn.preprocessing import normalize

def feature_processing(df):
    #df['f1']=(df['wind_speed']+df['generator_speed']+df['power'])/3.0
    df['wind_direction_avg'] = (df['wind_direction']+df['wind_direction_mean'])/2.0
    df['pitch_angle_avg'] = (df['pitch1_angle']+df['pitch2_angle']+df['pitch3_angle'])/3.0
    df['pitch_speed_avg'] = (df['pitch1_speed']+df['pitch2_speed']+df['pitch3_speed'])/3.0
    df['pitch_moto_tmp_avg'] = (df['pitch1_moto_tmp']+df['pitch2_moto_tmp']+df['pitch3_moto_tmp'])/3.0
    df['tmp_avg'] = (df['environment_tmp']+df['int_tmp'])/2.0
    df['pitch_ng5_tmp_avg'] = (df['pitch1_ng5_tmp']+df['pitch2_ng5_tmp']+df['pitch3_ng5_tmp'])/3.0
    drop_labels=['wind_direction','wind_direction_mean','pitch1_angle','pitch2_angle','pitch3_angle','pitch1_speed','pitch2_speed','pitch3_speed',
    'pitch1_moto_tmp','pitch2_moto_tmp','pitch3_moto_tmp','environment_tmp','int_tmp','pitch1_ng5_tmp','pitch2_ng5_tmp','pitch3_ng5_tmp','time','group']
    df = df.drop(drop_labels,axis=1)
    data = df.values
    data = normalize(data,norm='l2',axis=1)
    return data

def feature_processing_df(df):
    #df['f1']=(df['wind_speed']+df['generator_speed']+df['power'])/3.0
    df['wind_direction_avg'] = (df['wind_direction']+df['wind_direction_mean'])/2.0
    df['pitch_angle_avg'] = (df['pitch1_angle']+df['pitch2_angle']+df['pitch3_angle'])/3.0
    df['pitch_speed_avg'] = (df['pitch1_speed']+df['pitch2_speed']+df['pitch3_speed'])/3.0
    df['pitch_moto_tmp_avg'] = (df['pitch1_moto_tmp']+df['pitch2_moto_tmp']+df['pitch3_moto_tmp'])/3.0
    df['tmp_avg'] = (df['environment_tmp']+df['int_tmp'])/2.0
    df['pitch_ng5_tmp_avg'] = (df['pitch1_ng5_tmp']+df['pitch2_ng5_tmp']+df['pitch3_ng5_tmp'])/3.0
    drop_labels=['wind_direction','wind_direction_mean','pitch1_angle','pitch2_angle','pitch3_angle','pitch1_speed','pitch2_speed','pitch3_speed',
    'pitch1_moto_tmp','pitch2_moto_tmp','pitch3_moto_tmp','environment_tmp','int_tmp','pitch1_ng5_tmp','pitch2_ng5_tmp','pitch3_ng5_tmp','group','time']
    df = df.drop(drop_labels,axis=1)
    return df

if __name__ == '__main__':
    file_paths = ['../data/train/12/12_data.csv','../data/train/23/23_data.csv','../data/train/29/29_data.csv']
    save_paths = ['12_feature_processing_result.csv','23_feature_processing_result.csv','29_feature_processing_result.csv']
    for i in xrange(3):
        data_df = pd.read_csv(file_paths[i])
        data_df = feature_processing_df(data_df)
        print data_df.shape
        data_df.to_csv(save_paths[i],index=False)
        print save_paths[i] + ' has been saved!'




