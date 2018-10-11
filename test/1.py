#!/usr/bin/env python

import pandas as pd
import numpy as np
import os

fileList = []
print(os.getcwd())
os.chdir('test/')
for i in os.listdir(os.getcwd()):
    #if(os.path.isfile(filepath)):
    print(i)
    if(os.path.splitext(i)[1] ==".csv"):
        fileList.append(i)
print(fileList)
data_train_all = []
data_train_all = np.array(data_train_all)
data_train_all = data_train_all.reshape(0,2)
print(data_train_all)
for i in fileList:
    dataframe = pd.read_csv(i,header=None)

    data_train = dataframe.get([10,19]).values
    print(type(data_train))
    print(type(dataframe))
    data_train_new = []
    for i in range(int(len(data_train)/600)):
        data_train_new.append(data_train[600*i])

    data_train_new = np.array(data_train_new)

    for i in range(len(data_train_new)-1,0,-1):
        data_train_new[i] = data_train_new[i] - data_train_new[i-1]
        if(data_train_new[i,0] != 0):
            data_train_new[i,1] = data_train_new[i,1] / data_train_new[i,0]
    data_train_new = np.delete(data_train_new,0,axis=0)
    print(data_train_new.shape)
    data_train_all = np.concatenate((data_train_all,data_train_new),axis=0)
print(data_train_all.shape)

