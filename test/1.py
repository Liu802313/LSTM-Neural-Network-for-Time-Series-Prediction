#!/usr/bin/env python

import pandas as pd
import numpy as np
import os

fileList = []
print(os.getcwd())
os.chdir('test/')
a = os.getcwd()
for i in os.walk(os.getcwd()):
    if('cu1810.csv' in i[2]):
        fileList.append(i[0] + '/cu1810.csv')
fileList = sorted(fileList,key = lambda a:int(os.path.split(os.path.split(a)[0])[1]))
data_train_all = []
data_train_all = np.array(data_train_all)
data_train_all = data_train_all.reshape(0,2)
print(data_train_all)
for i in fileList:
    dataframe = pd.read_csv(i,header=None)
    data_train = dataframe.get([19,10]).values
    data_train_new = []
    for i in range(int(len(data_train)/600)):
        data_train_new.append(data_train[600*i])

    data_train_new = np.array(data_train_new)

    for i in range(len(data_train_new)-1,0,-1):
        data_train_new[i] = data_train_new[i] - data_train_new[i-1]
        if(data_train_new[i,1] != 0):
            data_train_new[i,0] = data_train_new[i,0] / data_train_new[i,1]
    data_train_new = np.delete(data_train_new,0,axis=0)
    data_train_all = np.concatenate((data_train_all,data_train_new),axis=0)
print(data_train_all)

