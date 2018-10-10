import csv
import pandas as pd
import numpy as np

dataframe = pd.read_csv('cu1809.csv',header=None)
data_sinewave = dataframe.get([18,15,17])
data_sinewave = data_sinewave.astype('float64')
print(data_sinewave)
data_print = np.array(data_sinewave[18])

print(np.mean(data_print))