import math
import numpy as np
import pandas as pd
import os

class DataLoader():
	"""A class for loading and transforming data for the lstm model"""

	def __init__(self, filename, filepath, split, cols):
		fileList = []
		for i in os.walk(filepath):
			if(filename in i[2]):
				fileList.append(os.path.join(i[0] ,filename))
		fileList = sorted(fileList,key = lambda a:int(os.path.split(os.path.split(a)[0])[1]))
		data_all = []
		data_all = np.array(data_all)
		data_all = data_all.reshape(0,2)
		for i in fileList:
			dataframe = pd.read_csv(i,header=None)
			data_train = dataframe.get([19,10]).values
			data_train_new = []
			for i in range(int(len(data_train)/120)):
				data_train_new.append(data_train[120*i])

			data_train_new = np.array(data_train_new)

			for i in range(len(data_train_new)-1,0,-1):
				data_train_new[i] = data_train_new[i] - data_train_new[i-1]
				if(data_train_new[i,1] != 0):
					data_train_new[i,0] = data_train_new[i,0] / data_train_new[i,1]
			data_train_new = np.delete(data_train_new,0,axis=0)
			data_all = np.concatenate((data_all,data_train_new),axis=0)
		i_split = int(data_all.shape[0] * split)
		self.data_train = data_all[:i_split]
		self.data_test = data_all[i_split:]
		print(self.data_train)
		# dataframe = pd.read_csv(filename,header=None)
		# i_split = int(len(dataframe) * split)
		# self.data_train = dataframe.get(cols).values[:i_split]
		# self.data_test  = dataframe.get(cols).values[i_split:]
		# self.data_train = self.data_train.astype('float64')
		# self.data_test = self.data_test.astype('float64')
		# self.data_train[:,0] = self.data_train[:,0]
		# self.data_test[:,0]  = self.data_test[:,0]
		# print(self.data_train)
		self.len_train  = len(self.data_train)
		self.len_test   = len(self.data_test)
		self.len_train_windows = None

	def get_test_data(self, seq_len, normalise):
		'''
		Create x, y test data windows
		Warning: batch method, not generative, make sure you have enough memory to
		load data, otherwise reduce size of the training split.
		'''
		data_windows = []
		for i in range(self.len_test - seq_len):		
			data_windows.append(self.data_test[i:i+seq_len])

		data_windows = np.array(data_windows).astype(float)
		data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
		x = data_windows[:, :-1]
		y = data_windows[:, -1, [0]]
		return x,y

	def get_train_data(self, seq_len, normalise):
		'''
		Create x, y train data windows
		Warning: batch method, not generative, make sure you have enough memory to
		load data, otherwise use generate_training_window() method.
		'''
		data_x = []
		data_y = []
		for i in range(self.len_train - seq_len):
			x, y = self._next_window(i, seq_len, normalise)
			data_x.append(x)
			data_y.append(y)
		return np.array(data_x), np.array(data_y)

	def generate_train_batch(self, seq_len, batch_size, normalise):
		'''Yield a generator of training data from filename on given list of cols split for train/test'''
		i = 0
		while i < (self.len_train - seq_len):
			x_batch = []
			y_batch = []
			for b in range(batch_size):
				if i >= (self.len_train - seq_len):
					# stop-condition for a smaller final batch if data doesn't divide evenly
					yield np.array(x_batch), np.array(y_batch)
				x, y = self._next_window(i, seq_len, normalise)
				x_batch.append(x)
				y_batch.append(y)
				i += 1
			yield np.array(x_batch), np.array(y_batch)

	def _next_window(self, i, seq_len, normalise):
		'''Generates the next data window from the given index location i'''
		window = self.data_train[i:i+seq_len]
		window = self.normalise_windows(window, single_window=True)[0] if normalise else window
		x = window[:-1]
		y = window[-1, [0]]
		return x, y

	def normalise_windows(self, window_data, single_window=False):
		'''Normalise window with a base value of zero'''
		normalised_data = []
		window_data = [window_data] if single_window else window_data
		for window in window_data:
			normalised_window = []
			for col_i in range(window.shape[1]):
				normalised_col = [((float(p) / float(window[0, col_i]))  - 1)  for p in window[:, col_i]]
				normalised_window.append(normalised_col)
			normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format				
			normalised_data.append(normalised_window)
		return np.array(normalised_data)