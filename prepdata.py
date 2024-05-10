import tensorflow as tf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mstats


index={'LSTM':1,'Bi-LSTM':2,'Stacked-LSTM':3,'GRU':4,'1D-CNN':5}
def winsorize_outliers(series, lower_percentile=0.05, upper_percentile=0.95):
	lower_limit = series.quantile(lower_percentile)
	upper_limit = series.quantile(upper_percentile)
	winsorized_values = mstats.winsorize(series, limits=(lower_percentile, 1 - upper_percentile))  
	return winsorized_values

def windowed_df_to_date_X_y(windowed_dataframe):
	df_as_np = windowed_dataframe.to_numpy()
	dates = df_as_np[:, 0]
	middle_matrix = df_as_np[:, 2:]
	X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
	Y = df_as_np[:, 1:2]
	return dates, X.astype(np.float32), Y.astype(np.float32)

def preparedata(commodity,model):
	link="DATA/Backup/"+commodity+".csv"
	df=pd.read_csv(link)
	scaler = MinMaxScaler(feature_range=(0, 1))
	df['price'] = scaler.fit_transform(df['price'].values.reshape(-1,1))
	if commodity != 'rice' and commodity != 'wheat':
		df['price1'] = winsorize_outliers(df['price'], lower_percentile=0.05, upper_percentile=0.95)
		df.pop('price')
		df['price']=df.pop('price1')
	if model=='1D-CNN':
		shift=3
	elif commodity=='potato' or commodity=='onion' and model!='1D-CNN':
		shift=2
	elif commodity=='maize' and model=='Bi-LSTM':
		shift=3
	else:
		shift=1
	for i in range(shift,0,-1):
	    df['target-'+str(i)] = df['price'].shift(i)
	d=df[shift:]

	dates, X, y = windowed_df_to_date_X_y(d)
	dates.shape, X.shape, y.shape
	q_80 = int(len(dates) * .8)
	q_90 = int(len(dates) * .9)
	dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
	dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
	dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
	return dates_train,X_train,y_train,dates_val,X_val,y_val,dates_test,X_test,y_test,shift,scaler,dates,X,y




	

	