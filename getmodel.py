import tensorflow as tf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def generateModel(model,best_params,X_train,y_train,X_val,y_val,X_test,y_test,shift,scaler):
	if model=='LSTM':
		best_lstm_units = best_params['lstm_units']
		best_dense_units = best_params['dense_units']
		best_learning_rate = best_params['learning_rate']
		m1 = Sequential([
			layers.Input((shift, 1)),
			layers.LSTM(best_lstm_units),
			layers.Dense(32, activation='relu'),
			layers.Dense(1)
		])
	if model=='Bi-LSTM':
		best_lstm_units = best_params['lstm_units']
		best_dense_units = best_params['dense_units']
		best_learning_rate = best_params['learning_rate']
		m1 = Sequential([
			layers.Input((shift, 1)),
			layers.LSTM(best_lstm_units),
			layers.Dense(32, activation='relu'),
			layers.Dense(1)
		])
	if model=='Stacked-LSTM':
		best_lstm_units = best_params['lstm_units']
		best_dense_units = best_params['dense_units']
		best_learning_rate = best_params['learning_rate']
		best_num_lstm_layers = best_params['num_lstm_layers']
		m1 = Sequential()
		m1.add(layers.Input((shift, 1)))
		for _ in range(best_num_lstm_layers):
			m1.add(layers.LSTM(best_lstm_units, return_sequences=True))
		m1.add(layers.Flatten())
		m1.add(layers.Dense(best_dense_units, activation='relu'))
		m1.add(layers.Dense(1))
	if model=='GRU':
		best_gru_units = best_params['gru_units']
		best_dense_units = best_params['dense_units']
		best_learning_rate = best_params['learning_rate']
		m1 = Sequential([
			layers.Input((shift, 1)),
			layers.GRU(64),
			layers.Dense(32, activation='relu'),
			layers.Dense(1)
			])
	if model=='1D-CNN':
		num_filters = best_params['num_filters']
		kernel_size = best_params['kernel_size']
		best_dense_units = best_params['dense_units']
		best_learning_rate = best_params['learning_rate']
		m1 = Sequential([
			layers.Input((shift, 1)),
			layers.Conv1D(num_filters, kernel_size, activation='relu',padding='same'),
			layers.MaxPooling1D(),
			layers.Flatten(),
			layers.Dense(32, activation='relu'),
			layers.Dense(1)
		])

	m1.compile(loss='mse', 
	                optimizer=Adam(learning_rate=best_learning_rate),
	                metrics=['mean_absolute_error'])

	h=m1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)
	test_loss = m1.evaluate(X_test, y_test, verbose=0)
	fig, ax = plt.subplots()
	ax.plot(h.history['loss'], label='Testing Loss')
	ax.set_xlabel('Epochs')
	ax.set_ylabel('Loss')
	ax.legend()
	st.pyplot(fig)
	st.write('done')
	return m1

def predict(m1,bestperms,dates_train,X_train,y_train,dates_val,X_val,y_val,dates_test,X_test,y_test,shift,scaler):
	test_predictions = m1.predict(X_test)
	train_predictions = m1.predict(X_train)
	val_predictions = m1.predict(X_val)
	y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
	y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
	y_train_pred_actual = scaler.inverse_transform(train_predictions.reshape(-1,1))
	y_test_pred_actual = scaler.inverse_transform(test_predictions.reshape(-1,1))
	y_val_pred_actual = scaler.inverse_transform(val_predictions.reshape(-1,1))
	y_val_actual = scaler.inverse_transform(y_val.reshape(-1,1))
	test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
	test_mse = mean_squared_error(y_test_actual, y_test_pred_actual)
	test_r = np.sqrt(test_mse)
	test_r2 = r2_score(y_test_actual, y_test_pred_actual)
	st.write("Mean Absolute Error (MAE):", test_mae)
	st.write("Mean Squared Error (MSE):", test_mse)
	st.write("Root Mean Squared Error (RMSE):", test_r)
	st.write("R-squared (R2) Score:", test_r2)

	fig, ax = plt.subplots()
	ax.plot(dates_test, test_predictions, linestyle="-", marker='o', markersize=3)
	ax.plot(dates_test, y_test, linestyle="--", marker='x', markersize=3)
	ax.legend(['Testing Predictions', 'Testing Observations'])
	ax.set_xlabel('Dates')
	ax.set_ylabel('Values')
	st.pyplot(fig)

	fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize as needed
	ax.plot(dates_train, train_predictions)
	ax.plot(dates_train, y_train)
	ax.plot(dates_val, val_predictions)
	ax.plot(dates_val, y_val)
	ax.plot(dates_test, test_predictions)
	ax.plot(dates_test, y_test)
	ax.legend(['Training Predictions', 'Training Observations',
	           'Validation Predictions', 'Validation Observations',
	           'Testing Predictions', 'Testing Observations'], loc='upper left')
	ax.set_xlabel('Dates')
	ax.set_ylabel('Values')

	st.pyplot(fig)


def predictfuture(days,m1,bestperms,dates_train,X_train,y_train,dates_val,X_val,y_val,dates_test,X_test,y_test,shift,scalar,dates,X,y):
	n = days
	forecast = []
	last_data = X[-1]  # Last X data point
	for i in range(n):
		next_y = m1.predict(last_data.reshape(1, X.shape[1], X.shape[2]))  
		forecast.append(next_y)
		last_data = np.append(last_data[1:], next_y)  
	forecast = scalar.inverse_transform(np.array(forecast).reshape(-1, 1))
	y_act=scalar.inverse_transform(np.array(y).reshape(-1, 1))
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(dates, y_act, label='Original Data')
	ax.set_xlabel('Time')
	ax.set_ylabel('Value')
	ax.set_title('Original Data and Forecast')
	ax.tick_params(axis='x', rotation=45)
	ax.legend()
	forecast_dates = np.arange(len(y_act), len(y_act) + n)
	ax.plot(forecast_dates, forecast, label='Forecast', color='red')
	ax.legend()
	st.pyplot(fig)









