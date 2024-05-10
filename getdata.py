import tensorflow as tf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def getInput():
	with st.form(key='forecast_form'):
		commodity = st.selectbox(
		    'Select the Commodity',
		    ('rice', 'wheat', 'maize', 'potato', 'onion'))
		model = st.selectbox(
		    'Select the Model',
		    ('LSTM', 'Bi-LSTM', 'Stacked-LSTM', 'GRU', '1D-CNN'))
		number = st.number_input(
		    "Enter no of Months",
		    min_value=1,
		    max_value=10,  
		    value=3,  
		    step=1)
		forecast_button = st.form_submit_button(label='Forecast')
	if forecast_button:
		#if commodity is not None and model is not None and number is not None:
		st.empty()
		return commodity, model, number