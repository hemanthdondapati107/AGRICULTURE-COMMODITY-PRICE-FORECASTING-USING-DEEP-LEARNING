import tensorflow as tf
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from getdata import *
from getperms import *

def main():
    st.title('Time Series Forecasting with Deep Learning Model and Streamlit')
    inputs=getInput()
    if inputs:
        commodity=inputs[0]
        model=inputs[1]
        days=inputs[2]
        load(commodity,model,days)


if __name__ == '__main__':
    main()