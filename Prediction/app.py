from fastapi import FastAPI
#from darts.models import ExponentialSmoothing
import mlflow.pyfunc
import pandas
import json
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from darts import TimeSeries
#from darts.models import ExponentialSmoothing
import requests
import streamlit as st
from datetime import datetime

response = requests.get('http://127.0.0.1:8000/predict/24')
print(response.json())
df = pd.DataFrame(response.json())


# data_table1.iloc[1] = data_table1.iloc[1].round(2)
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'time','price':'spot price(Euro/Mwh)'})
df["spot price(Euro/Mwh)"] = df["spot price(Euro/Mwh)"].apply(lambda x : round(x, 2))
df["time"] = df["time"].apply(lambda x : x[11:16])

#df = df.rename(columns={'index': 'time', 'price':'spot price(Euro/Mwh)'})
#df['spot price(Euro/Mwh)'] = df['spot price(Euro/Mwh)'].round(2)
#columns = data_table1['date', 'spot price(Euro/Mwh)']

# date
#df = pd.DataFrame(data_table1, columns=columns)
today = datetime.today().date()
today_str = today.strftime("%Y-%m-%d")

# price 
high= df['spot price(Euro/Mwh)'].max().round(2)
low = df['spot price(Euro/Mwh)'].min().round(2)
avg = df['spot price(Euro/Mwh)'].mean().round(2)


# streamlit
st.sidebar.header("Electric Price Prediction")
st.sidebar.metric("Today", today_str, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.sidebar.metric("Highest Price", high, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.sidebar.metric("Average Price", avg, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.sidebar.metric("Lowest Price", low, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.title(" Electric Price Prediction")

st.markdown('Prediction Price :') 

talbe_width = 800
table_height = min(len(df) * 100, 500)
st.dataframe(df, width=talbe_width, height=table_height)

st.markdown('Chart of Prediction Price  :') 
st.bar_chart(data=df, x='time', y='spot price(Euro/Mwh)', width=0, height=0, use_container_width=True)
# st.line_chart(df['time'], df['spot price(Euro/Mwh)'])
