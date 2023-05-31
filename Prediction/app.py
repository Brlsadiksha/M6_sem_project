from fastapi import FastAPI
from darts.models import ExponentialSmoothing
import mlflow.pyfunc
import pandas as pd
import json
import streamlit as st
import requests
from datetime import datetime, timedelta
from streamlit_extras.dataframe_explorer import dataframe_explorer


st.title('Electricity Price Prediction')
response_next_day = requests.get('http://127.0.0.1:8000/predict/24')

#print(response.json())
df = pd.DataFrame(response_next_day.json())

#data_table1.iloc[1] = data_table1.iloc[1].round(2)

df.reset_index(inplace=True)
df = df.rename(columns={'index': 'time','price':'spot price(DKK/kWh)'})
df["spot price(DKK/kWh)"] = df["spot price(DKK/kWh)"].apply(lambda x : round(x, 4))


#df['spot price(DKK/kWh)'] = df['spot price(DKK/kWh)'].round(2)


today = datetime.today().date()
tomorrow = today + timedelta(days=1)

today_str = today.strftime("%Y-%m-%d")
tomorrow_str = tomorrow.strftime("%Y-%m-%d")

# price

high= df['spot price(DKK/kWh)'].max().round(4)
low = df['spot price(DKK/kWh)'].min().round(4)
avg = df['spot price(DKK/kWh)'].mean().round(4)

# streamlit
st.sidebar.header("Electric Price Prediction")
st.sidebar.metric("Today", today_str, delta=None, delta_color="normal", help=None, label_visibility="visible")
st.sidebar.metric("Hight Price", high, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.sidebar.metric("Average Price", avg, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.sidebar.metric("Low Price", low, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.title(" Electric Price Prediction")
st.markdown('Prediction Price :')
filtered_df = dataframe_explorer(df, case=False)
st.dataframe(filtered_df, use_container_width=True)

st.markdown('Chart of Prediction Price  :')
st.line_chart(df)

chart_data = pd.DataFrame(response.json(),
    columns=[ 'date', 'price'])

st.line_chart(chart_data)

