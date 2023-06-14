#from fastapi import FastAPI
# from darts.models import ExponentialSmoothing
# import mlflow.pyfunc
# import pandas as pd
# import json
# import streamlit as st
# import requests
# from datetime import datetime, timedelta


# st.title('Electricity Price Prediction')
# response_next_day = requests.get('http://127.0.0.1:8000/predict/24')

# #print(response.json())
# df = pd.DataFrame(response_next_day.json())

# #data_table1.iloc[1] = data_table1.iloc[1].round(2)

# df.reset_index(inplace=True)
# df = df.rename(columns={'index': 'time','price':'spot price(DKK/kWh)'})
# df["spot price(DKK/kWh)"] = df["spot price(DKK/kWh)"].apply(lambda x : round(x, 4))


# #df['spot price(DKK/kWh)'] = df['spot price(DKK/kWh)'].round(2)


# today = datetime.today().date()
# tomorrow = today + timedelta(days=1)

# today_str = today.strftime("%Y-%m-%d")
# tomorrow_str = tomorrow.strftime("%Y-%m-%d")

# # price

# high= df['spot price(DKK/kWh)'].max().round(4)
# low = df['spot price(DKK/kWh)'].min().round(4)
# avg = df['spot price(DKK/kWh)'].mean().round(4)

# # streamlit
# st.sidebar.header("Electric Price Prediction")
# st.sidebar.metric("Today", today_str, delta=None, delta_color="normal", help=None, label_visibility="visible")
# st.sidebar.metric("Hight Price", high, delta=None, delta_color="inverse", help=None, label_visibility="visible")
# st.sidebar.metric("Average Price", avg, delta=None, delta_color="inverse", help=None, label_visibility="visible")
# st.sidebar.metric("Low Price", low, delta=None, delta_color="inverse", help=None, label_visibility="visible")
# st.title(" Electric Price Prediction")
# st.markdown('Prediction Price :')
# filtered_df = dataframe_explorer(df, case=False)
# st.dataframe(filtered_df, use_container_width=True)

# st.markdown('Chart of Prediction Price  :')
# st.line_chart(df)

# chart_data = pd.DataFrame(response.json(), columns=[ 'date', 'price'])

# st.line_chart(chart_data)


from fastapi import FastAPI
#from darts.models import ExponentialSmoothing
import mlflow.pyfunc
import pandas
import json
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_extras.dataframe_explorer import dataframe_explorer
import altair as alt


#from darts import TimeSeries
#from darts.models import ExponentialSmoothing
import requests
import streamlit as st
from datetime import datetime, timedelta


response = requests.get('http://127.0.0.1:8000/predict/24')
#print(response.json())
df = pd.DataFrame(response.json())


# data_table1.iloc[1] = data_table1.iloc[1].round(2)
df.reset_index(inplace=True)
df = df.rename(columns={'index': 'time','price':'predicted spot price(DKK/Kwh)'})
df['predicted spot price(DKK/Kwh)'] = df['predicted spot price(DKK/Kwh)'].apply(lambda x : round(x, 2))
df['time'] = pd.to_datetime(df['time'])
df.set_index("time", inplace=True)
#df['time'] = df['time'].dt.strftime('%H:%M:%S')

#df = df.rename(columns={'index': 'time', 'price':'spot price(DKK/Kwh)'})
#df['spot price(DKK/Kwh)'] = df['spot price(DKK/Kwh)'].round(2)
#columns = data_table1['date', 'spot price(DKK/Kwh)']

# date
#df = pd.DataFrame(data_table1, columns=columns)
tomorrow = datetime.today() + timedelta(days=1)
tomorrow_str = tomorrow.strftime("%Y-%m-%d")

# price 
high= df['predicted spot price(DKK/Kwh)'].max().round(2)
low = df['predicted spot price(DKK/Kwh)'].min().round(2)
avg = df['predicted spot price(DKK/Kwh)'].mean().round(2)


# streamlit
st.sidebar.header('Electric Price Prediction')
st.sidebar.metric('Tomorrow', tomorrow_str, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.sidebar.metric('Highest Price', high, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.sidebar.metric('Average Price', avg, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.sidebar.metric('Lowest Price', low, delta=None, delta_color="inverse", help=None, label_visibility="visible")
st.title('Electric Price Prediction')

#st.markdown('Prediction Price :') 

past_days = st.number_input(label ='Past data for (days)',
                 min_value=0, max_value=7, value=0, step=1)



if past_days != 0:
    daily_response = requests.get(f'http://127.0.0.1:8000/past/true/{past_days*24}')
    daily_df = pd.DataFrame(daily_response.json()).reset_index()
    daily_df = daily_df.rename(columns={'index': 'time','price':'actual spot price(DKK/Kwh)'})
    daily_df['actual spot price(DKK/Kwh)'] = daily_df['actual spot price(DKK/Kwh)'].apply(lambda x : round(x, 2))
    daily_df['time'] = pd.to_datetime(daily_df['time'])
    daily_df.set_index("time", inplace=True)


    historical_response = requests.get(f'http://127.0.0.1:8000/past/predict/{past_days*24}')
    historical_df = pd.DataFrame(historical_response.json()).reset_index()
    historical_df = historical_df.rename(columns={'index': 'time','price':'predicted spot price(DKK/Kwh)'})
    historical_df['predicted spot price(DKK/Kwh)'] = historical_df['predicted spot price(DKK/Kwh)'].apply(lambda x : round(x, 2))
    historical_df['time'] = pd.to_datetime(historical_df['time'])
    historical_df.set_index("time", inplace=True)
    
    
    merged_df = pd.concat([daily_df, historical_df], axis=1)
    merged_df = pd.concat([merged_df, df], axis=0)
    
    y_columns = ['predicted spot price(DKK/Kwh)', 'actual spot price(DKK/Kwh)']

else:
    merged_df = df # If past days 0 then it would just be this
    y_columns = ['predicted spot price(DKK/Kwh)']

merged_df.reset_index(inplace=True) # To get time as column, it would also let us filter it on time


tab1, tab2 = st.tabs([':bar_chart: Chart', ':page_with_curl: Table'])


with tab1:
    st.markdown('Chart of Prediction Price  :') 
    st.bar_chart(data=merged_df, x='time', y=y_columns, use_container_width=True)
    

with tab2:
    
    filtered_df = dataframe_explorer(merged_df, case=False)
    st.dataframe(filtered_df, use_container_width=True)    

