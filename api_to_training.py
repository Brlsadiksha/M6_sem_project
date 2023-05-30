import requests
from xml.etree import ElementTree
from datetime import datetime, timedelta

import pytz
import xmltodict

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing

import mlflow


days_past = 180 # 6 months 
end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d") # tomorrow's date, As API gives us price until today midnight.  
print("Importing data from API")
print("Using end date : ", end_date)
print("Fetching data for past (days) :", days_past)


def calculate_start_date(end_date, days_past):
    local_end_date = datetime.strptime(end_date, "%Y-%m-%d")    
    local_tz = pytz.timezone("Europe/Copenhagen") #our local timezone
    
    utc_end_date = local_tz.localize(local_end_date).astimezone(pytz.utc) #localize adds the local timezone, astimezone(pytz.utc) changes the time UTC time.. API expects in UTC
    utc_end_date_string = utc_end_date.strftime("%Y%m%d%H%M") # API expects date time as 202305292200
    
    utc_start_date =utc_end_date - timedelta(days=days_past) #calculating start date
    utc_start_date_string = utc_start_date.strftime("%Y%m%d%H%M") #date to string
    
    print("Start date UTC : ", utc_start_date_string)
    print("End date UTC : ", utc_end_date_string)

    return utc_start_date_string, utc_end_date_string



start_date, end_date = calculate_start_date(end_date, days_past)
payload={}
headers = {}

print(f"Sending API request with start date as {start_date} and end_date as {end_date}")


token = "79d02170-206a-4f7d-ac31-01b492203cf4"
region = "10YDK-1--------W"
document_type = "A44"
url = f"https://web-api.tp.entsoe.eu/api?documentType={document_type}&in_Domain={region}&out_Domain={region}&periodStart={start_date}&periodEnd={end_date}&securityToken={token}"


response = requests.request("GET", url, headers=headers, data=payload)
dict_data = xmltodict.parse(response.content)["Publication_MarketDocument"] # xml to dictionary 


def convert_utc_to_copenhagen(utc_string): #API returns time as UTC but we want our local time
    # Parse the UTC string into a datetime object
    utc_time = datetime.strptime(utc_string, "%Y-%m-%dT%H:%MZ") #extract time from string

    copenhagen_timezone = pytz.timezone('Europe/Copenhagen')
    copenhagen_time = pytz.utc.localize(utc_time).astimezone(copenhagen_timezone) #convert time to local time
    copenhagen_time_string = copenhagen_time.strftime("%Y%m%d%H%M") # back to string, used as start and end values in data_time index

    return copenhagen_time_string


dates = [] 
values = []

#if only one day of data is fetched from API, the results is in Dict. for more it is list of dict
if type(dict_data['TimeSeries']) != list: # for our code we need list of dict, so converting
    dict_data['TimeSeries'] = [dict_data['TimeSeries']]

print("Extracting data from API response")

for api_day_data in dict_data["TimeSeries"]: #each day is a dict
    day_values = []
    start = convert_utc_to_copenhagen(api_day_data["Period"]["timeInterval"]["start"]) 
    # start datetime of the day, API gives it in UTC (202305252200), we convert to localtime
    end = convert_utc_to_copenhagen(api_day_data["Period"]["timeInterval"]["end"])
    date_range = pd.date_range(start=start, end=end, freq="H", inclusive="left")
    #generating datatime for 24 hr
    api_ts = api_day_data["Period"]["Point"]

    for hour in api_ts[:24]: # limit to 24 in case of duplicate, the date_range values always has 24 hr sometime API gives 23 or 25 values because of change in timezone. if 25 limit to 24
        day_values.append(float(hour["price.amount"]))
    
    while len(day_values) < 24: #if 23 values add nan 
        day_values.append(np.NaN)
        # add np.NaN so that we can fill them later       
    
# we extract values and date day by day because sometimes API has missing values
    values.extend(day_values) # all the prices
    dates.extend(date_range) # all the dates

print("Creating darts TimeSeries")


df = pd.DataFrame({"date": dates, "price": values})
median_price = df['price'].median() #for the missing values


#series = TimeSeries.from_dataframe(df, time_col="date", value_cols ="price", fill_missing_dates=False, freq="H") # Just for plotting

filled_series = TimeSeries.from_dataframe(df, time_col="date", value_cols ="price", fill_missing_dates=True, fillna_value=median_price, freq="H") # use this for training, add missing dates and filled with median values 

data_path = f"data/data_{end_date}.csv"
filled_series.to_csv(data_path)
print("Data saved to : ", data_path)

print("Training and saving model")
# Training and saving to mlflow
mlflow.set_experiment("elprice_predictor")
mlflow.start_run(run_name="model_es_pred", nested=True) # Start MLflow run
mlflow.sklearn.autolog()

model_es = ExponentialSmoothing(seasonal_periods=24)
model_es.fit(filled_series)
mlflow.sklearn.log_model(model_es, "model_es_pred")

# Register the model in the MLflow registry
run_id_active = mlflow.active_run().info.run_id
model_uri = "runs:/" + run_id_active + "/model_es_pred"
model_version = mlflow.register_model(model_uri, "model_es_pred")

mlflow.end_run()
print("Training ended")