from fastapi import FastAPI
from darts.models import ExponentialSmoothing
import mlflow.pyfunc
import pandas as pd
import json
import glob


app = FastAPI()

@app.get("/predict/{num_pred}")
def results(num_pred):
    if int(num_pred) == 0:
        return {}
    model_uri ='models:/model_es_pred/latest'
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    pred = loaded_model.predict(int(num_pred))
    return pred.pd_dataframe().to_dict() #change according to datatype needed to streamlit

@app.get('/past/true/{num_hours}')
def past_true(num_hours):
    num_hours = int(num_hours)
    if num_hours == 0:
        return {}
    daily_files = glob.glob('Daily_DATA/data_*.csv')
    daily_files.sort(reverse=True)
    daily_df = pd.read_csv(daily_files[0])
    daily_df.set_index('date', inplace=True)
    return daily_df[-num_hours:].to_dict()

@app.get('/past/predict/{num_hours}')
def past_predict(num_hours):
    num_hours = int(num_hours)
    if num_hours == 0:
        return {}
    historical_files = glob.glob('Daily_DATA/historical-forecast_*.csv')
    historical_files.sort(reverse=True)
    historical_df = pd.read_csv(historical_files[0])
    historical_df.set_index('time', inplace=True)
    
    return historical_df[-num_hours:].to_dict()