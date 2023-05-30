from fastapi import FastAPI
from darts.models import ExponentialSmoothing
import mlflow.pyfunc
import pandas
import json

app = FastAPI()

@app.get("/predict/{num_pred}")
def results(num_pred):
    model_uri = "models:/model_es_pred/latest"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    pred = loaded_model.predict(int(num_pred))
    return pred.pd_dataframe().to_dict() #change according to datatype needed to streamlit