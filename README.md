# M6 - Electricity Price Forecasting

🚀 Description

Hello! you can find in this Repo all the technical information regarding the model we created for our semester project.


📋 Table of Contents

- Data
<br> Data used for training and testing the different models, download from ENTSO-E. Inside this folder we have two datsets 
  - Day-ahead Prices.csv
     Dataset of prices from 2023 Jan 1 to 2023 May 26
  - df_clean.csv 
    Clean dataset after doing data preparation
    
- Models
<br>Different models' training, testing and comparisson. The model we have use for the prediction for future 24 hrs is Exponential_Smothing which is inside the 'Exponential Smoothing Model.ipynb' notebook. 

- Prediction using API
 <br> The selected model for EPF combined with API and the deployment (streamlit). Inside this we have .py files which has full process from downloading the live data through API to streamlit.
    
    `python api_to_training.py`
   <br> This fetch the data till today's midnight (23:00) and train the model and save it into the mlflow.
   
    `unicorn prediction:app --reload`
   <br> This will get the model from mlflow and do the prediction and run fastapi for us to see the prediction
   
    `localhost:8000/predict/24`
  <br> We can see the prediction for 24hrs starting from today midnight. If we want to predict for more or less hour than 24, we can just change the number 24 to number of hours we want to predict. 
  
    `streamlit run app.py`
  <br> To run our streamlit app
  
  
- Preprocessing
 <br> Data preprocessing code chunk in order to fit the data in the models. 

🔧 Usage

You can navigate through the Repo as follows:

  1. See the preprocessing code chunk, form we got the final dataset called 'df_clean.csv'
  2. Navigate through the models to see how we trained and tested them. Also see the comparison. We have used the model Exponnetial_Smothing for our prediction. 
  3. Go to the predictions in order to see how the final model pulls data through an API and retrains itself, you can also find information for our Streamlit app.
  
