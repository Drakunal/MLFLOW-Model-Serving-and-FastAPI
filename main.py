from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from io import StringIO
import requests

# Initialize FastAPI application
app = FastAPI()

# Define a Pydantic model for the input data schema expected by the churn prediction model
class ChurnClass(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    IsActiveMember: float
    EstimatedSalary: float
    Geography_France: float
    Geography_Germany: float
    Geography_Spain: float
    Gender_Female: float
    Gender_Male: float

# Basic root endpoint to check if the API is running
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Endpoint to predict churn based on a single user's data
@app.post("/predict")
async def predict_churn(churn: ChurnClass):
    """
    Receives input data for a single user, transforms it to the required format,
    sends it to the machine learning model API, and returns the model's response.
    """
    # Convert Pydantic model to dictionary
    data = churn.dict()

    # Format input data as a 2D array (list of lists) for the model
    data_in = [[
        data['CreditScore'], 
        data['Age'], 
        data['Tenure'], 
        data['Balance'],
        data['NumOfProducts'],
        data['IsActiveMember'],
        data['EstimatedSalary'],
        data['Geography_France'],
        data['Geography_Germany'],
        data['Geography_Spain'],
        data['Gender_Female'],
        data['Gender_Male']
    ]]

    # Print data_in for debugging purposes
    print("Input data for model:", data_in)

    # Define the endpoint for the machine learning model
    endpoint = "http://localhost:7777/invocations"
    
    # Create a request payload for the model
    inference_request = {"dataframe_records": data_in}
    
    # Send the request to the ML model and receive response
    response = requests.post(endpoint, json=inference_request)
    
    # Return the response text from the model as the API response
    return response.text

# Endpoint to perform batch predictions from a CSV file
@app.post("/files/")
async def batch_prediction(file: bytes = File(...)):
    """
    Accepts a CSV file, processes its contents to match the input format of the ML model,
    and returns predictions for each row in the CSV file.
    """
    # Convert uploaded file bytes to a string
    s = str(file, 'utf-8')
    
    # Use StringIO to read the CSV data as a pandas DataFrame
    data = StringIO(s)
    df = pd.read_csv(data)
    
    # Convert the DataFrame to a list of lists (expected format for model input)
    lst = df.values.tolist()
    
    # Create a request payload for the batch data
    inference_request = {"dataframe_records": lst}
    
    # Define the model's endpoint
    endpoint = "http://localhost:7777/invocations"
    
    # Send the request to the ML model and receive the batch response
    response = requests.post(endpoint, json=inference_request)
    
    # Print the response text for debugging
    print("Batch response:", response.text)
    
    # Return the response text from the model as the API response
    return response.text

# To start the FastAPI server, run the following command in the terminal:
# uvicorn main:app --reload
