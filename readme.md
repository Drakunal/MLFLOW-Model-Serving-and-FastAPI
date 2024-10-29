
# Customer Churn Prediction API

This project is a FastAPI-based API that allows single and batch customer churn predictions, using a machine learning model hosted by MLflow. The API enables efficient prediction handling, with structured routes for single-customer JSON input and batch predictions via CSV upload.

## Table of Contents
1. [Project Setup](#project-setup)
2. [Virtual Environment](#virtual-environment)
3. [Installing Dependencies](#installing-dependencies)
4. [Serving the Model](#Serving-the-Model-with-MLflow)
5. [Running the API](#running-the-api)
6. [API Endpoints](#api-endpoints)
7. [Code Explanation](#code-explanation)
8. [Postman Implementation](#postman)

---

## Project Setup

### Requirements

To ensure compatibility, make sure to have a `requirements.txt` file containing the exact versions of packages used in the project.

Example `requirements.txt`:
```
fastapi==<version>
uvicorn==<version>
python-multipart==<version>
requests==<version>
pandas==<version>
numpy==<version>
```

Ensure each dependency version matches(with the one which you already used for the previous lab, the first 3 can be latest version) to avoid compatibility issues, especially if MLflow is serving the model.

---

## Virtual Environment

Create and activate a virtual environment to keep dependencies isolated:

```bash
# On macOS/Linux
python3.10 -m venv venv

# On Windows
python -m venv venv
```

Activate the virtual environment:

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

## Installing Dependencies

With the virtual environment activated, install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Serving the Model with MLflow

1. **Install pyenv**: MLflow requires `pyenv` to manage Python environments.

   - Download and install pyenv.
   - Add pyenv to your system path: `C:\Users\<Username>\.pyenv\pyenv-win\bin`.

2. **Serve the model**:
   Navigate to the directory containing the `mlruns` folder and run the following command:

   ```bash
   mlflow models serve -m "models:/model_name/version_number" -p 1234
   ```

   Replace `"model_name/version_number"` with your model's name and version.

---

## Running the API

Install necessary FastAPI dependencies:

```bash
pip install fastapi uvicorn python-multipart
```

To start the API, use the following command:

```bash
uvicorn main:app --reload
```

---

## API Endpoints

### 1. GET `/`

- **Description**: A basic health check route to confirm that the API is up and running.
- **Method**: `GET`
- **Response**: `{"message": "Hello World"}`

### 2. POST `/predict`

- **Description**: Predicts the churn probability for a single customer based on provided features.
- **Method**: `POST`
- **Input**: JSON body with customer details in the format specified by `ChurnClass`.
- **Response**: Prediction result in JSON format from the MLflow model server.

### 3. POST `/files/`

- **Description**: Handles batch predictions by accepting a CSV file containing customer data.
- **Method**: `POST`
- **Input**: File (CSV) uploaded via form-data.
- **Response**: Predictions for each customer in the CSV.

---

Here's a more detailed explanation of the code sections in your project:

---

## Code Explanation

The API is structured in the FastAPI framework with routes for single and batch customer churn predictions, using an MLflow-served model as the prediction backend.

### 1. Key Imports and Setup

```python
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from io import StringIO
import requests
```

- **`FastAPI`**: The primary class for building APIs in FastAPI. It enables routing, validation, and serialization of data.
- **`File`, `UploadFile`, `Form`**: FastAPI classes for handling file uploads (`File`, `UploadFile`) and form data (`Form`). These simplify data ingestion from external sources like Postman.
- **`BaseModel`**: Part of Pydantic, used to define the structure and data types for input validation in FastAPI.
- **`pickle`, `numpy`, `pandas`**: Libraries for data manipulation and loading models (if local). Here, `pandas` is used for processing CSV files, while `numpy` assists with numerical data handling.
- **`StringIO`**: Converts byte-encoded strings into file-like objects. Essential for reading file content (CSV) uploaded via API.
- **`requests`**: Python’s HTTP library for sending requests to the MLflow server hosting the model.

### 2. Initializing the FastAPI App and Data Model

```python
app = FastAPI()
```

- This line creates a FastAPI application instance, `app`, which organizes all routes and configurations for the API.

#### Defining the Customer Data Model

```python
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
```

- **`ChurnClass`**: This class inherits from `BaseModel` and defines the expected data fields for a single customer’s information, including data types (all `float` in this case). When a request is made to the `/predict` endpoint, FastAPI will automatically validate and parse the incoming JSON data to match this structure.
  
- **Attributes**: Each attribute represents a feature required by the ML model to make a prediction, such as `CreditScore`, `Age`, `Geography_France`, etc.

### 3. Route Definitions

FastAPI allows you to define endpoints (routes) for the API. Routes determine the URL pattern, HTTP method, and data flow for different functionalities. Decorators (e.g., `@app.get`) register each function to a route.

#### Route 1: GET `/`

```python
@app.get("/")
async def root():
    return {"message": "Hello World"}
```

- **Purpose**: A simple route to confirm the API server is running.
- **HTTP Method**: `GET` is used here, as this endpoint is only fetching data (in this case, a JSON message).
- **Decorator**: `@app.get("/")` tells FastAPI that this function should handle `GET` requests at the root URL (`/`).
- **`async` Function**: Using `async` here allows this route to be non-blocking, enhancing performance by allowing other tasks to execute concurrently.

#### Route 2: POST `/predict`

```python
@app.post("/predict")
async def predict_churn(churn: ChurnClass):
    data = churn.dict()
    data_in = [[
        data['CreditScore'], data['Age'], data['Tenure'], data['Balance'],
        data['NumOfProducts'], data['IsActiveMember'], data['EstimatedSalary'],
        data['Geography_France'], data['Geography_Germany'], data['Geography_Spain'],
        data['Gender_Female'], data['Gender_Male']
    ]]
    
    endpoint = "http://localhost:7777/invocations"
    inference_request = {"dataframe_records": data_in}
    
    response = requests.post(endpoint, json=inference_request)
    return response.text
```

- **Purpose**: This route accepts JSON data for a single customer and returns a churn prediction.
- **HTTP Method**: `POST` is used because we’re sending data to the server to create a new result (the prediction).
- **Input (`churn: ChurnClass`)**: The function accepts a single parameter, `churn`, which should be a JSON object that matches the `ChurnClass` schema. FastAPI validates this input before running the function.
- **Data Transformation**:
  - `data = churn.dict()`: Converts the Pydantic model instance (`churn`) into a Python dictionary.
  - `data_in`: Transforms the dictionary into a nested list, the format required by the MLflow model.
- **Inference Request**:
  - `endpoint`: URL of the MLflow model server.
  - `inference_request`: A dictionary with the key `dataframe_records`, where `data_in` is sent as the value.
  - `requests.post`: Sends this data to the model server, which performs the prediction.
- **Return**: `response.text` returns the response from the model server as plain text, typically the prediction value.

#### Route 3: POST `/files/`

```python
@app.post("/files/")
async def batch_prediction(file: bytes = File(...)):
    s = str(file, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    lst = df.values.tolist()
    
    inference_request = {"dataframe_records": lst}
    endpoint = "http://localhost:7777/invocations"
    response = requests.post(endpoint, json=inference_request)
    
    return response.text
```

- **Purpose**: Handles batch predictions by accepting a CSV file of customer data.
- **HTTP Method**: `POST`, as we are uploading a file to generate results.
- **Input**: The function accepts a single parameter, `file`, which represents the uploaded file. `File(...)` tells FastAPI that this parameter should come from a file upload.
- **Data Transformation**:
  - `s = str(file, 'utf-8')`: Converts the byte-encoded file content into a UTF-8 string.
  - `data = StringIO(s)`: Creates a file-like object from the string, which `pandas` can read.
  - `df = pd.read_csv(data)`: Reads the file as a DataFrame, organizing the data into rows and columns.
  - `lst = df.values.tolist()`: Converts the DataFrame into a list of lists (`lst`), which the model expects.
- **Inference Request**:
  - `inference_request`: Dictionary with the key `dataframe_records` and `lst` as the value.
  - `requests.post`: Sends the batch data to the model server at `endpoint`.
- **Return**: The text response from the model server, which includes predictions for each entry in the uploaded CSV.

---

## Postman
![image](https://github.com/user-attachments/assets/d05e9288-403f-4ba0-a0a0-cb9570de1893)
![image](https://github.com/user-attachments/assets/9ba87ea2-52ed-4003-bf95-1024c5928fc9)


