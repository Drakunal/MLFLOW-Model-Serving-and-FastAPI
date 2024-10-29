
# Customer Churn Prediction API

This project is a FastAPI-based API that allows single and batch customer churn predictions, using a machine learning model hosted by MLflow. The API enables efficient prediction handling, with structured routes for single-customer JSON input and batch predictions via CSV upload.

## Table of Contents
1. [Project Setup](#project-setup)
2. [Virtual Environment](#virtual-environment)
3. [Installing Dependencies](#installing-dependencies)
4. [Serving the Model](#serving-the-model)
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

## Code Explanation

The API uses FastAPI to handle requests, with a Pydantic model to validate incoming data. It includes two primary routes: one for single predictions and one for batch predictions.

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

- **`FastAPI`**: The main class for creating the API, providing tools for routing and data validation.
- **`File`, `UploadFile`, `Form`**: Enable file uploads (CSV) and form data handling.
- **`BaseModel`**: A Pydantic class used to define the expected input data format and validate it.
- **`pickle`, `numpy`, `pandas`**: These libraries support data handling and model management.
- **`StringIO`**: Converts byte-encoded strings to a file-like object, essential for reading CSV content.
- **`requests`**: Used to send HTTP requests to the MLflow model server for predictions.

### 2. Initializing the FastAPI App and Data Model

```python
app = FastAPI()
```

Creates a FastAPI application instance that organizes routes and configurations.

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

- **Purpose**: Defines the expected fields for customer data, ensuring consistency in data format for model predictions.
- **Attributes**: Each attribute maps to a feature required by the model.

### 3. Route Definitions

FastAPI routes determine the URL pattern, HTTP method, and data handling. Decorators like `@app.get` and `@app.post` specify HTTP methods and paths.

#### Route 1: GET `/`

```python
@app.get("/")
async def root():
    return {"message": "Hello World"}
```

- **Purpose**: Checks API availability.
- **HTTP Method**: `GET`.
- **Decorator**: Registers the route with FastAPI.
- **`async` Function**: Ensures non-blocking execution for better performance.

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

- **Purpose**: Predicts churn for a single customer.
- **HTTP Method**: `POST`.
- **Input (`churn: ChurnClass`)**: Validates that the incoming JSON data matches the `ChurnClass` schema.
- **Data Processing**: Converts the data into a format compatible with the MLflow model (`data_in`).
- **Inference Request**: Sends `data_in` to the MLflow model for prediction.
- **Return**: Model server response.

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

- **Purpose**: Processes batch predictions from a CSV file.
- **HTTP Method**: `POST`.
- **Input**: CSV file uploaded as form-data.
- **Data Processing**:
  - Converts file bytes to a string and loads it as a DataFrame.
  - Converts DataFrame to list format for the model.
- **Inference Request**: Sends the formatted data to the model server.
- **Return**: Model server response with predictions.

---

## Postman
![image](https://github.com/user-attachments/assets/d05e9288-403f-4ba0-a0a0-cb9570de1893)
![image](https://github.com/user-attachments/assets/9ba87ea2-52ed-4003-bf95-1024c5928fc9)


