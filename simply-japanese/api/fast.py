from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()
app.state.model = load_model() # Needs to be defined

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(input_data):
    """

    """

    # preprocess data (sentence splitting, normalization, )
    X_pred = "cleaned and tokenized data"

    # make predictions
    y_pred = app.state.model.predict(X_pred)

    # Format y_pred to match output data requirements
    output_data = y_pred

    return {"fare_amount": output_data}


@app.get("/")
def root():
    return {'greeting': 'Hello'}
