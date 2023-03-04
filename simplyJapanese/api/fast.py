from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simplyJapanese.model.registery import load_model, load_tokenizer

app = FastAPI()
app.state.model = load_model()
app.state.tokenizer = load_tokenizer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(input_data):
    """\
    Taking the input from the user into our.\\
    """

    # preprocess data (sentence splitting, normalization, )
    X_pred = "cleaned and tokenized data"

    # make predictions
    y_pred = app.state.model.predict(X_pred)

    # Format y_pred to match output data requirements
    output_data = y_pred

    print(app.state.tokenizer)
    return {"activity": "It's happpening"}


@app.get("/")
def root():
    return {'greeting': 'Hello'}
