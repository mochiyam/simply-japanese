from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from simplyJapanese.model.registery import load_the_model, load_tokenizer

app = FastAPI()
app.state.model = load_the_model()
app.state.tokenizer = load_tokenizer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(input_text: str):
    """
    Taking the input text and predict using our model.
    """
    tokenized = app.state.tokenizer([input_text], return_tensors='np')
    generated = app.state.model.generate(**tokenized, max_length=128)
    pred_text = app.state.tokenizer.decode(generated[0], skip_special_tokens=True)

    return {"activity": pred_text}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
