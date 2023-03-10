import os
import pandas as pd
from transformers import TFT5ForConditionalGeneration
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer
from simplyJapanese.utils.scoring import evaluate_blue_score

def load_mainmodel():
    path = "/home/andi/code/playground/model"
    model = load_model(path, custom_objects={"TFT5ForConditionalGeneration":TFT5ForConditionalGeneration}, compile=False, options=None)
    MODEL_NAME = "sonoisa/t5-base-japanese"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print('Model loaded!')
    return model, tokenizer

def pred(text, model, tokenizer):
    tokenized = tokenizer([text], return_tensors='np')
    generated = model.generate(**tokenized, max_length=128)
    pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(pred_text)
    return pred_text


def get_data(file):
    """
    Gets csv data under from data
    Returns as Dataframe where columns=['original','simplified']
    """
    path = "/home/andi/code/mochiyam/simply-japanese/simply-japanese"
    path = os.path.join(path, "simplyJapanese", "data", "1_RawData")

    df = pd.read_excel(os.path.join(path, file))

    df.drop(columns=['#英語(原文)','#固有名詞'], inplace=True, errors='ignore')
    df.rename(columns={"#日本語(原文)": "original", "#やさしい日本語": "simplified"}, inplace=True)
    print(f"Dataframe created! {file}")
    return df

def predict_main(file):
    # load model and tokenizer
    model, tokenizer = load_mainmodel()
    # Create df
    df = get_data(file)
    # Make preds (list)
    predictions = [pred(sentence, model, tokenizer) for sentence in df["original"]]
    # add preds to df
    df = df.assign(predictions=predictions)
    print("df assigned")
    return df

def save_main(file):
    predictions = predict_main(file)
    ## Score baseline
    # Add BLUE score for original simplification
    # Add BLUE score for baseline model simplification; original data vs preds
    predictions = evaluate_blue_score(predictions, 0, 2, "Baseline BLUE score vs original")
    # Define output path for final XLSX
    path = "/home/andi/code/mochiyam/simply-japanese/simply-japanese"
    output_path = os.path.join(path, "simplyJapanese", \
                                "data", "2_ProcessedData")
    os.chdir(output_path)
    predictions.to_excel(f"{file}_main_predictions.xlsx")


# file = "SNOW_T15.xlsx"
file = "SNOW_T15_150.xlsx"
# file = "SNOW_T15_2.xlsx"

save_main(file)
