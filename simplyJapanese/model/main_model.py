import os
import time
import pandas as pd
import numpy as np
from datasets import (Dataset,
                      DatasetDict,
                      load_dataset,
                      load_metric)
from sklearn.model_selection import train_test_split

from transformers import (TFAutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq,
                          AdamWeightDecay)
from transformers.keras_callbacks import KerasMetricCallback
from tensorflow.keras.callbacks import EarlyStopping

from simplyJapanese.model.params import (MODEL_NAME,
                          INPUT_COL_NAME,
                          LABEL_COL_NAME,
                          MAX_TOKEN_INPUT_LENGTH,
                          MAX_TOKEN_TARGET_LENGTH)

from simplyJapanese.model.params import (BATCH_SIZE,
                          LEARNING_RATE,
                          WEIGHT_DECAY,
                          NUM_EPOCHS)

from simplyJapanese.model.registery import save_model, load_tokenizer

def preprocess_and_train():
    """
    Preprocess the raw data retreived as an xlsx file.
    Fine-tuning and train on pretrained model.
    """
    path = os.path.join("simplyJapanese", 'data', "1_RawData")

    if os.environ.get("DATA_SOURCE") == "TEST":
        file = os.environ.get("TEST_DATA")
    elif os.environ.get("DATA_SOURCE") == "DEPLOY":
        file = os.environ.get("DEPLOY_DATA")
    else:
        raise Exception ("Data source not or incorrectly specified in env.")

    df = pd.read_excel(os.path.join(path, file))

    df = clean_data(df)
    datasets = train_val_test_split(df)
    tokenizer = load_tokenizer()

    # FIXME this gotta go outta here
    def tokenize_fn(datasets: DatasetDict):
        inputs = [doc for doc in datasets[INPUT_COL_NAME]]
        model_inputs = tokenizer(inputs, max_length=MAX_TOKEN_INPUT_LENGTH , truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                datasets[LABEL_COL_NAME], max_length=MAX_TOKEN_TARGET_LENGTH, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]

        print("\n✅ Data Tokenized!")

        return model_inputs

    tokenized_datasets = datasets.map(tokenize_fn, batched=True)

    # Load the metric score.
    metric = load_metric("sacrebleu")

    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, from_pt=True)

    # Data collator that will dynamically pad the inputs received, as well as the labels.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="np")
    # Compile generation loop with XLA generation to improve speed!  add pad_to_multiple_of to avoid
    # variable input shape, because XLA no likey.
    generation_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="np", pad_to_multiple_of=128)

    # Convert tf.data.Dataset to Model.prepare_tf_dataset
    # using Model to choose which columns you can use as input.
    train_dataset = model.prepare_tf_dataset(
        tokenized_datasets["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    validation_dataset = model.prepare_tf_dataset(
        tokenized_datasets["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    generation_dataset = model.prepare_tf_dataset(
        tokenized_datasets["validation"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=generation_data_collator
    )

    # Initialize optimizer and loss
    # Note: Most Transformers models compute loss internally
    # We can train on this as our loss value simply by not specifying a loss when we compile().
    optimizer = AdamWeightDecay(learning_rate=LEARNING_RATE,
                                weight_decay_rate=WEIGHT_DECAY)
    model.compile(optimizer=optimizer)

    def metric_fn(eval_predictions):
        preds, labels = eval_predictions
        if isinstance(preds, tuple): preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = metric.compute(predictions=[decoded_preds], references=[decoded_labels], tokenize='ja-mecab ')
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    metric_callback = KerasMetricCallback(
        metric_fn, eval_dataset=generation_dataset, predict_with_generate=True, use_xla_generation=True
    )

    es = EarlyStopping(monitor='bleu',
                    mode='max',
                    patience=5,
                    verbose=1,
                    restore_best_weights=True)


    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[metric_callback, es],
        verbose=1)
    )
    print(time.time() - start_time)
    print("/n⏰ Time to train:  {time.time() - start_time} seconds")

    save_model(model)

    print("/n✅ Woohoo! Model Saved! 😎 ")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by renaming the columns and dropping the English column.
    # """
    df.drop(columns=['#英語(原文)'], inplace=True)
    df.rename(columns={"#日本語(原文)": INPUT_COL_NAME, "#やさしい日本語": LABEL_COL_NAME}, inplace=True)

    print("\n✅ Data Cleaned!")

    return df


def train_val_test_split(df: pd.DataFrame) -> DatasetDict:
    """
    Split the data into Train : Validatin : Test as 80 : 10 : 10
    Then convert and return as Datasets
    """
    train, test = train_test_split(df, test_size=0.2)
    test, val = train_test_split(test, test_size=0.5)

    datasets = DatasetDict({
        "train": Dataset.from_dict(train),
        "validation": Dataset.from_dict(val),
        "test": Dataset.from_dict(test)
    })

    print("\n✅ Data Splitted!")

    return datasets



if __name__ == '__main__':
    preprocess_and_train()
