import os
import time
import mlflow

from tensorflow.keras.models import load_model

def save_model(model):
    print(f"{model} my modellllll {type(model)}")
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if os.environ.get("MODEL_TARGET") == "mlflow":

        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=mlflow_experiment)

        with mlflow.start_run():

            # if params is not None:
            #     mlflow.log_params(params)

            # if metrics is not None:
            #     mlflow.log_metrics(metrics)

            if model is not None:
                mlflow.keras.log_model(keras_model=model,
                                       artifact_path="model",
                                       keras_module="tensorflow.keras",
                                       registered_model_name=mlflow_model_name)

        print("\n✅ Model saved to mlflow!")

        return None

    print("\n Save model to local disk...")

    # if params is not None:
    #     params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
    #     print(f"- params path: {params_path}")
    #     with open(params_path, "wb") as file:
    #         pickle.dump(params, file)

    # if metrics is not None:
    #     metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
    #     print(f"- metrics path: {metrics_path}")
    #     with open(metrics_path, "wb") as file:
    #         pickle.dump(metrics, file)

    if model is not None:
        model_path = os.path.join("simplyJapanese", 'data', "4_MainModel")
        model.save(model_path)

    print("\n✅ Model saved locally!")

    return None


def load_model():
    """
    Load the latest model!
    """
    if os.environ.get("MODEL_TARGET") == "mlflow":
        stage = "Production"

        print(f"/n Loading model from mlflow from {stage} stage")

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        model_uri = f"models:/{mlflow_model_name}/{stage}"
        print(f"- uri: {model_uri}")

        try:
                model = mlflow.keras.load_model(model_uri=model_uri)
                print("\n✅ Model loaded from mlflow!")
        except:
            print(f"\n❌ No model in stage {stage} on mlflow...")
            return None

    if os.environ.get("MODEL_TARGET") == "local":
        model_path = os.path.join("simplyJapanese", 'data', "4_MainModel")
        model = load_model(model_path)
        print("\n✅ Model loaded from disk!")

    return model
