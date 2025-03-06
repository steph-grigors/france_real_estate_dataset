import glob
import os
import time
import pickle

from colorama import Fore, Style
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
from google.cloud import storage

from real_estate_parquet.params import *

def save_results(params: dict, metrics: dict, model_type = MODEL_TYPE) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")


    # Save params locally
    if params is not None:
        if model_type == 'xgboost':
            params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", 'xgboost', timestamp + ".pickle")
        elif model_type == 'keras':
            params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", 'keras', timestamp + ".pickle")
        else:
            raise Exception(Fore.RED + f"⚠️ Please select a valid model_type" + Style.RESET_ALL)

        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        if model_type == 'xgboost':
            metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", 'xgboost', timestamp + ".pickle")
        elif model_type == 'keras':
            metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", 'keras', timestamp + ".pickle")
        else:
            raise Exception(Fore.RED + f"⚠️ Please select a valid model_type" + Style.RESET_ALL)

        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)


    print("✅ Results saved locally")


def save_model(model = None, model_type = MODEL_TYPE) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    if model_type == 'xgboost':
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "xgboost", f"{timestamp}.json")
        model.save_model(model_path)
    elif model_type == 'keras':
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "keras", f"{timestamp}.h5")
        model.save(model_path)
    else:
        raise Exception(Fore.RED + f"⚠️ Please select a valid model_type" + Style.RESET_ALL)

    print("✅ Model saved locally")

    # Save model to gcs
    if MODEL_TARGET == "gcs":

        model_filename = "/".join(os.path.split(model_path)[-2:])
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None


def load_model(model_type = MODEL_TYPE):
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'

    Return None (but do not Raise) if no model is found

    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], "GPU")


    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        if model_type == 'xgboost':
            # Get the latest model version name by the timestamp on disk
            local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", "xgboost")
        elif model_type == 'keras':
            local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models", "keras")

        local_model_paths = glob.glob(f"{local_model_directory}/*")


        if not local_model_paths:
            print("⚠️ No models found!")
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        print(f"Most recent model path: {most_recent_model_path_on_disk}")

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        if model_type == 'xgboost':
            latest_model = xgb.Booster().load_model(most_recent_model_path_on_disk)
        elif model_type == 'keras':
            latest_model = keras.models.load_model(most_recent_model_path_on_disk)
        else:
            return None

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            if model_type == 'xgboost':
                latest_model = xgb.Booster().load_model(latest_model_path_to_save)

            elif model_type == 'keras':
                latest_model = keras.models.load_model(latest_model_path_to_save)


            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None
